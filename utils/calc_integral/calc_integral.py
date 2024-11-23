from utils.calc_integral.integration_algo import multi_integral_function
from utils.calc_integral.create_grids import create_nested_grid
from joblib import Parallel, delayed
import time
import numpy as np


def calc_grids_and_integrals_results(
        T,
        unique_var_values,
        unique_indices,
        num_points,
        dim,
        var_function,
        lower_bound,
        upper_bound,
        grids_generations_params,
        integrations_params_t,
        integrations_params_static,
        copula_params,
        integrated_function,
        copula_density,
        unpack_copula_params,
        weights
                                     ):
    """             forecasts_by_state=forecasts_by_state,
            unique_vol_states=unique_vol_states, grids_generations_params
    Precomputes grids and step sizes for each unique variance value, then calculates results in parallel.

    Parameters:
    - T: int
        Number of iterations in the outer loop, representing time steps.
    - unique_var_values: list or array
        Unique variance values for which grids and step sizes are generated.
    - unique_indices: list or array
        Array of indices mapping each `i` in T to an index in unique_var_values.
    - num_points: int
        Number of points in each dimension for the grid.
    - dim: int
        Number of dimensions for the grid.
    - var_function: function
        Function that provides the variance bounds for the innermost dimension.
    - lower_bound, upper_bound: float
        Integration bounds for the grid generation.
    - x_values: np.ndarray
        Array of 1D grid values for each dimension.
    - densities: np.ndarray
        Array of densities corresponding to each grid value in x_values.
    - params: np.ndarray
        Array of parameters used to dynamically select density indices.
    - N: int
        Number of integration steps.
    - estimated_vol: np.ndarray
        Array of estimated volatility values for each time step.
    - inv_cdf_params, copula_params: np.ndarray
        Arrays holding parameters for the copula and inverse CDF functions.

    Returns:
    - np.ndarray
        Array of calculated results for each time step T.
    """

    unique_values = len(unique_var_values)  # Number of unique variance values
    grids_list = []  # List to store grids for each unique variance value
    step_sizes_list = []  # List to store step sizes for each unique variance value

    # Timer for measuring grid generation time
    start_grid_time = time.time()

    # Loop to generate grids and step sizes for each unique variance value
    for idx in range(unique_values):
        var_val = unique_var_values[idx]  # Current unique variance value

        # Call function to generate grid points and step sizes for the current variance value
        grids, step_sizes = create_nested_grid(
            num_points=num_points,
            dim=dim,
            g=var_function,
            var=var_val,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            grids_generation_params=grids_generations_params,
            weights=weights
        )

        # Append generated grid points and step sizes to respective lists
        grids_list.append(grids)
        step_sizes_list.append(step_sizes)

    # End timer for grid generation and print elapsed time
    end_grid_time = time.time()
    print(f"Grid generation time: {end_grid_time - start_grid_time:.2f} seconds")

    # Timer for measuring calculation time
    start_calc_time = time.time()

    # Perform parallel calculations for each time step T
    var_result = calculate_results_parallel(
        T=T,
        unique_indices=unique_indices,
        grids_list=grids_list,
        step_sizes_list=step_sizes_list,
        integrations_params_t=integrations_params_t,
        integrations_params_static=integrations_params_static,
        copula_params=copula_params,
        integrated_function=integrated_function,
        copula_density=copula_density,
        unpack_copula_params=unpack_copula_params
    )

    # End timer for calculations and print elapsed time
    end_calc_time = time.time()
    print(f"Calculation time for calculate_results_parallel: {end_calc_time - start_calc_time:.2f} seconds")

    # Calculate and print total execution time
    total_time = end_calc_time - start_grid_time
    print(f"Total execution time: {total_time:.2f} seconds")

    return var_result  # Return the final calculated results


def calculate_result_for_i(
        i,
        unique_indices,
        grids_list,
        step_sizes_list,
        integrations_params_t,
        integrations_params_static,
        copula_params,
        integrated_function,
        copula_density,
        unpack_copula_params
                           ):
    """
    Calculates results for a single index `i` using precomputed grids and step sizes.

    Parameters:
    - i: int
        Index for the current time step.
    - unique_indices: list or array
        Array mapping each time step to an index in the precomputed grids and step sizes.
    - estimated_vol: np.ndarray
        Array of estimated volatility values for each time step.
    - grids_list: list of np.ndarray
        List of 2D arrays containing grid points for each unique variance.
    - step_sizes_list: list of np.ndarray
        List of 2D arrays containing step sizes for each unique variance.
    - copula_params: np.ndarray
        Parameters for the copula function.

    Returns:
    - result_i: float or np.ndarray
        Calculated result for index `i`.
    """
    cache_index = unique_indices[i]  # Index to access precomputed grid and step sizes
    grids = grids_list[cache_index].copy()  # Get and copy grid for current index
    step_sizes = step_sizes_list[cache_index].copy()  # Get and copy step sizes for current index
    integrations_params_i = [integrations_params_t[t][i] for t in range(len(integrations_params_t))]
    # Calculate result for index `i` using vectorized integration function
    result_i = multi_integral_function(
        grids=grids,
        step_sizes=step_sizes,
        integrated_function=integrated_function,
        copula_params=copula_params,
        integrations_params_i=integrations_params_i,
        integrations_params_static=integrations_params_static,
        copula_density=copula_density,
        unpack_copula_params=unpack_copula_params
    )

    return result_i  # Return the calculated result for index `i`


def calculate_results_parallel(
        T,
        unique_indices,
        grids_list,
        step_sizes_list,
        integrations_params_t,
        integrations_params_static,
        copula_params,
        integrated_function,
        copula_density,
        unpack_copula_params,
                               ):
    """

    Executes parallel calculations for each time step in T.

    Parameters:
    - T: int
        Total number of time steps.
    - N: int
        Number of integration steps.
    - unique_indices: list or array
        Array mapping each time step to an index in precomputed grids and step sizes.
    - estimated_vol: np.ndarray
        Array of estimated volatility values for each time step.
    - grids_list: list of np.ndarray
        List of 2D arrays containing grid points for each unique variance.
    - step_sizes_list: list of np.ndarray
        List of 2D arrays containing step sizes for each unique variance.
    - copula_params: np.ndarray
        Parameters for the copula function.

    Returns:
    - np.ndarray
        Array containing results for each time step.
    """
    # Use parallel processing to calculate results for each time step in range(T)
    result = Parallel(n_jobs=-1)(delayed(calculate_result_for_i)(
        i,
        unique_indices,
        grids_list,
        step_sizes_list,
        integrations_params_t,
        integrations_params_static,
        copula_params,
        integrated_function,
        copula_density,
        unpack_copula_params
    )
                                 for i in range(T))

    return np.array(result)  # Convert the list of results to a NumPy array and return
