import numpy as np
from numba import njit

# Function for the example dynamic bound
@njit
def var_function(previous_points, var, weights):
    """
    Example function for generating dynamic bounds in the innermost dimension.

    Parameters:
    - previous_points: np.ndarray
        Points from the previous dimensions.
    - var: float
        Variable used in determining the dynamic bound.

    Returns:
    - np.ndarray
        Calculated upper bound for the innermost dimension.
    """
    return (var - np.sum(previous_points * weights[1:])) / weights[0]


def multi_integral_function(
        grids,
        step_sizes,
        integrated_function,
        copula_params=None,
        integrations_params_i=None,
        integrations_params_static=None,
        copula_density=None,
        unpack_copula_params=None
):
    """
    Performs N multidimensional nested integrations using Numba for optimization in batches.
        grids=grids,
        step_sizes=step_sizes,
        integrated_function=integrated_function,
        copula_params=copula_params,
        integrations_params_i=integrations_params_i,
        integrations_params_static=integrations_params_static,
        copula_density=copula_density,
        unpack_copula_params=unpack_copula_params
    Parameters:
    - var: float
        Input variable for the dynamic bound function g.
    - dim: int
        Number of dimensions to integrate over.
    - g: function
        Function defining the dynamic bounds for the innermost dimension.
    - lower_bound, upper_bound: float
        Fixed bounds for the outer dimensions.
    - integrated_function: function
        The function that computes the integrand, which can be any function passed with corresponding parameters.
    - params: np.ndarray (2D array)
        2D array where each row contains the parameters needed for one integration.
        There are N rows, meaning N separate integrals are calculated.
    - num_points: int, default 1000
        Number of grid points per dimension.
    - batch_size: int, default 5
        Number of integrals to process per batch.

    Returns:
    - integral_results: np.ndarray
        Integrated results for each of the N integrals.
    """

    # Calculate the integral function for generated grids
    results = integrated_function(
        grids=grids,
        step_sizes=step_sizes,
        copula_params=copula_params,
        integrations_params_i=integrations_params_i,
        integrations_params_static=integrations_params_static,
        copula_density=copula_density,
        unpack_copula_params=unpack_copula_params
    )

    result = np.sum(results)

    return result