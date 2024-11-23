from utils.calc_var_ABC import VaRCalculationMethod, SharedCacheCopulaMSMVaR
from markov_switching_multifractal.opti import Optimizer as MsmOptimizer
from markov_switching_multifractal.calc_marginals import calc_marginals, calc_densities, calc_forecasts
from utils.calc_integral.integration_functions.msm_integration_function import integrated_function
import numpy as np
import pandas as pd
from itertools import product


class MSMEstimation(VaRCalculationMethod):
    """
    Base class for models using the MSM estimation framework.
    Implements the MSM-specific logic for parameter estimation and marginal calculations.
    """

    @staticmethod
    def model_params_insample(in_sample_dict, k):
        """
        Perform MSM-specific optimization for each ticker's returns and store results in a dictionary.

        Parameters:
        in_sample_dict: dict
            A dictionary where keys are ticker symbols and values are in-sample returns.
        k: int
            The number of components for the MsmOptimizer.

        Returns:
        results_dict: dict
            A dictionary where keys are ticker symbols and values are dictionaries containing:
                - optimal_params: The optimized parameters for the model.
        """
        results_dict = {}
        for ticker, returns in in_sample_dict.items():
            # Check if the result is already cached
            cache_key = (ticker, k)
            if cache_key in SharedCacheCopulaMSMVaR.cache:
                results_dict[ticker] = SharedCacheCopulaMSMVaR.cache[cache_key]
                continue  # Skip optimization if cached

            # Initialize the MSM optimizer for the specific index (ticker)
            optimizer = MsmOptimizer(returns, k)
            optimal_params = optimizer.optimize()

            # Store the results
            m_0, sig, b, gamma = optimal_params[0], optimal_params[3], optimal_params[1], optimal_params[2]
            results_dict[ticker] = {
                'optimal_params': {'m_0': m_0, 'sig': sig, 'b': b, 'gamma': gamma}
            }

            # Cache the result
            SharedCacheCopulaMSMVaR.cache[cache_key] = results_dict[ticker]

        return results_dict

    @staticmethod
    def calculate_marginals_and_densities_in_sample(in_sample_dict, in_sample_params, k):
        """
        Calculate marginals for each ticker using optimized parameters from in_sample_params and returns from in_sample_dict.
        Based on the observed densities and marginals from in sample data we fit the copula params

        Parameters:
        in_sample_dict: dict
            A dictionary where keys are ticker symbols and values are the in-sample returns for each ticker.
        k: int
            The number of components for the MsmOptimizer.
        in_sample_params: dict
            A dictionary where keys are ticker symbols and values contain optimized parameters.

        Returns:
        marginals_array: np.ndarray
            A single stacked array of marginals for all tickers.
        densities_array: np.ndarray
            A single stacked array of densities for all tickers.
        vol_states_dict: dict
            A dictionary where keys are ticker symbols and values are the vol_states for each ticker.
        """
        marginals_list = []
        densities_list = []
        vol_states_dict = {}

        for ticker, params in in_sample_params.items():
            # Cache key specific to marginals/densities/vol_states (not the same as model_params_insample)
            cache_key = (ticker, f"marginals_{k}")
            if cache_key in SharedCacheCopulaMSMVaR.cache:
                cached_result = SharedCacheCopulaMSMVaR.cache[cache_key]
                marginals_list.append(cached_result['marginals'].reshape(-1, 1))
                densities_list.append(cached_result['densities'].reshape(-1, 1))
                vol_states_dict[ticker] = cached_result['vol_states']
                continue

            # Retrieve optimized parameters from in_sample_params
            m_0 = params['optimal_params']['m_0']
            sig = params['optimal_params']['sig']
            b = params['optimal_params']['b']
            gamma = params['optimal_params']['gamma']
            returns = in_sample_dict[ticker]

            # Calculate marginals, densities, and vol_states
            marginals, eps, vol_states = calc_marginals(k, m_0, sig, b, gamma, returns)
            densities = calc_densities(k, m_0, sig, b, gamma, returns)

            # Cache marginals, densities, and vol_states separately
            SharedCacheCopulaMSMVaR.cache[cache_key] = {
                'marginals': marginals,
                'densities': densities,
                'vol_states': vol_states
            }

            # Store in lists and dictionary
            marginals_list.append(marginals.reshape(-1, 1))
            densities_list.append(densities.reshape(-1, 1))
            vol_states_dict[ticker] = vol_states

        marginals_array = np.hstack(marginals_list)
        densities_array = np.hstack(densities_list)

        # Convert the dictionaries to arrays (Numba-compatible)
        vol_states_array = np.array([vol_states_dict[ticker] for ticker in vol_states_dict])

        return marginals_array, densities_array, vol_states_array


    def integration_params_retrieval(self, dim, rolling_windows_dict, in_sample_params, num_points, vol_state_array):

        k = int(np.sqrt(vol_state_array.shape[1]))

        forecasts_array = self.forecasts_array(rolling_windows_dict, in_sample_params, k)
        forecasts_by_states, unique_vol_states = self.sum_forecast_by_state(vol_state_array, forecasts_array)
        densities, x_values, step_size = self.compute_normal_densities(unique_vol_states, num_points)
        vol_combinations = self.create_vol_combinations(unique_vol_states)
        forecasts = self.compute_forecast_combinations(forecasts_by_states)

        integrations_params_t = forecasts_by_states, forecasts
        integrations_params_static = unique_vol_states
        grids_generations_params = densities, x_values, step_size, vol_combinations

        return integrations_params_t, integrations_params_static, grids_generations_params

    @staticmethod
    def forecasts_array(rolling_windows_dict, in_sample_params, k):
        """
        Calculate forecasted volatility states for each ticker and rolling window.

        Parameters:
        - in_sample_dict: dict
            A dictionary where keys are ticker symbols and values are in-sample returns.
        - rolling_windows_dict: dict
            A dictionary where keys are dates and values are dictionaries of rolling window returns.
        - k: int
            The number of components for the MsmOptimizer.
        - in_sample_params: dict
            A dictionary where keys are ticker symbols and values contain optimized parameters.

        Returns:
        - forecasts_states_dict: dict
            A dictionary where keys are ticker symbols and values are DataFrames containing forecasted states
            for each rolling window date (dates as index, and one column for each state).
        """
        forecasts_states_dict = {}

        # Loop through each ticker and get the optimized parameters
        for ticker, params in in_sample_params.items():
            # Retrieve optimized parameters for the current ticker
            m_0 = params['optimal_params']['m_0']
            sig = params['optimal_params']['sig']
            b = params['optimal_params']['b']
            gamma = params['optimal_params']['gamma']

            # Get the number of states (k)
            num_states = 2**k  # Number of states

            # Initialize an empty DataFrame with the correct number of columns (one for each state)
            forecasted_states_df = pd.DataFrame(columns=[f"State_{i}" for i in range(1, num_states+1)])

            # Loop through each rolling window (date) and calculate forecasted states
            for date, ticker_dict in rolling_windows_dict.items():
                # Cache key specific to forecasts for each date and ticker
                cache_key = (date, f"forecasts_{k}", ticker)
                returns = ticker_dict[ticker]  # Get the rolling window returns for the ticker

                # Check if forecasts are already cached
                if cache_key in SharedCacheCopulaMSMVaR.cache:
                    cached_result = SharedCacheCopulaMSMVaR.cache[cache_key]
                    forecasted_states = cached_result['forecasts_states']
                else:
                    # Calculate forecasts (volatility states) for the ticker based on returns and optimized parameters
                    forecasted_states = calc_forecasts(k, m_0, sig, b, gamma, returns)

                    # Cache the forecasts_states to avoid recalculating
                    SharedCacheCopulaMSMVaR.cache[cache_key] = {
                        'forecasts_states': forecasted_states,
                    }

                # Add the forecasted states for this date to the DataFrame
                forecasted_states_df.loc[date] = forecasted_states

            # Store the DataFrame in the dictionary with ticker as the key
            forecasts_states_dict[ticker] = forecasted_states_df

            forecasts_array = np.array([forecasts_states_dict[ticker].values for ticker in forecasts_states_dict])

        return forecasts_array

    @staticmethod
    def sum_forecast_by_state(vol_state_array, forecasts_array, tol=1e-6):
        """
        Summarizes the forecasts_array values by summing all values in forecasts_array[i, k, :]
        where the respective values in vol_state_array[i, :] are almost the same (within a given tolerance).

        Args:
            vol_state_array (np.ndarray): A 2D array with dimensions (dim, k) representing the state values.
            forecasts_array (np.ndarray): A 3D array with dimensions (dim, N, k) representing forecast values.
            tol (float): The tolerance level to consider values in vol_state_array[i, :] as the same.

        Returns:
            tuple: A tuple containing:
                - np.ndarray: A 3D array where forecasts are summed based on the approximately unique values in vol_state_array.
                - np.ndarray: A 2D array of the unique (or rounded) state values used for the summation.
        """
        dim = vol_state_array.shape[0]
        N = forecasts_array.shape[1]

        # Prepare the results for forecasts and the unique vol states
        summed_forecasts = []
        unique_vol_states = []

        for i in range(dim):
            rounded_states = np.round(vol_state_array[i, :] / tol) * tol
            unique_states, inverse_idx = np.unique(rounded_states, return_inverse=True)

            # Sum forecasts based on the unique (rounded) state
            summed_forecast_i = np.zeros((N, len(unique_states)))
            for n in range(N):
                summed_forecast_i[n] = np.array(
                    [forecasts_array[i, n, :][inverse_idx == idx].sum() for idx in range(len(unique_states))]
                )

            # Store the results for this 'i'
            summed_forecasts.append(summed_forecast_i)
            unique_vol_states.append(unique_states)

        # Convert lists to numpy arrays
        summed_forecasts = np.array(summed_forecasts)
        unique_vol_states = np.array(unique_vol_states)

        transposed_summed_forecasts = summed_forecasts.transpose(1, 0, 2)

        return transposed_summed_forecasts, unique_vol_states

    @staticmethod
    def compute_combinations_product(array):
        """
        Compute the forecasted products of unique volatility states for each time step.

        This function takes a 3D array representing unique volatility states and computes
        the product of all possible combinations of the states across the specified dimension `dim`.
        For each time step `t` (from 1 to `N`), it generates all combinations of `q` unique states across
        `dim` dimensions and calculates the product for each combination, resulting in `q ** dim` products.

        Parameters:
        - array: np.ndarray, shape (dim, N, q), where:
            - dim: the number of volatility components
            - N: the number of time steps
            - q: the number of unique volatility states for each component

        Returns:
        - result: np.ndarray, shape (N, q ** dim), containing the products of all combinations
          of the unique volatility states for each time step.
        """
        dim, N, q = array.shape
        result = np.empty((N, q ** dim))  # Initialize the result array with shape (N, q**dim)

        for i in range(N):
            # For each i, get all combinations of the q values across dim
            combinations = product(*array[:, i, :])  # Generates (q**dim) combinations of length `dim`

            # Calculate the product for each combination
            result[i] = [np.prod(comb) for comb in combinations]

        return result

    @staticmethod
    def compute_normal_densities(unique_vol_states_array, num_points, x_min=-5, x_max=5):
        """
        Compute the normal density for each standard deviation in unique_vol_states_array
        with a varying density of points: twice as many points between -1 and 1,
        more points between -2.5 and 2.5, and fewer points beyond.

        Parameters:
        - unique_vol_states_array: np.ndarray, shape (dim, q), where each element is a standard deviation.
        - num_points: int, total number of points to evaluate across the whole range.
        - x_min: float, minimum x-value for evaluation.
        - x_max: float, maximum x-value for evaluation.

        Returns:
        - densities: np.ndarray, shape (dim, q, num_points), containing the normal densities.
        - x_values: np.ndarray, shape (num_points,), the x values where densities are evaluated.
        - step_size: np.ndarray, shape (num_points,), the difference between consecutive x values.
        """
        dim, q = unique_vol_states_array.shape

        # Calculate distribution of points across the regions
        outer_points = num_points // 4  # Points for [-5, -2.5] and [2.5, 5]
        middle_points = num_points // 7  # Points for [-2.5, -1] and [1, 2.5]
        central_points = num_points - 2 * outer_points - 2 * middle_points  # Remaining points for [-1, 1]

        # Create x values for each range
        x_outer_left = np.linspace(x_min, -2.5, outer_points, endpoint=False)
        x_middle_left = np.linspace(-2.5, -1, middle_points, endpoint=False)
        x_central = np.linspace(-1, 1, central_points, endpoint=False)
        x_middle_right = np.linspace(1, 2.5, middle_points, endpoint=False)
        x_outer_right = np.linspace(2.5, x_max, outer_points, endpoint=True)

        # Combine all x values into one array, ensuring total length matches `num_points`
        x_values = np.concatenate([x_outer_left, x_middle_left, x_central, x_middle_right, x_outer_right])

        # Compute step sizes between each consecutive x value
        step_size = np.diff(x_values, prepend=x_values[0])
        step_size[0] = step_size[1]

        # Initialize the result array to hold densities for each (i, j) and each x_value
        densities = np.zeros((dim, q, num_points))

        for i in range(dim):
            for j in range(q):
                std_dev = unique_vol_states_array[i, j]
                # Compute normal density for each x given std_dev
                densities[i, j, :] = (1 / (np.sqrt(2 * np.pi) * std_dev)) * np.exp(-0.5 * (x_values / std_dev) ** 2)

        return densities, x_values, step_size

    @staticmethod
    def compute_normal_densities_uniform(unique_vol_states_array, num_points, x_min=-5, x_max=5):
        """
        Compute the normal density for each standard deviation in unique_vol_states_array
        with equally spaced x-values across the entire range.

        Parameters:
        - unique_vol_states_array: np.ndarray, shape (dim, q), where each element is a standard deviation.
        - num_points: int, total number of points to evaluate across the whole range.
        - x_min: float, minimum x-value for evaluation.
        - x_max: float, maximum x-value for evaluation.

        Returns:
        - densities: np.ndarray, shape (dim, q, num_points), containing the normal densities.
        - x_values: np.ndarray, shape (num_points,), the x values where densities are evaluated.
        - step_size: float, the difference between consecutive x values (constant for equally spaced points).
        """
        dim, q = unique_vol_states_array.shape

        # Create equally spaced x values across the range [x_min, x_max]
        x_values = np.linspace(x_min, x_max, num_points)

        # Compute step sizes between each consecutive x value
        step_size = np.diff(x_values, prepend=x_values[0])

        # Initialize the result array to hold densities for each (i, j) and each x_value
        densities = np.zeros((dim, q, num_points))

        for i in range(dim):
            for j in range(q):
                std_dev = unique_vol_states_array[i, j]
                # Compute normal density for each x given std_dev
                densities[i, j, :] = (1 / (np.sqrt(2 * np.pi) * std_dev)) * np.exp(-0.5 * (x_values / std_dev) ** 2)

        return densities, x_values, step_size

    @staticmethod
    def create_vol_combinations(unique_vol_states):
        """
        Create an array with all index combinations for a 2D array of shape (dim, q).

        Parameters:
        - dim: int, the number of rows (dimensions).
        - q: int, the number of elements in each row (range of indices is 0 to q-1).

        Returns:
        - index_combinations: np.ndarray of shape (q^dim, dim), containing all possible index combinations.
        """

        dim, q = unique_vol_states.shape

        # Generate a meshgrid of indices for each dimension
        grids = np.meshgrid(*[np.arange(q) for _ in range(dim)], indexing='ij')

        # Stack and reshape to get all combinations
        index_combinations = np.stack(grids, axis=-1).reshape(-1, dim)

        return index_combinations

    @staticmethod
    def compute_forecast_combinations(summed_forecasts):
        """
        Compute the product of all combinations of the q values for each N for any dimension.

        Parameters:
        - summed_forecasts: np.ndarray of shape (dim, N, q)

        Returns:
        - result: np.ndarray of shape (N, q^dim)
        """
        N, dim, q = summed_forecasts.shape  # Get the number of dimensions (dim), N, and q

        # Create the result array, which will store the product of all combinations for each N
        result = np.zeros((N, q ** dim))

        # Loop over each N
        for n in range(N):
            # Extract the dim rows for the current n, we have summed_forecasts[:, n, :]
            forecast_rows = [summed_forecasts[n, d, :] for d in range(dim)]

            # Create a meshgrid of all combinations of the rows (for arbitrary dimensions)
            product_combinations = np.array(np.meshgrid(*forecast_rows)).T.reshape(-1, dim)

            # Compute the product along the rows of the combinations
            result[n] = np.prod(product_combinations, axis=1)

        return result

    @staticmethod
    def create_params(params_creation, estimated_vol):
        N = estimated_vol.shape[0]
        p = params_creation.shape[0]
        q = params_creation.shape[1]

        # Correcting np.zeros to accept a tuple for shape
        result = np.zeros((N, p, q+2))

        for i in range(N):
            # Stack the estimated_vol for each i across p rows
            stacked = np.tile(estimated_vol[i], (p, 1))

            # Concatenate params_creation with stacked array
            result[i] = np.hstack((params_creation, stacked))

        return result

    def copula_or_correl_params_insample(self):
        pass

    def density_function(self):
        pass

    @staticmethod
    def integrated_function(
            grids,
            step_sizes,
            copula_params,
            integrations_params_i,
            integrations_params_static,
            copula_density,
            unpack_copula_params
    ):
        return integrated_function(
            grids,
            step_sizes,
            copula_params,
            integrations_params_i,
            integrations_params_static,
            copula_density,
            unpack_copula_params
        )
