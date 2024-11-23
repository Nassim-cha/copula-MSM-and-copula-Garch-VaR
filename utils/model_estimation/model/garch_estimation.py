from utils.calc_var_ABC import VaRCalculationMethod, SharedCacheCopulaGarchVaR
from garch.opti import GarchOptimizer
from garch.estimation import ProbEstimation
from garch.forecast import calc_forecast
import numpy as np
import pandas as pd
from utils.utils import norm_cdf_array, norm_pdf_array
from utils.calc_integral.integration_functions.garch_integration_function import integrated_function


class GarchEstimation(VaRCalculationMethod):
    """
    Base class for models using the MSM estimation framework.
    Implements the MSM-specific logic for parameter estimation and marginal calculations.
    """

    @staticmethod
    def model_params_insample(in_sample_dict):
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
            cache_key = (ticker)
            if cache_key in SharedCacheCopulaGarchVaR.cache:
                results_dict[ticker] = SharedCacheCopulaGarchVaR.cache[cache_key]
                continue  # Skip optimization if cached

            # Initialize the MSM optimizer for the specific index (ticker)
            optimizer = GarchOptimizer(returns)
            # Run the optimizer over different (p, q) combinations
            best_pq, best_params, best_result, best_bic = optimizer.optimize()

            # Store the results
            results_dict[ticker] = {
                'optimal_params': {'best_pq': best_pq, 'best_params': best_params, 'best_bic': best_bic}
            }

            # Cache the result
            SharedCacheCopulaGarchVaR.cache[cache_key] = results_dict[ticker]

        return results_dict

    @staticmethod
    def calculate_marginals_and_densities_in_sample(in_sample_dict, in_sample_params):
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
            cache_key = (ticker, f"marginals")
            if cache_key in SharedCacheCopulaGarchVaR.cache:
                cached_result = SharedCacheCopulaGarchVaR.cache[cache_key]
                marginals_list.append(cached_result['marginals'].reshape(-1, 1))
                densities_list.append(cached_result['densities'].reshape(-1, 1))
                vol_states_dict[ticker] = cached_result['vol_states']
                continue

            # Retrieve optimized parameters from in_sample_params
            omega = params['optimal_params']['best_params'][0]
            p = params['optimal_params']['best_pq'][0]
            alpha_vect = params['optimal_params']['best_params'][1:p+1]
            beta_vect = params['optimal_params']['best_params'][p+1:]
            returns = in_sample_dict[ticker]

            likelihood = ProbEstimation(returns, omega, alpha_vect, beta_vect)

            eps = likelihood.calculate_eps_t()

            marginals = norm_cdf_array(eps)
            densities = norm_pdf_array(eps)

            # Cache marginals, densities, and vol_states separately
            SharedCacheCopulaGarchVaR.cache[cache_key] = {
                'marginals': marginals,
                'densities': densities,
            }

            # Store in lists and dictionary
            marginals_list.append(marginals.reshape(-1, 1))
            densities_list.append(densities.reshape(-1, 1))

        marginals_array = np.hstack(marginals_list)
        densities_array = np.hstack(densities_list)

        return marginals_array, densities_array, None

    def forecasts_array(self):
        pass

    def sum_forecast_by_state(self):
        pass

    def copula_or_correl_params_insample(self):
        pass

    def density_function(self):
        pass

    def integration_params_retrieval(self, dim, rolling_windows_dict, in_sample_params, num_points, vol_state_array):

        densities, x_values, step_size = self.compute_normal_densities(dim, num_points)

        params = np.zeros((1, dim))

        grids_generations_params = densities, x_values, step_size, params

        forecasts = self.compute_forecast(rolling_windows_dict, in_sample_params)

        integrations_params_static = None

        return forecasts, integrations_params_static, grids_generations_params

    @staticmethod
    def compute_normal_densities(dim, num_points, x_min=-5, x_max=5):
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

        # Calculate distribution of points across the regions
        outer_points = num_points // 8  # Points for [-5, -2.5] and [2.5, 5]
        middle_points = num_points // 5  # Points for [-2.5, -1] and [1, 2.5]
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
        densities = np.ones((dim, 1, num_points))

        return densities, x_values, step_size

    def compute_forecast(self, rolling_windows_dict, in_sample_params):
        forecasts_states_dict = {}

        # Loop through each ticker and get the optimized parameters
        for ticker, params in in_sample_params.items():
            # Retrieve optimized parameters for the current ticker
            omega = params['optimal_params']['best_params'][0]
            p = params['optimal_params']['best_pq'][0]
            alpha_vect = params['optimal_params']['best_params'][1:p+1]
            beta_vect = params['optimal_params']['best_params'][p+1:]

            # Initialize an empty DataFrame with the correct number of columns (one for each state)
            forecasts_df = pd.DataFrame(columns=[f"Forecasts"])

            # Loop through each rolling window (date) and calculate forecasted states
            for date, ticker_dict in rolling_windows_dict.items():
                # Cache key specific to forecasts for each date and ticker
                cache_key = (date, f"forecasts", ticker)
                returns = ticker_dict[ticker]  # Get the rolling window returns for the ticker

                # Check if forecasts are already cached
                if cache_key in SharedCacheCopulaGarchVaR.cache:
                    cached_result = SharedCacheCopulaGarchVaR.cache[cache_key]
                    forecasts = cached_result['forecasts']
                else:
                    # Calculate forecasts (volatility states) for the ticker based on returns and optimized parameters
                    forecasts = calc_forecast(omega, alpha_vect, beta_vect, returns)

                    # Cache the forecasts_states to avoid recalculating
                    SharedCacheCopulaGarchVaR.cache[cache_key] = {
                        'forecasts_states': forecasts,
                    }

                # Add the forecasted states for this date to the DataFrame
                forecasts_df.loc[date] = forecasts

            # Store the DataFrame in the dictionary with ticker as the key
            forecasts_states_dict[ticker] = forecasts_df

            forecasts_array = np.array([forecasts_states_dict[ticker].values for ticker in forecasts_states_dict])

        return [forecasts_array.transpose(1, 0, 2).squeeze(-1)]

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
