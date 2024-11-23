from utils.calc_var_ABC import VaRCalculationMethod
from copulas.gaussian.gaussian import copula_density
from copulas.gaussian.opti import GaussianCopulaOptimizer
import numpy as np


class GaussianCopulaVaR(VaRCalculationMethod):
    def __init__(self, estimation_method):
        # Store the estimation method instance (either MSMEstimation or GarchEstimation)
        self.estimation_method = estimation_method

    @staticmethod
    def unpack_copula_params(copula_params):
        # Extract rho values
        rho = copula_params

        # Reconstruct the correlation matrix from rho
        n = int((1 + np.sqrt(1 + 8 * len(rho))) / 2)  # Solve for matrix size: n(n-1)/2 = len(rho)
        corr_matrix = np.eye(n)  # Initialize an identity matrix
        corr_matrix[np.triu_indices(n, k=1)] = rho  # Fill the upper triangle with rho values
        corr_matrix[np.tril_indices(n, k=-1)] = rho  # Fill the lower triangle to make it symmetric

        return None, corr_matrix

    @staticmethod
    def copula_or_correl_params_insample(marginals, densities):
        # Initialize the optimizer
        g_optimizer = GaussianCopulaOptimizer(marginals, densities)

        # Run the optimizer for multiple nu values and retrieve the best result
        best_g_params = g_optimizer.optimize()

        return best_g_params

    @staticmethod
    def copula_integrations_params(best_g_params):
        """
        Retrieve the copula params
        """
        # Extract the upper triangle of the correlation matrix (excluding the diagonal)
        corr_matrix = best_g_params['corr_matrix']
        rho = corr_matrix[np.triu_indices_from(corr_matrix, k=1)]  # Extract off-diagonal elements as a 1D array

        return rho

    # Implement all abstract methods
    def calculate_marginals_and_densities_in_sample(self, *args, **kwargs):
        return self.estimation_method.calculate_marginals_and_densities_in_sample(*args, **kwargs)

    def density_function(self, *args, **kwargs):
        return self.estimation_method.density_function(*args, **kwargs)

    def forecasts_array(self, *args, **kwargs):
        return self.estimation_method.forecasts_array(*args, **kwargs)

    def model_params_insample(self, *args, **kwargs):
        return self.estimation_method.model_params_insample(*args, **kwargs)

    def sum_forecast_by_state(self, *args, **kwargs):
        return self.estimation_method.sum_forecast_by_state(*args, **kwargs)

    def compute_normal_densities(self, *args, **kwargs):
        return self.estimation_method.compute_normal_densities(*args, **kwargs)

    def create_vol_combinations(self, *args, **kwargs):
        return self.estimation_method.create_vol_combinations(*args, **kwargs)

    def compute_forecast_combinations(self, *args, **kwargs):
        return self.estimation_method.compute_forecast_combinations(*args, **kwargs)

    def integrated_function(self, *args, **kwargs):
        return self.estimation_method.integrated_function(*args, **kwargs)

    @staticmethod
    def copula_density(cdf, corr_matrix, **kwargs):
        return copula_density(cdf, corr_matrix)

    def integration_params_retrieval(self, *args, **kwargs):
        return self.estimation_method.integration_params_retrieval(*args, **kwargs)
