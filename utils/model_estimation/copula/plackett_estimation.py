from utils.calc_var_ABC import VaRCalculationMethod
from copulas.plackett.plackett import copula_density
from copulas.plackett.opti import PlackettCopulaOptimizer


class PlackettCopulaVaR(VaRCalculationMethod):
    def __init__(self, estimation_method):
        # Store the estimation method instance (either MSMEstimation or GarchEstimation)
        self.estimation_method = estimation_method

    @staticmethod
    def unpack_copula_params(copula_params):
        # Extract rho values
        nu = copula_params

        return nu, None

    @staticmethod
    def copula_or_correl_params_insample(marginals, densities):
        # Initialize the optimizer
        p_optimizer = PlackettCopulaOptimizer(marginals, densities)

        # Run the optimizer for multiple nu values and retrieve the best result
        best_p_params = p_optimizer.optimize()

        return best_p_params

    @staticmethod
    def copula_integrations_params(best_p_params):
        """
        Retrieve the copula params
        """
        # Extract the upper triangle of the correlation matrix (excluding the diagonal)
        nu = best_p_params['theta']

        return nu

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
    def copula_density(cdf, nu, **kwargs):
        return copula_density(cdf, nu)

    def integration_params_retrieval(self, *args, **kwargs):
        return self.estimation_method.integration_params_retrieval(*args, **kwargs)
