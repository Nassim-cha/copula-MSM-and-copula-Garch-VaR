from copulas.plackett.inference_for_margins import MarginalLikelihoodInferencePlackett
from scipy.optimize import minimize
import numpy as np


class PlackettCopulaOptimizer:
    def __init__(self, marginals, densities, tol=1e-9, max_iter=5000):
        """
        Initialize the optimizer with the marginals and densities.

        Parameters:
        marginals: array-like
            A 2D array where each row is a pair of marginals (N x 2).
        densities: array-like
            A 2D array where each row is a pair of densities (N x 2).
        tol: float, optional
            The tolerance for convergence.
        max_iter: int, optional
            Maximum number of iterations.
        """
        self.marginals = np.array(marginals)
        self.densities = np.array(densities)
        self.N = self.marginals.shape[0]  # Number of observations
        self.objective_func = self.negative_log_likelihood
        self.tol = tol
        self.max_iter = max_iter

    def negative_log_likelihood(self, theta):
        """
        Compute the negative log-likelihood for the given Plackett copula parameter theta.
        """
        # Create an instance of the MarginalLikelihoodInferencePlackett class
        marginal_inference = MarginalLikelihoodInferencePlackett(
            marginals=self.marginals,
            densities=self.densities,
            theta=theta
        )

        # Compute the total log-likelihood using the inference class
        total_log_likelihood = marginal_inference.compute_marginal_likelihood()

        return -total_log_likelihood

    def optimize(self, theta_range=None, method='L-BFGS-B'):
        """
        Optimize the Plackett copula by minimizing the negative log-likelihood over a range of theta values.

        Parameters:
        theta_range: list or array-like, optional
            List or array of initial theta values to try. Default is a range from 0.5 to 50.
        method: str, optional
            The optimization method to be used by scipy.optimize.minimize.

        Returns:
        best_result: dict
            The best result with the lowest negative log-likelihood.
        """
        # Set default theta range if not provided
        if theta_range is None:
            theta_range = np.linspace(0.5, 50, 10)  # Default range from 0.5 to 50

        best_nll = np.inf
        best_theta = None

        # Create bounds: theta > 0
        bounds = [(0.1, None)]  # Plackett parameter theta must be positive

        for initial_theta in theta_range:
            print(f"Optimizing for initial theta = {initial_theta}")

            # Optimize theta using minimize
            res = minimize(
                fun=self.negative_log_likelihood,  # Objective function for theta optimization
                x0=[initial_theta],  # Initial guess for theta
                method=method,  # Optimization method
                bounds=bounds,  # Bounds for theta
                tol=self.tol,  # Tolerance for convergence
                options={'maxiter': self.max_iter}  # Maximum iterations
            )

            optimized_theta = res.x[0]
            final_nll = res.fun

            print(f"Optimized theta = {optimized_theta}, Negative Log-Likelihood = {final_nll}")

            # Track the best result
            if final_nll < best_nll:
                best_nll = final_nll
                best_theta = optimized_theta

        best_result = {
            'theta': best_theta,
            'nll': best_nll,
            'optimized_params': best_theta
        }

        return best_result


if __name__ == "__main__":
    # Define marginals and densities (for N pairs of observations)
    marginals = np.random.rand(1000, 2)  # 1000 samples, 2 variables
    densities = np.random.rand(1000, 2)  # Densities corresponding to marginals

    # Initialize the PlackettCopulaOptimizer
    optimizer = PlackettCopulaOptimizer(marginals, densities)

    # Optimize the Plackett copula parameter theta
    best_result = optimizer.optimize()

    print(f"Best theta: {best_result['theta']}")
    print(f"Best negative log-likelihood: {best_result['nll']}")
