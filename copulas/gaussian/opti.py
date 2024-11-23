from copulas.gaussian.inference_for_margins import GaussianMarginalLikelihoodInference
from scipy.optimize import minimize
import numpy as np
from scipy.linalg import cholesky


class GaussianCopulaOptimizer:
    def __init__(self, marginals, densities, tol=1e-9, max_iter=5000):
        """
        Initialize the optimizer with the marginals and densities.

        Parameters:
        marginals: list of arrays
            A list where each element is a 1D array of marginals for a variable.
        densities: list of arrays
            A list where each element is a 1D array of densities for a variable.
        tol: float, optional
            The tolerance for convergence.
        max_iter: int, optional
            Maximum number of iterations.
        """
        self.marginals = np.array(marginals)
        self.densities = np.array(densities)
        self.N = self.marginals.shape[0]  # Number of observations
        self.dim = self.marginals.shape[1]  # Number of variables
        self.objective_func = self.negative_log_likelihood
        self.tol = tol
        self.max_iter = max_iter

    def negative_log_likelihood(self, corr_params):
        """
        Compute the negative log-likelihood for the given correlation parameters.
        """
        # Reconstruct the correlation matrix
        corr_matrix = self.construct_correlation_matrix(corr_params)

        # Ensure the correlation matrix is valid
        if np.isnan(corr_matrix).any() or np.isinf(corr_matrix).any():
            return 1e10  # Return a large value if invalid correlation matrix

        # Ensure the correlation matrix is positive-definite
        try:
            cholesky(corr_matrix)
        except np.linalg.LinAlgError:
            return 1e10  # Return a large number if the matrix is not positive-definite

        # Compute the log-likelihood for each observation
        marginal_inference = GaussianMarginalLikelihoodInference(
            marginals=self.marginals,
            densities=self.densities,
            correlations=corr_matrix
        )

        total_log_likelihood = marginal_inference.compute_marginal_likelihood()

        return -total_log_likelihood

    def construct_correlation_matrix(self, corr_params):
        """
        Construct a correlation matrix from the parameter vector.

        Parameters:
        corr_params: array-like
            The flattened lower triangular elements of the correlation matrix.

        Returns:
        corr_matrix: ndarray
            The reconstructed correlation matrix.
        """
        corr_matrix = np.eye(self.dim)
        idx = 0
        for i in range(self.dim):
            for j in range(i):
                corr_matrix[i, j] = corr_params[idx]
                corr_matrix[j, i] = corr_params[idx]
                idx += 1
        return corr_matrix

    def optimize(self, initial_corr=None, method='L-BFGS-B'):
        """
        Optimize the Gaussian copula by minimizing the negative log-likelihood.

        Parameters:
        initial_corr: array-like, optional
            Initial guess for the correlation parameters.
        method: str, optional
            The optimization method to be used by scipy.optimize.minimize.

        Returns:
        best_result: dict
            The best result with the lowest negative log-likelihood.
        """
        # If no initial correlation provided, initialize it with 0.5
        if initial_corr is None:
            initial_corr = np.full((self.dim * (self.dim - 1)) // 2, 0.5)

        # Create bounds: correlations between -0.99 and 0.99
        corr_bounds = [(-0.99, 0.99)] * ((self.dim * (self.dim - 1)) // 2)

        # Optimize correlation parameters using minimize
        res_corr = minimize(
            fun=self.negative_log_likelihood,  # Objective function for correlation optimization
            x0=initial_corr,  # Initial guess for correlation parameters
            method=method,  # Optimization method
            bounds=corr_bounds,  # Bounds for correlation parameters
            tol=self.tol,  # Tolerance for convergence
            options={'maxiter': self.max_iter}  # Maximum iterations
        )

        # Get the optimized correlation parameters
        optimized_corr_params = res_corr.x

        # Calculate the final negative log-likelihood
        final_nll = self.negative_log_likelihood(optimized_corr_params)

        # Construct the final correlation matrix
        best_corr_matrix = self.construct_correlation_matrix(optimized_corr_params)

        print(f"Final Negative Log-Likelihood: {final_nll}")
        print(f"Optimized correlation matrix: \n{best_corr_matrix}")

        best_result = {
            'corr_matrix': best_corr_matrix,
            'nll': final_nll,
            'optimized_params': optimized_corr_params
        }

        return best_result


if __name__ == "__main__":
    # Define marginals, densities, and correlation matrix
    marginals = np.random.rand(1000, 2)  # 1000 samples, 2 variables
    densities = np.random.rand(1000, 2)  # Densities corresponding to marginals

    # Initialize the GaussianCopulaOptimizer
    optimizer = GaussianCopulaOptimizer(marginals, densities)

    # Optimize the Gaussian copula parameters
    best_result = optimizer.optimize()

    print(f"Best correlation matrix: \n{best_result['corr_matrix']}")
    print(f"Best negative log-likelihood: {best_result['nll']}")
