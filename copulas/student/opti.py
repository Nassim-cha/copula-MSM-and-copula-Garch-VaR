from copulas.student.inference_for_margins import MarginalLikelihoodInference
from copulas.student.generate import generate_student_t_copula_data
from scipy.optimize import minimize
import numpy as np
from scipy.linalg import cholesky


class Optimizer:
    def __init__(self, marginals, densities, nu_values=np.linspace(2.1, 30, 10), tol=1e-9, max_iter=5000):
        """
        Initialize the optimizer with the marginals, densities, and nu_values.

        Parameters:
        marginals: list of arrays
            A list where each element is a 1D array of marginals for a variable.
        densities: list of arrays
            A list where each element is a 1D array of densities for a variable.
        nu_values: list or array-like
            A list or array of nu values to try during optimization.
        tol: float, optional
            The tolerance for convergence.
        max_iter: int, optional
            Maximum number of iterations.
        """
        self.marginals = np.array(marginals)
        self.densities = np.array(densities)
        self.nu_values = nu_values  # Directly accept nu values here
        self.N = self.marginals.shape[0]  # Number of observations
        self.dim = self.marginals.shape[1]  # Number of variables
        self.objective_func = self.negative_log_likelihood
        self.tol = tol
        self.max_iter = max_iter

    def negative_log_likelihood(self, params):
        """
        Compute the negative log-likelihood for the given parameters.
        """
        nu = params[0]
        corr_params = params[1:]

        # Reconstruct the correlation matrix
        corr_matrix = self.construct_correlation_matrix(corr_params)

        # Ensure the correlation matrix is positive-definite and doesn't contain NaN or Inf
        if np.isnan(corr_matrix).any() or np.isinf(corr_matrix).any():
            return 1e10  # Return a large value if invalid correlation matrix

        # Ensure the correlation matrix is positive-definite
        try:
            cholesky(corr_matrix)
        except np.linalg.LinAlgError:
            return 1e10  # Return a large number if the matrix is not positive-definite

        # Compute the log-likelihood for each observation
        marginal_inference = MarginalLikelihoodInference(
            marginals=self.marginals,
            densities=self.densities,
            nu=nu,
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
        Run the optimizer in two stages:
        1. Optimize correlation parameters for each nu in nu_values.
        2. With the best correlation parameters, minimize nu.

        Parameters:
        initial_corr: array-like, optional
            Initial guess for the correlation parameters (same for all nu values).
        method: str, optional
            The optimization method to be used by scipy.optimize.minimize.

        Returns:
        best_result: dict
            The best result with the lowest negative log-likelihood.
        """
        best_nll = np.inf
        best_corr_params = None
        best_result = None

        # If no initial correlation provided, initialize it with 0.5
        if initial_corr is None:
            initial_corr = np.full((self.dim * (self.dim - 1)) // 2, 0.5)

        # Create bounds: (nu > 2.01, correlations between -0.99 and 0.99)
        corr_bounds = [(-0.99, 0.99)] * ((self.dim * (self.dim - 1)) // 2)
        nu_bounds = [(2.01, 50)]

        # Step 1: Optimize correlation parameters for each nu
        for nu in self.nu_values:
            print(f"\nOptimizing correlations for initial nu = {nu}")

            # Initial guess for the correlation parameters
            initial_corr_params = initial_corr

            # Define the objective function for optimizing correlation parameters for a fixed nu
            def corr_objective(corr_params):
                return self.negative_log_likelihood(np.hstack(([nu], corr_params)))

            # Optimize correlation parameters using minimize
            res_corr = minimize(
                fun=corr_objective,  # Objective function for correlation optimization
                x0=initial_corr_params,  # Initial guess for correlation parameters
                method=method,  # Optimization method (trust-constr, etc.)
                bounds=corr_bounds,  # Bounds for correlation parameters
                tol=self.tol,  # Tolerance for convergence
                options={'maxiter': self.max_iter}  # Maximum iterations
            )

            # Get the optimized correlation parameters
            optimized_corr_params = res_corr.x

            # Calculate the negative log-likelihood for the optimized correlation parameters
            nll_corr = self.negative_log_likelihood(np.hstack(([nu], optimized_corr_params)))
            print(f"Negative Log-Likelihood for nu={nu}: {nll_corr}")
            print(f"Optimized correlation parameters: {optimized_corr_params}")

            # Track the best correlation parameters
            if nll_corr < best_nll:
                best_nll = nll_corr
                best_corr_params = optimized_corr_params

        # Step 2: Optimize nu with the best correlation parameters found
        print("\nOptimizing nu with the best correlation parameters")

        # Define the objective function for optimizing nu with fixed best_corr_params
        def nu_objective(nu):
            return self.negative_log_likelihood(np.hstack((nu, best_corr_params)))

        # Optimize nu using minimize
        res_nu = minimize(
            fun=nu_objective,  # Objective function for nu optimization
            x0=[10],  # Initial guess for nu
            method=method,  # Optimization method
            bounds=nu_bounds,  # Bounds for nu (nu > 2)
            tol=self.tol,  # Tolerance for convergence
            options={'maxiter': self.max_iter}  # Maximum iterations
        )

        # Get the optimized nu
        optimized_nu = res_nu.x[0]

        # Calculate the negative log-likelihood for the optimized nu and best_corr_params
        optimized_nu = np.array([optimized_nu])
        final_nll = self.negative_log_likelihood(np.hstack((optimized_nu, best_corr_params)))

        print(f"\nFinal Negative Log-Likelihood: {final_nll}")
        print(f"Optimized nu: {optimized_nu}")
        print(f"Optimized correlation matrix: {self.construct_correlation_matrix(best_corr_params)}")

        best_result = {
            'nu': optimized_nu,
            'corr_matrix': self.construct_correlation_matrix(best_corr_params),
            'nll': final_nll,
            'optimized_params': np.hstack((optimized_nu, best_corr_params))
        }

        return best_result


# Testing the optimizer with multiple nu values
if __name__ == "__main__":
    # Parameters for synthetic data
    N = 5000
    dim = 3
    nu_true = 20
    corr_matrix_true = np.array([
        [1.0, 0.5],
        [0.5, 1.0]
    ])

    # Generate Student-t copula samples
    marginals, densities = generate_student_t_copula_data()

    # Initialize the optimizer
    optimizer = Optimizer(marginals, densities)

    # Run the optimizer for the provided nu values and retrieve the best result
    best_result = optimizer.optimize()

    # Print the best result
    print("\nBest Optimization Result:")
    print(f"Optimized nu: {best_result['nu']}")
    print("Optimized Correlation Matrix:")
    print(best_result['corr_matrix'])
    print(f"Best Negative Log-Likelihood: {best_result['nll']}")

