from garch.estimation import ProbEstimation
from garch.generate_data import GenerateData
import numpy as np
from matplotlib import pyplot as plt


class GarchOptimizer:
    def __init__(self, returns, p_max=3, q_max=3, tol=1e-10, max_iter=1000, epsilon=1e-5):
        self.returns = returns
        self.p_max = p_max
        self.q_max = q_max
        self.best_pq = None
        self.best_params = None
        self.best_result = None
        self.best_bic = None
        self.tol = tol
        self.max_iter = max_iter
        self.epsilon = epsilon  # Small step for finite difference approximation

    def negative_log_likelihood(self, params, p, q):
        """
        Calculate the negative log-likelihood for given parameters.
        params: omega + alpha_vect (p elements) + beta_vect (q elements)
        """
        omega = params[0]
        alpha_vect = params[1:p + 1]
        beta_vect = params[p + 1:]

        # Check if the sum of alpha_vect + beta_vect violates the constraint
        if np.sum(alpha_vect) + np.sum(beta_vect) >= 1:
            return 1e10  # Return a large penalty value if the constraint is violated

        # Create prob_estimation object (assuming it is similar to garch_likelihood)
        prob = ProbEstimation(self.returns, omega, alpha_vect, beta_vect)

        # Return the negative log-likelihood
        return -prob.calculate_log_likelihood()

    def numerical_gradient(self, params, p, q):
        """
        Approximate the gradient using finite differences.
        """
        grad = np.zeros_like(params)
        for i in range(len(params)):
            params_step_up = np.array(params)
            params_step_down = np.array(params)
            params_step_up[i] += self.epsilon
            params_step_down[i] -= self.epsilon

            grad[i] = (self.negative_log_likelihood(params_step_up, p, q) -
                       self.negative_log_likelihood(params_step_down, p, q)) / (2 * self.epsilon)

        return grad

    def numerical_hessian(self, params, p, q):
        """
        Approximate the Hessian using finite differences.
        """
        n = len(params)
        hess = np.zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                params_step_up_i = np.array(params)
                params_step_up_j = np.array(params)
                params_step_down_i = np.array(params)
                params_step_down_j = np.array(params)

                params_step_up_i[i] += self.epsilon
                params_step_up_j[j] += self.epsilon
                params_step_down_i[i] -= self.epsilon
                params_step_down_j[j] -= self.epsilon

                if i == j:
                    # Diagonal elements (second derivative w.r.t. same parameter)
                    hess[i, i] = (self.negative_log_likelihood(params_step_up_i, p, q) -
                                  2 * self.negative_log_likelihood(params, p, q) +
                                  self.negative_log_likelihood(params_step_down_i, p, q)) / (self.epsilon ** 2)
                else:
                    # Off-diagonal elements (mixed second derivatives)
                    f_up_up = self.negative_log_likelihood(params_step_up_i, p, q)
                    f_up_down = self.negative_log_likelihood(params_step_up_j, p, q)
                    f_down_up = self.negative_log_likelihood(params_step_down_i, p, q)
                    f_down_down = self.negative_log_likelihood(params_step_down_j, p, q)

                    hess[i, j] = hess[j, i] = (f_up_up - f_up_down - f_down_up + f_down_down) / (4 * self.epsilon ** 2)

        return hess

    def optimize(self):
        """
        Perform optimization over the range of p and q values using Newton-Raphson.
        """
        best_result = None
        best_params = None
        best_pq = None
        best_bic = None

        n_obs = len(self.returns)  # Number of observations

        # Iterate over different combinations of p and q
        for p in range(1, self.p_max + 1):
            for q in range(1, self.q_max + 1):
                # Initial guess for the parameters (omega + alpha_vect + beta_vect)
                alpha_beta_sum = 0.5 / (p + q)  # Distribute values to satisfy sum < 1
                initial_guess = [0.1] + [alpha_beta_sum] * p + [alpha_beta_sum] * q

                # Optimize using Newton-Raphson
                params, neg_log_likelihood = self.newton_raphson(initial_guess, p, q)

                if params is None:
                    print(f"Failed to converge for p={p}, q={q}")
                    continue

                # Calculate the number of parameters (omega + p alphas + q betas)
                num_params = 1 + p + q

                # Calculate the BIC for this model
                current_bic = self.bic(-neg_log_likelihood, n_obs, num_params)

                if best_result is None or current_bic < best_bic:
                    best_result = neg_log_likelihood
                    best_params = params
                    best_pq = (p, q)
                    best_bic = current_bic

                print(f"p = {p}, q = {q}, -Log-Likelihood = {neg_log_likelihood}, BIC = {current_bic}")

        self.best_pq = best_pq
        self.best_params = best_params
        self.best_result = best_result
        self.best_bic = best_bic

        print(
            f"\nBest model: p = {self.best_pq[0]}, q = {self.best_pq[1]}, -Log-Likelihood = {self.best_result}, BIC = {self.best_bic}")
        print(f"Best parameters: {self.best_params}")

        return self.best_pq, self.best_params, self.best_result, self.best_bic

    def newton_raphson(self, initial_params, p, q):
        """
        Newton-Raphson method for optimizing the negative log-likelihood.
        """
        params = np.array(initial_params)
        for i in range(self.max_iter):
            grad = self.numerical_gradient(params, p, q)
            hess = self.numerical_hessian(params, p, q)

            # Check if Hessian is invertible (using np.linalg.pinv for robustness)
            try:
                hess_inv = np.linalg.pinv(hess)
            except np.linalg.LinAlgError:
                return None, None  # Hessian not invertible, return failure

            # Newton-Raphson update rule
            delta_params = -hess_inv @ grad
            params += delta_params

            # Get the sum of all elements after the first one
            sum_rest = np.sum(params[1:])

            # If the sum of elements after the first is greater than 1, normalize them
            if sum_rest > 1:
                params[1:] = params[1:] / sum_rest

            # Ensure all elements in params are greater than 0
            params = np.maximum(params, self.epsilon + 1e-7)

            # Check convergence
            if np.linalg.norm(delta_params) < self.tol:
                break

        return params, self.negative_log_likelihood(params, p, q)

    def bic(self, log_likelihood, n_obs, num_params):
        """
        Calculate the Bayesian Information Criterion (BIC).
        log_likelihood: The value of the log-likelihood.
        n_obs: The number of observations (length of the returns).
        num_params: The number of parameters in the model.
        """
        return -2 * log_likelihood + num_params * np.log(n_obs)

    def unpack_garch_parameters(self, result):
        """
        Unpack GARCH parameters from the optimizer result.

        Parameters:
        result (tuple): The result from the optimizer containing:
                        - Tuple of p and q values (number of ARCH and GARCH terms).
                        - Array of estimated parameters [omega, alphas, betas].

        Returns:
        omega (float): The omega parameter.
        alpha_vect (np.array): The alpha vector (ARCH terms).
        beta_vect (np.array): The beta vector (GARCH terms).
        """
        (p, q), params = result[0], result[1]

        # The first parameter is omega
        omega = params[0]

        # The next p parameters are the alpha vector (ARCH terms)
        alpha_vect = np.array(params[1:p + 1])

        # The next q parameters are the beta vector (GARCH terms)
        beta_vect = np.array(params[p + 1:p + 1 + q])

        return omega, alpha_vect, beta_vect


if __name__ == "__main__":
    # Example usage
    omega = 0.1
    alpha_vect = [0.2, 0.1, 0.1]
    beta_vect = [0.1, 0.3]

    data_generator = GenerateData(omega, alpha_vect, beta_vect)
    returns, sigma2, eps_g = data_generator.generate(1000)

    # Initialize the GarchOptimizer class
    optimizer = GarchOptimizer(returns)

    # Run the optimizer over different (p, q) combinations
    best_pq, best_params, best_result, best_bic = optimizer.optimize()

    result = best_pq, best_params, best_result, best_bic

    omega, alpha_vect, beta_vect = optimizer.unpack_garch_parameters(result)

    prob = ProbEstimation(returns, omega, alpha_vect, beta_vect)

    eps = prob.calculate_eps_t()

    # Creating two subplots to plot eps and eps2 separately
    fig, axs = plt.subplots(2, 1, figsize=(10, 10))

    # Plotting eps in the first subplot
    axs[0].plot(eps, label="Eps_t (Residuals)", color='b')
    axs[0].set_title("Residuals (Eps_t) Over Time")
    axs[0].set_xlabel("Time")
    axs[0].set_ylabel("Eps_t")
    axs[0].legend()
    axs[0].grid(True)

    # Plotting eps2 in the second subplot
    axs[1].plot(eps_g, label="Eps2_t (Another Series)", color='r')
    axs[1].set_title("Residuals (Eps2_t) Over Time")
    axs[1].set_xlabel("Time")
    axs[1].set_ylabel("Eps2_t")
    axs[1].legend()
    axs[1].grid(True)

    # Show the plots
    plt.tight_layout()
    plt.show()