from garch.generate_data import GenerateData
import numpy as np
from numba import njit
from matplotlib import pyplot as plt


class ProbEstimation:
    def __init__(self, returns, omega, alpha_vect, beta_vect):
        self.returns = returns
        self.omega = omega
        self.alpha_vect = np.array(alpha_vect)
        self.beta_vect = np.array(beta_vect)
        self.p = len(alpha_vect)  # Number of ARCH terms
        self.q = len(beta_vect)  # Number of GARCH terms

        # Small epsilon to prevent division by zero or log(0)
        self.epsilon = 1e-7

        # Perform parameter verifications
        self.verify_params()

    def verify_params(self):
        """
        Verifies that:
        1. The sum of alpha_vect and beta_vect is less than 1.
        2. All parameters (omega, alpha_vect, and beta_vect) are greater than 0.
        """
        # Check if all elements in alpha_vect and beta_vect are positive
        if not all(a > 0 for a in self.alpha_vect):
            raise ValueError("All elements of alpha_vect must be positive.")
        if not all(b > 0 for b in self.beta_vect):
            raise ValueError("All elements of beta_vect must be positive.")
        if self.omega <= 0:
            raise ValueError("Omega must be positive.")

        # Check if the sum of alpha_vect + beta_vect is less than 1
        if sum(self.alpha_vect) + sum(self.beta_vect) >= 1:
            raise ValueError("The sum of alpha_vect and beta_vect must be less than 1.")

    def calculate_conditional_variances(self):
        """
        Calculate the conditional variances (sigma^2) for the given returns
        and parameters (omega, alpha_vect, beta_vect).
        """
        n = len(self.returns)
        extra_size = max(self.p, self.q)

        # Initialize variance array (sigma^2) with extra initial values
        sigma2 = np.zeros(n)

        # Initialize the starting variance using the unconditional variance approximation
        sigma2[0] = self.omega / (1 - sum(self.alpha_vect) - sum(self.beta_vect))

        # Loop to calculate the conditional variances
        for t in range(1, n):
            sigma2[t] = self.omega
            for i in range(min(self.p, t)):
                sigma2[t] += self.alpha_vect[i] * (self.returns[t - i - 1] ** 2)
            for j in range(min(self.q, t)):
                sigma2[t] += self.beta_vect[j] * sigma2[t - j - 1]

            # Ensure variance is positive by adding a small epsilon
            sigma2[t] = max(sigma2[t], self.epsilon)

        return sigma2

    def calculate_log_likelihood(self):
        """
        Calculate the log-likelihood of the given returns under the GARCH model.
        """
        # Calculate the log-likelihood using a Numba-optimized helper function
        log_likelihood = numba_garch_log_likelihood(self.returns, self.omega, self.alpha_vect, self.beta_vect,
                                                    self.epsilon)
        return log_likelihood

    def calculate_eps_t(self):
        """
        Calculate the residuals (epsilon_t) for the given returns and parameters.
        """
        # Calculate the conditional variances
        sigma2 = self.calculate_conditional_variances()

        # Calculate the conditional standard deviations
        sigma = np.sqrt(sigma2)

        # Calculate epsilon_t = returns / sigma
        eps_t = self.returns / sigma

        return eps_t

@njit
def numba_garch_log_likelihood(returns, omega, alpha_vect, beta_vect, epsilon):
    """
    Numba-optimized function to calculate the log-likelihood of a GARCH(p, q) model.
    """
    n = len(returns)
    p = len(alpha_vect)
    q = len(beta_vect)
    extra_size = max(p, q)

    # Initialize the variance array (sigma^2)
    sigma2 = np.zeros(n)

    # Initialize the starting variance using the unconditional variance approximation
    sigma2[0] = omega / (1 - np.sum(alpha_vect) - np.sum(beta_vect))

    # Loop to calculate the conditional variances
    for t in range(1, n):
        sigma2[t] = omega
        for i in range(min(p, t)):
            sigma2[t] += alpha_vect[i] * (returns[t - i - 1] ** 2)
        for j in range(min(q, t)):
            sigma2[t] += beta_vect[j] * sigma2[t - j - 1]

        # Ensure variance is positive by adding a small epsilon
        sigma2[t] = max(sigma2[t], epsilon)

    # Chop off the first max(p, q) values from returns and sigma2
    returns_chopped = returns[extra_size:]
    sigma2_chopped = sigma2[extra_size:]

    # Calculate the log-likelihood
    log_likelihood = -0.5 * np.sum(np.log(2 * np.pi * sigma2_chopped) + (returns_chopped ** 2) / sigma2_chopped)

    return log_likelihood


if __name__ == "__main__":
    # Example usage:
    omega = 0.1
    alpha_vect = [0.2, 0.1, 0.1]
    beta_vect = [0.1, 0.3]

    data_generator = GenerateData(omega, alpha_vect, beta_vect)
    y, sigma2, eps_g = data_generator.generate(1000)

    likelihood = ProbEstimation(y, omega, alpha_vect, beta_vect)
    L = likelihood.calculate_log_likelihood()

    eps = likelihood.calculate_eps_t()

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