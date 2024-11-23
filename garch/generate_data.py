import numpy as np


class GenerateData:
    def __init__(self, omega, alpha_vect, beta_vect):
        # Initialize the parameters
        self.omega = omega
        self.alpha_vect = alpha_vect
        self.beta_vect = beta_vect
        self.p = len(alpha_vect)  # Number of ARCH terms
        self.q = len(beta_vect)   # Number of GARCH terms

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

    def generate(self, n):
        """
        Generates a time series of length n using a GARCH(p, q) process.
        :param n: The number of data points to generate.
        :return: A numpy array representing the generated time series (y),
                 the conditional variances (sigma2),
                 and the innovations (eps).
        """
        # Determine the additional size needed based on p and q
        extra_size = max(self.p, self.q)

        # Initialize arrays for storing the returns (y), variances (sigma2), and innovations (eps)
        y = np.zeros(n + extra_size)
        sigma2 = np.zeros(n + extra_size)
        eps = np.zeros(n + extra_size)

        # Start with an initial variance
        sigma2[0] = self.omega / (1 - sum(self.alpha_vect) - sum(self.beta_vect))

        # Loop to generate the series
        for t in range(1, n + extra_size):
            # Calculate conditional variance sigma2[t] based on past residuals and variances
            sigma2[t] = self.omega
            for i in range(min(self.p, t)):
                sigma2[t] += self.alpha_vect[i] * (y[t-i-1] ** 2)
            for j in range(min(self.q, t)):
                sigma2[t] += self.beta_vect[j] * sigma2[t-j-1]

            # Generate the innovation (eps) based on normal distribution with variance sigma2[t]
            eps[t] = np.random.normal(0, 1)

            # Generate the return (y) based on the innovation (eps) and current variance
            y[t] = eps[t] * np.sqrt(sigma2[t])

        # Remove the initial `extra_size` values to ensure a clean time series of length n
        return y[extra_size:], sigma2[extra_size:], eps[extra_size:]


if __name__ == "__main__":

    # Example usage:
    omega = 0.1
    alpha_vect = [0.2, 0.1]
    beta_vect = [0.6]

    data_generator = GenerateData(omega, alpha_vect, beta_vect)
    y, sigma2, eps = data_generator.generate(100)

    print("Generated returns (y):", y)
    print("Generated variances (sigma^2):", sigma2)
