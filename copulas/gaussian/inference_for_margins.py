from copulas.gaussian.gaussian import GaussianCopula
import numpy as np


class GaussianMarginalLikelihoodInference:
    def __init__(self, marginals, densities, correlations):
        """
        Initialize the GaussianMarginalLikelihoodInference class.

        Parameters:
        marginals: array-like
            Series of marginals (N elements, 1D).
        densities: array-like
            Series of densities (N elements, 1D).
        correlations: array-like
            Correlation matrix for the series.
        """
        self.marginals = np.array(marginals)
        self.densities = np.array(densities)
        self.cov_matrix = np.array(correlations)

        # Verify that marginals and densities are compatible
        self.verify_sizes()

    def verify_sizes(self):
        """Verifies that the marginals and densities have the same size."""
        if len(self.marginals) != len(self.densities):
            raise ValueError("Marginals and densities must have the same length.")
        if self.cov_matrix.shape[0] != self.cov_matrix.shape[1]:
            raise ValueError("Covariance matrix must be square.")
        if self.cov_matrix.shape[0] != self.marginals.shape[1]:
            raise ValueError("Covariance matrix dimensions must match the size of the marginals and densities.")

    def compute_marginal_likelihood(self):
        """
        Compute the marginal likelihood using the Gaussian copula.

        Returns:
        log_likelihood: float
            The log-likelihood of the series of marginals.
        """
        # Create an instance of the GaussianCopula class
        gaussian_copula = GaussianCopula(self.marginals, self.cov_matrix)

        # Compute the copula PDF and uniform marginals
        copula_pdf = gaussian_copula.compute_copula()

        copula_pdf = np.maximum(copula_pdf, 1e-10)

        # Compute the marginal likelihood using the copula PDF and densities
        log_likelihood = np.sum(np.sum(np.log(self.densities))) + np.sum(np.log(copula_pdf))

        return log_likelihood


if __name__ == "__main__":
    # Define marginals, densities, and correlation matrix
    marginals = np.random.rand(1000, 2)  # 1000 samples, 2 variables
    densities = np.random.rand(1000, 2)  # Densities corresponding to marginals
    corr_matrix = np.array([[1.0, 0.5], [0.5, 1.0]])  # Correlation matrix

    # Initialize the GaussianMarginalLikelihoodInference and compute the log-likelihood
    inference = GaussianMarginalLikelihoodInference(marginals, densities, corr_matrix)
    log_likelihood = inference.compute_marginal_likelihood()

    print(log_likelihood)
