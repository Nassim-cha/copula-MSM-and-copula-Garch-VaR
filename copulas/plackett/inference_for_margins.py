from copulas.plackett.plackett import PlackettCopula
import numpy as np


class MarginalLikelihoodInferencePlackett:
    def __init__(self, marginals, densities, theta):
        """
        Initialize the MarginalLikelihoodInferencePlackett class.

        Parameters:
        marginals: array-like (N x 2)
            Series of marginals (N samples with 2 variables each).
        densities: array-like
            Series of densities (N elements, 1D).
        theta: float
            The Plackett copula parameter that controls the strength of dependence.
        """
        self.marginals = np.array(marginals)
        self.densities = np.array(densities)
        self.theta = theta

        # Verify that marginals and densities are compatible
        self.verify_sizes()

    def verify_sizes(self):
        """Verifies that the marginals and densities have compatible sizes."""
        if len(self.marginals) != len(self.densities):
            raise ValueError("Marginals and densities must have the same length.")
        if self.marginals.shape[1] != 2:
            raise ValueError("Plackett copula is only defined for 2-dimensional marginals.")

    def compute_marginal_likelihood(self):
        """
        Compute the marginal likelihood using the Plackett copula.

        Returns:
        log_likelihood: float
            The log-likelihood of the series of marginals.
        """
        # Create an instance of the PlackettCopula class
        plackett_copula = PlackettCopula(self.marginals, self.theta)

        # Compute the copula PDF
        copula_pdf = plackett_copula.compute_copula()

        # Compute the marginal likelihood using the copula PDF and densities
        log_likelihood = np.sum(np.log(self.densities)) + np.sum(np.log(copula_pdf))

        return log_likelihood
