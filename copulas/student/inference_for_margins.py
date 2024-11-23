import numpy as np
from copulas.student.student import StudentCopula
from copulas.student.generate import generate_student_t_copula_data


class MarginalLikelihoodInference:
    def __init__(self, marginals, densities, nu, correlations):
        """
        Initialize the MarginalLikelihoodInference class.

        Parameters:
        marginals: array-like
            Series of marginals (N elements, 1D).
        densities: array-like
            Series of densities (N elements, 1D).
        nu: float
            Degrees of freedom for the Student copula.
        correlations: array-like
            Correlation matrix for the series.
        """
        self.marginals = np.array(marginals)
        self.densities = np.array(densities)
        self.nu = nu
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
        Compute the marginal likelihood using the Student copula.

        Returns:
        log_likelihood: float
            The log-likelihood of the series of marginals.
        """
        # Create an instance of the StudentCopula class
        student_copula = StudentCopula(self.marginals, self.cov_matrix, self.nu)

        # Compute the copula PDF and uniform marginals
        copula_pdf = student_copula.compute_copula()

        # Compute the marginal likelihood using the copula PDF and densities
        log_likelihood = np.sum(np.sum(np.log(self.densities))) + np.sum(np.log(copula_pdf))

        return log_likelihood


if __name__ == "__main__":
    # Parameters for synthetic data
    N = 5000
    dim = 3
    nu_true = 5
    corr_matrix_true = np.array([
        [1.0, 0.5],
        [0.5, 1.0]
    ])

    # Generate Student-t copula samples
    marginals, densities = generate_student_t_copula_data()

    marginal_inference = MarginalLikelihoodInference(
        marginals=marginals,
        densities=densities,
        nu=nu_true,
        correlations=corr_matrix_true
    )
    nll = marginal_inference.compute_marginal_likelihood()

    print("Log Likelihood:", nll)