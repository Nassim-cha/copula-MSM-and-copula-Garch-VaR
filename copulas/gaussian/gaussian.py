from scipy.stats import norm
import numpy as np
from numba import njit
from numpy.linalg import inv, det


class GaussianCopula:
    def __init__(self, marginals, corr_matrix):
        """
        Initialize the Gaussian copula class.

        Parameters:
        marginals: 2D array-like (N x d)
            Array of marginals (N samples with d variables each).
        corr_matrix: array-like (d x d)
            Correlation matrix for the multivariate normal distribution.
        """
        self.marginals = np.array(marginals)
        self.corr_matrix = np.array(corr_matrix)
        self.N, self.d = self.marginals.shape

        # Verify sizes
        self.verify_sizes()

    def verify_sizes(self):
        """Verifies that the correlation matrix has the appropriate size relative to the marginals."""
        if self.corr_matrix.shape[0] != self.corr_matrix.shape[1]:
            raise ValueError("Correlation matrix must be square.")
        if self.corr_matrix.shape[0] != self.d:
            raise ValueError("Correlation matrix dimensions must match the number of variables (d).")

    def compute_copula(self):
        """
        Compute the Student-t copula density for all samples (each row of marginals).

        Returns:
        copula_densities: np.ndarray (N,)
            The density of the Student-t copula for each sample.
        """
        return copula_density(self.marginals, self.corr_matrix)


def inv_density(cdf, **kwargs):
    return norm.ppf(cdf)


def copula_density(u_values, corr_matrix, **kwargs):
    # Step 1: Transform marginals to normal marginals using the inverse CDF of the normal distribution
    inv = inv_density(u_values)

    # Step 2: Compute the multivariate normal PDF for each sample
    pdf_values = multivariate_normal_pdf_vectorized(inv, corr_matrix)

    # Step 3: Compute the product of the univariate normal PDFs for each sample
    univariate_pdfs = univariate_normal_pdf_vectorized(inv)
    product_of_univariate_pdfs = np.prod(univariate_pdfs, axis=1)

    # Step 4: Compute the Gaussian copula density for each sample
    copula_densities = pdf_values / product_of_univariate_pdfs

    return copula_densities


@njit
def univariate_normal_pdf_vectorized(values):
    """
    Compute the PDF of the univariate normal distribution for multiple values using Numba's njit.

    Parameters:
    values: 2D array (N x d)
        The input values for which the PDF should be computed.

    Returns:
    pdf_values: 2D array (N x d)
        The computed PDF of the univariate normal distribution for each input.
    """
    N, d = values.shape
    pdf_values = np.zeros((N, d))

    for i in range(N):
        for j in range(d):
            pdf_values[i, j] = (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * values[i, j] ** 2)

    return pdf_values


@njit
def multivariate_normal_pdf_vectorized(u_values, corr_matrix):
    """
    Vectorized calculation of the multivariate normal distribution PDF for multiple samples.

    Parameters:
    u_values: 2D array (N x d)
        The input values (normal marginals) to evaluate.
    corr_matrix: 2D array (d x d)
        Correlation matrix of the multivariate normal distribution.

    Returns:
    pdf_values: 1D array (N,)
        The computed multivariate normal distribution PDF values for each sample.
    """
    N, d = u_values.shape
    pdf_values = np.zeros(N)

    sigma_inv = inv(corr_matrix)  # Inverse of the correlation matrix
    det_cov = det(corr_matrix)  # Determinant of the correlation matrix

    for i in range(N):
        x = u_values[i]
        quad_form = np.dot(np.dot(x.T, sigma_inv), x)  # Quadratic form

        term1 = 1 / (np.sqrt((2 * np.pi) ** d * det_cov))
        term2 = np.exp(-0.5 * quad_form)

        pdf_values[i] = term1 * term2

    return pdf_values


if __name__ == "__main__":
    # Define marginals and correlation matrix
    marginals = np.random.rand(1000, 2)  # 1000 samples, 2 variables
    corr_matrix = np.array([[1.0, 0.5], [0.5, 1.0]])  # Correlation matrix

    # Initialize GaussianCopula and compute copula densities
    copula = GaussianCopula(marginals, corr_matrix)
    copula_densities = copula.compute_copula()

    print(copula_densities)