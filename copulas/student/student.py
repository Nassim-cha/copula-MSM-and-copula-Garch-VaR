from scipy.special import gamma
from math import gamma
from copulas.student.generate import generate_student_t_copula_data
import numpy as np
from scipy.stats import t
from numba import njit


class StudentCopula:
    def __init__(self, marginals, cov_matrix, nu):
        """
        Initialize the Student copula class.

        Parameters:
        marginals: 2D array-like (N x d)
            Array of marginals (N samples with d variables each).
        cov_matrix: array-like (d x d)
            Covariance matrix for the multivariate t-distribution.
        nu: float
            Degrees of freedom for the t-distribution.
        """
        self.marginals = np.array(marginals)
        self.cov_matrix = np.array(cov_matrix)
        self.nu = nu
        self.N, self.d = self.marginals.shape

        # Verify sizes
        self.verify_sizes()

    def verify_sizes(self):
        """Verifies that the covariance matrix has the appropriate size relative to the marginals array."""
        if self.cov_matrix.shape[0] != self.cov_matrix.shape[1]:
            raise ValueError("Covariance matrix must be square.")
        if self.cov_matrix.shape[0] != self.d:
            raise ValueError("Covariance matrix dimensions must match the number of variables (d).")

    def compute_copula(self):
        """
        Compute the Student-t copula density for all samples (each row of marginals).

        Returns:
        copula_densities: np.ndarray (N,)
            The density of the Student-t copula for each sample.
        """
        return copula_density(self.marginals, self.nu, self.cov_matrix)


#@njit
def copula_density(u_values, nu, corr_matrix):
    """
    Numba-compatible version of the copula_density function.

    Parameters:
    u_values: np.ndarray
        The marginals (uniform probabilities).
    nu: float
        Degrees of freedom for the t-distribution.
    corr_matrix: np.ndarray
        Covariance matrix for the copula.

    Returns:
    copula_densities: np.ndarray
        The density of the Student-t copula for each sample.
    """
    # Step 1: Transform marginals to uniform margins using the inverse t-cdf
    inv = inverse_t_cdf_vectorized(u_values, nu)

    # Step 2: Compute the multivariate t-PDF for each sample
    pdf_values = multivariate_t_pdf_vectorized(inv, corr_matrix, nu)

    # Step 3: Compute the product of the univariate Student-t PDFs for each sample
    univariate_pdfs = univariate_t_pdf_vectorized(inv, nu)

    product_of_univariate_pdfs = np.prod(univariate_pdfs, axis=1)

    # Step 4: Compute the Student-t copula density for each sample
    copula_densities = pdf_values / product_of_univariate_pdfs

    return copula_densities


def inverse_t_cdf_vectorized(cdf, nu, **kwargs):
    """
    Vectorized version of inverse_t_cdf using scipy's t.ppf for multiple values.

    Parameters:
    values: 2D array (N x d)
        The input values (probabilities) to transform to uniform.
    nu: float
        Degrees of freedom for the t-distribution.

    Returns:
    u_values: 2D array (N x d)
        Transformed values into uniform marginals.
    """
    N, d = cdf.shape
    u_values = np.zeros((N, d))

    # Use scipy's t.ppf to calculate the inverse t-distribution for each marginal
    for i in range(N):
        for j in range(d):
            u_values[i, j] = t.ppf(cdf[i, j], df=nu)

    return u_values

@njit
def multivariate_t_pdf_vectorized(u_values, cov_matrix, nu):
    """
    Vectorized calculation of the multivariate t-distribution PDF for multiple samples.

    Parameters:
    u_values: 2D array (N x d)
        The input values (uniform marginals) to evaluate.
    cov_matrix: 2D array (d x d)
        Covariance matrix of the multivariate t-distribution.
    nu: float
        Degrees of freedom for the t-distribution.

    Returns:
    pdf_values: 1D array (N,)
        The computed multivariate t-distribution PDF values for each sample.
    """
    N, d = u_values.shape
    pdf_values = np.zeros(N)

    sigma_inv = np.linalg.inv(cov_matrix)  # Inverse of the covariance matrix
    det_cov = np.linalg.det(cov_matrix)  # Determinant of the covariance matrix

    for i in range(N):
        x = u_values[i]

        # Handle cases where any x contains inf, -inf, or NaN using np.isfinite
        if not np.all(np.isfinite(x)):
            pdf_values[i] = 0.0
        else:
            quad_form = np.dot(np.dot(x.T, sigma_inv), x)  # Quadratic form

            term1 = gamma((nu + d) / 2) / (gamma(nu / 2) * ((nu * np.pi) ** (d / 2)) * np.sqrt(det_cov))
            term2 = (1 + quad_form / nu) ** (-(nu + d) / 2)

            pdf_values[i] = term1 * term2

    return pdf_values


@njit
def univariate_t_pdf_vectorized(values, nu):
    """
    Compute the PDF of the univariate Student-t distribution for multiple values using Numba's njit.

    Parameters:
    values: 2D array (N x d)
        The input values for which the PDF should be computed.
    nu: float
        Degrees of freedom of the Student-t distribution.

    Returns:
    pdf_values: 2D array (N x d)
        The computed PDF of the univariate Student-t distribution for each input.
    """
    N, d = values.shape
    pdf_values = np.zeros((N, d))

    gamma_term = gamma((nu + 1) / 2) / (np.sqrt(nu * np.pi) * gamma(nu / 2))

    for i in range(N):
        for j in range(d):
            # Handle inf, -inf, or NaN values using np.isfinite
            if not np.isfinite(values[i, j]):
                pdf_values[i, j] = 0.0
            else:
                pdf_values[i, j] = gamma_term * (1 + (values[i, j] ** 2 / nu)) ** (-(nu + 1) / 2)

    return pdf_values


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

    # Initialize the StudentCopula class
    copula = StudentCopula(marginals, corr_matrix_true, nu_true)

    # Compute the copula
    pdf_value = copula.compute_copula()

    print("Multivariate t-distribution PDF (pdf_value):", pdf_value)