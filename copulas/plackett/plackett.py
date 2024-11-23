import numpy as np
from numba import njit


class PlackettCopula:
    def __init__(self, marginals, theta):
        """
        Initialize the Plackett copula class.

        Parameters:
        marginals: 2D array-like (N x 2)
            Array of marginals (N samples with 2 variables each).
        theta: float
            The Plackett copula parameter that controls the strength of dependence.
        """
        self.marginals = np.array(marginals)
        self.theta = float(theta)
        self.N, self.d = self.marginals.shape

        if self.d != 2:
            raise ValueError("Plackett copula is only defined for 2-dimensional marginals.")

    def compute_copula(self):
        """
        Compute the Plackett copula density for all samples (each row of marginals).

        Returns:
        copula_densities: The density of the Plackett copula for each sample (N,).
        """

        copula_densities = copula_density(self.marginals, self.theta)

        return copula_densities

def copula_density(cdf, theta):
    # Step 1: Extract marginals u and v
    u_values = cdf[:, 0]
    v_values = cdf[:, 1]

    # Step 2: Compute the copula density for each sample
    copula_densities = plackett_density_vectorized(u_values, v_values, theta)

    return copula_densities

@njit
def plackett_density_vectorized(u, v, theta):
    """
    Compute the density of the Plackett copula for multiple (u, v) pairs using Numba's njit.

    Parameters:
    u: 1D array (N,)
        The first set of marginals.
    v: 1D array (N,)
        The second set of marginals.
    theta: float
        The Plackett copula parameter.

    Returns:
    densities: 1D array (N,)
        The computed density of the Plackett copula for each pair of marginals.
    """
    N = u.shape[0]
    densities = np.zeros(N)

    for i in range(N):
        num = theta * (1 + (theta - 1) * (u[i] + v[i] - 2 * u[i] * v[i]))
        denom = ((1 + (theta - 1) * (u[i] + v[i])) *
                 (1 + (theta - 1) * (1 - u[i] - v[i]))) ** 2
        densities[i] = num / denom

    return densities


if __name__ == "__main__":
    # Define marginals and correlation matrix
    marginals = np.random.rand(1000, 2)  # 1000 samples, 2 variables
    theta = 5  # theta

    # Initialize GaussianCopula and compute copula densities
    copula = PlackettCopula(marginals, theta)
    copula_densities = copula.compute_copula()

    print(copula_densities)