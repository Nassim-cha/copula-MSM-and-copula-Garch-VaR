from scipy.stats import t, norm
import numpy as np
from numba import njit
from math import gamma, sqrt, pi

@njit
def t_cdf(x, nu):
    """
    Compute the CDF of the Student's t-distribution.
    """
    if nu <= 0:
        raise ValueError("Degrees of freedom must be positive")

    if nu == 1:
        return 0.5 + np.arctan(x) / pi

    a = gamma((nu + 1) / 2) / (sqrt(nu * pi) * gamma(nu / 2))
    b = (1 + (x ** 2) / nu) ** (- (nu + 1) / 2)

    return 0.5 + x * a * b

@njit
def inverse_t_cdf(u, nu, tol=1e-6, max_iter=100):
    """
    Use bisection method to compute the inverse of the t-distribution CDF.
    """
    a, b = -1000, 1000
    fa = t_cdf(a, nu) - u
    fb = t_cdf(b, nu) - u

    if fa * fb >= 0:
        return 0

    for iteration in range(max_iter):
        c = (a + b) / 2
        fc = t_cdf(c, nu) - u

        if abs(fc) < tol or (b - a) / 2 < tol:
            return c

        if fa * fc < 0:
            b = c
            fb = fc
        else:
            a = c
            fa = fc

    raise RuntimeError("Bisection method did not converge.")

@njit
def t_copula(u1, u2, rho, nu):
    """
    Calculate the t-copula value for a pair (u1, u2) and correlation rho.
    """
    x1 = inverse_t_cdf(u1, nu)
    x2 = inverse_t_cdf(u2, nu)

    # Multivariate t-copula formula, simplified assuming rho as correlation
    term1 = 1 / sqrt(1 - rho ** 2)
    term2 = (x1 ** 2 + x2 ** 2 - 2 * rho * x1 * x2) / (nu * (1 - rho ** 2))

    copula_value = (1 + term2) ** (- (nu + 2) / 2)

    return copula_value

def generate_student_t_copula_data(n=100000, nu=5, rho=0.5, top_n=1000):
    """
    Generate `n` random couples, compute t-copula values, and keep the top `top_n` results.
    """
    np.random.seed(42)  # For reproducibility
    # Generate n random pairs between 0 and 1
    random_couples = np.random.rand(n, 2)
    copula_values = np.zeros(n)

    # Calculate t-copula for each pair
    for i in range(n):
        u1, u2 = random_couples[i]
        copula_values[i] = t_copula(u1, u2, rho, nu)

    # Find the indices of the top `top_n` copula values
    top_indices = np.argsort(copula_values)[-top_n:]

    # Return the best `top_n` random couples based on copula values
    best_couples = random_couples[top_indices]

    x = norm.ppf(t.cdf(best_couples, nu))

    densities = norm.pdf(x)

    return best_couples, densities


if __name__ == "__main__":
    # Example usage
    best_couples, best_copula_values = generate_student_t_copula_data()

    print(best_couples)
