from kalman_mean_reverting.generate import DataGeneration
import numpy as np
from matplotlib import pyplot as plt
from numba import njit


class KalmanFilterVolEstimation:
    def __init__(self,
                 a,
                 l,
                 q,
                 init_log_vol,
                 init_var,
                 n_steps,
                 returns_series,
                 alpha=1.6,
                 beta=2,
                 kappa=1.75):
        self.a = a  # Mean reversion speed
        self.l = l  # Long-term mean
        self.q = q  # Volatility of process
        self.alpha = alpha  # UKF tuning parameter
        self.beta = beta  # UKF tuning parameter
        self.kappa = kappa  # UKF tuning parameter
        self.init_log_vol = init_log_vol  # Initial log correlation
        self.init_var = init_var  # Initial variance
        self.n_steps = n_steps  # Number of time steps
        self.returns_series = returns_series
        self.state_estimation, self.var_setimation, self.LL, self.forecasts = self.calc_log_likelihood_numba()

    def calc_log_likelihood_numba(self):
        state_estimation, var_estimation, LL, forecasts = calculate_loglikelihood(
            self.n_steps,
            self.returns_series,
            self.init_log_vol,
            self.q,
            self.init_var,
            self.a,
            self.l,
            self.alpha,
            self.kappa,
            self.beta
        )
        return state_estimation, var_estimation, LL, forecasts

    def sto_vol_estimation(self):
        """ Compute the estimated correlation using exp of the state estimation. """
        return np.exp(self.state_estimation)

    def calc_eps_t(self):
        return self.returns_series / self.sto_vol_estimation()

@njit
def custom_cholesky(A, epsilon=1e-8):
    """
    Custom Cholesky decomposition that works with Numba.
    Adds small regularization (epsilon) to the diagonal elements if decomposition fails.
    Handles only 2D positive-definite matrices.
    """
    # Ensure A is a 2D array (matrix case)
    n = A.shape[0]
    L = np.zeros_like(A)

    for i in range(n):
        for j in range(i + 1):
            sum_val = 0.0
            for k in range(j):
                sum_val += L[i, k] * L[j, k]
            if i == j:
                diff = A[i, i] - sum_val
                # Regularization: If diff is non-positive, add epsilon to the diagonal
                if diff <= 0:
                    diff += epsilon
                L[i, j] = np.sqrt(diff)
            else:
                L[i, j] = (A[i, j] - sum_val) / L[j, j]

    return L

@njit
def generate_aug_sp(aug_x, aug_cov, L, lambda_):
    """ Generate sigma points using the augmented state vector and covariance matrix."""
    # Compute the custom Cholesky decomposition of the augmented covariance matrix
    P_sqrt = custom_cholesky(aug_cov)

    # Compute phi as sqrt(L + lambda_)
    phi = np.sqrt(L + lambda_)

    # Initialize the sigma points array
    X1 = np.zeros((2 * L + 1, aug_x.size))

    # First sigma point is the mean (aug_x)
    X1[0] = aug_x

    # Generate the rest of the sigma points
    for i in range(L):
        X1[i + 1] = aug_x + phi * P_sqrt[:, i]
        X1[i + L + 1] = aug_x - phi * P_sqrt[:, i]

    return X1

@njit
def calc_weights(L, lambda_, alpha, beta):
    """ Calculate UKF weights. """
    wm = np.full(2 * L + 1, 1 / (2 * (L + lambda_)))
    wc = np.full(2 * L + 1, 1 / (2 * (L + lambda_)))
    wm[0] = lambda_ / (L + lambda_)
    wc[0] = wm[0] + (1 - alpha ** 2 + beta)
    return wm, wc

@njit
def calc_weights_2(L, lambda_, alpha, beta):
    """ Calculate UKF weights for measurement update. """
    wm2 = np.full(L + 1, 1 / (2 * (L + lambda_)))
    wm2[0] = lambda_ / (L + lambda_)
    return wm2

@njit
def initialize_ukf_state(init_var, init_log_correl):
    """ Initialize the augmented covariance and state vector. """
    aug_cov = np.zeros((2, 2))  # Initialize a 2x2 matrix of zeros
    aug_cov[0, 0] = init_var
    aug_cov[1, 1] = 1.0  # Set the value for the second diagonal entry
    aug_x = np.zeros(2)
    aug_x[0] = init_log_correl
    return aug_cov, aug_x


@njit
def f_vectorized(x1, x2, a, l, q):
    """
    Vectorized transition function for state evolution, compatible with Numba.
    Applies mean reversion and clipping to the state variables.

    x1: array of previous state variables
    x2: array of noise or random process variables
    a: mean reversion speed
    l: long-term mean (or target value)
    """
    # Apply mean-reverting process element-wise
    return a * (x1 - l) + l + q * x2


@njit
def generate_sp(P, L, lambda_, X_mean):
    """
    Generate sigma points for the update step using covariance matrix P and mean X_mean.

    P: Covariance matrix (assumed to be already Cholesky-decomposed or sqrt of covariance matrix)
    L: Dimension of the state vector (here assumed to be 1, scalar case)
    lambda_: Scaling parameter for sigma points
    X_mean: Predicted state mean (float scalar)
    """
    # Compute the scaling factor
    phi = np.sqrt(L + lambda_)

    P_sqrt = np.sqrt(P)

    # Initialize the sigma points array
    X2 = np.zeros(3)  # 3 sigma points for L = 1

    # First sigma point is the mean
    X2[0] = X_mean

    # Generate sigma points based on square root of P and phi
    X2[1] = X_mean + phi * P_sqrt
    X2[2] = X_mean - phi * P_sqrt

    return X2


@njit
def normal_pdf(x):
    """ Calculate the probability density function of a standard normal distribution. """
    return (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * x ** 2)

@njit
def safe_tanh(x):
    """ Safe tanh that avoids extreme values close to -1 or 1 without using np.clip. """
    eps = 1e-10  # A small epsilon for stability

    # Define the safe bounds
    lower_bound = -1 + eps
    upper_bound = 1 - eps

    y = np.tanh(x)

    # Apply clipping manually in a vectorized way
    y = np.where(y < lower_bound, lower_bound, y)
    y = np.where(y > upper_bound, upper_bound, y)

    # Return the tanh of the clipped values
    return y


@njit
def vectorized_update_step(X2, r, wm2):
    """
    Perform the vectorized update step in the Kalman filter.

    X2: 1D array of sigma points
    first, second: Scalars or arrays representing the first and second measurements
    wm2: Weights for the measurement update
    """

    # Vectorized calculation of eta assuming X2 is 1D
    eta = r / np.exp(X2)  # Assuming `h` is based on this formula

    # Calculate the normal density for each eta value
    prob_eta = normal_pdf(eta)

    # Vectorized calculation of h_x
    h_x = prob_eta * np.abs(eta)

    # Calculate normalization constant Z
    Z = np.sum(wm2 * h_x)

    # Error handling for Z
    if Z <= 0 or Z < 1e-10:
        return np.nan, np.nan, -1e10  # Error: Z is either too small or non-positive

    # Calculate the mean using the sigma points and h_x
    mean = np.sum((wm2 * X2 * h_x) / Z)

    # Calculate the variance using the updated sigma points
    var = np.sum(wm2 * ((h_x / Z) * (X2 - mean) ** 2))

    return mean, var, Z

@njit
def calculate_loglikelihood(n_steps, returns_series, init_log_vol, q, init_var, a, l, alpha, kappa, beta):
    """ Main function to estimate state and variance using UKF, in a Numba-compatible manner. """
    L = 2
    lambda_ = (alpha ** 2) * (L + kappa) - L

    # Initialize the augmented covariance and state
    aug_cov, aug_x = initialize_ukf_state(init_var, init_log_vol)

    # Calculate UKF weights
    wm, wc = calc_weights(L, lambda_, alpha, beta)
    wm2 = calc_weights_2(L, lambda_, alpha, beta)

    state_estimation = np.zeros(n_steps)
    var_estimation = np.zeros(n_steps)

    X_mean=0

    LL = 0

    for t in range(n_steps):
        ### PREDICTION STEP
        X1 = generate_aug_sp(aug_x, aug_cov, L, lambda_)

        # Apply the vectorized `f` function directly to the arrays `X1[:, 0]` and `X1[:, 1]`
        X = f_vectorized(X1[:, 0], X1[:, 1], a, l, q)

        # Predicting state mean and covariance
        X_mean = np.dot(X.T, wm)  # X.T @ wm works in Numba

        X_diff = X - X_mean
        # Element-wise multiplication for the covariance without using np.diag
        P = np.dot(X_diff.T * wc, X_diff)  # Equivalent to X_diff.T @ np.diag(wc) @ X_diff

        ### UPDATE STEP
        X2 = generate_sp(P, L, lambda_, X_mean)

        mean, var, Z = vectorized_update_step(X2, returns_series[t], wm2)

        # If an error occurred (mean, var, or Z is NaN), return error values
        if np.isnan(mean) or np.isnan(var) or np.isnan(Z):
            return None, None, -1e10, None

        state_estimation[t] = mean
        var_estimation[t] = var

        LL += np.log(np.abs(Z))

        # Initialize the augmented covariance and state
        aug_cov, aug_x = initialize_ukf_state(var, mean)

    return state_estimation, var_estimation, LL, X_mean


if __name__ == "__main__":
    # Define the test parameters for a simple example
    a = 0.95
    l = 0
    q = 0.2
    init_correl = l  # Initial log correlation, which can be set as the long-term mean
    init_var = q     # Set init_var equal to q as per your earlier requirements
    n_steps = 500    # Number of time steps

    # Example: Generate synthetic data using the DataGeneration class
    data_gen = DataGeneration(theta=a, mu=l, sigma=q, n_steps=n_steps)
    X, vol, Y = data_gen.generate_process_volatility_returns()  # Generate OU process

    # Example: Run the Kalman Filter correlation estimation
    kalman_filter = KalmanFilterVolEstimation(a, l, q, init_correl, init_var, n_steps, Y)
    state_estimation = kalman_filter.state_estimation
    LL = kalman_filter.LL

    # Compute the estimated stochastic correlation
    estimated_vol = kalman_filter.sto_vol_estimation()

    def update_a_with_ols(state_estimates, l):
        """Update 'a' using Ordinary Least Squares (OLS)."""
        y = state_estimates[1:] #- l
        x = state_estimates[:-1] #- l

        numerator = np.sum(x * y)
        denominator = np.sum(x ** 2)
        if denominator == 0:
            return 0.01  # Avoid division by zero, fallback to small 'a'

        return numerator / denominator

    a_est = update_a_with_ols(state_estimation, l)
    q_est = np.std(state_estimation) * np.sqrt(1 - a**2)
    print(a_est)
    print(q_est)

    # Plot the OU process and state estimation
    plt.figure(figsize=(10, 6))
    plt.plot(X, label="OU Process", color='blue')
    plt.plot(state_estimation, label="State Estimation", color='red', linestyle='--')
    plt.title(f'Ornstein-Uhlenbeck Process and State Estimation (a={a}, l={l}, q={q})')
    plt.xlabel('Time Steps')
    plt.ylabel('Value')
    plt.legend()
    # Display the plot
    plt.show()

    # Plot the generated stochastic correlation and estimated correlation
    plt.figure(figsize=(10, 6))
    plt.plot(vol, label="Generated Stochastic Correlation", color='blue')
    plt.plot(estimated_vol, label="Estimated Stochastic Correlation", color='red', linestyle='--')
    plt.title(f'Generated and Estimated Correlation (a={a}, l={l}, q={q})')
    plt.xlabel('Time Steps')
    plt.ylabel('Value')
    plt.legend()
    # Display the plot
    plt.show()

    print("Example completed. Plots displayed.")

    print(LL)

