import numpy as np
from kalman_mean_reverting.generate import DataGeneration
from kalman_mean_reverting.estimate import KalmanFilterVolEstimation


class VolOptimizer:
    def __init__(
            self,
            a,
            l,
            q,
            max_iter=1000,
            tol=1e-7,
            perturb_scale=0.05,
            restart_attempts=5
    ):
        self.a = a
        self.l = l
        self.q = q
        self.max_iter = max_iter
        self.tol = tol
        self.perturb_scale = perturb_scale
        self.restart_attempts = restart_attempts
        self.best_LL = -np.inf
        self.best_params = np.array([a, l, q])
        self.n_steps = None # pre-allocate

    def e_step(self, params, returns):
        """E-step: Estimate the state and variance using UKF."""
        a, l, q = params
        ukf = KalmanFilterVolEstimation(a, l, q, l, q, self.n_steps, returns)
        return ukf

    def update_a_with_ols(self, state_estimates, a, l):
        """Update 'a' using Ordinary Least Squares (OLS)."""
        y = state_estimates[1:] - a*l
        x = state_estimates[:-1] - a*l

        numerator = np.sum(x * y)
        denominator = np.sum(x ** 2)
        if denominator == 0:
            return 0.01  # Avoid division by zero, fallback to small 'a'

        return numerator / denominator

    def update_l(self, mu, q, a):
        """Recalculate 'l' based on new 'a' and 'q'."""
        return q**2 / (2*(1-a**2))

    def update_q(self, a, state_estimation):
        """Update 'q' based on 'a' and 'l'."""
        std = np.std(state_estimation)
        return std * np.sqrt(1 - a**2)

    def random_perturbation(self, params, mu, returns):
        """Apply random perturbation to parameters."""
        # Keep retrying until a valid state estimation is obtained
        while True:

            # Perturb 'a' within the defined range [0.5, 0.999999]
            a = np.clip(params[0] + np.random.uniform(-self.perturb_scale, self.perturb_scale), 0.5, 0.999999)
            params[0] = a

            # E-step: Estimate state and variance using current 'a', 'l', and 'q'
            ukf = self.e_step(params, returns)

            # Check if the state_estimation is valid (not None)
            if ukf.state_estimation is not None:
                state_estimation = ukf.state_estimation
                break  # Exit the loop once a valid state estimation is found
            else:
                print("State estimation is None, retrying with a new perturbation.")

        q = self.update_q(a, ukf.state_estimation)
        l = self.update_l(mu, q, a)
        return np.array([a, l, q])

    def em_algorithm(self, returns):
        """EM algorithm to estimate 'a', 'l', and 'q'."""
        params = np.array([self.a, self.l, self.q])
        self.n_steps = len(returns)

        a, l, q = self.a, self.l, self.q

        abs_log_returns = np.log(abs(returns))

        mu = np.mean(abs_log_returns)

        print(f"Starting EM algorithm with max iterations: {self.max_iter}")
        print(f"Initial parameters: a={self.a}, l={self.l}, q={self.q}")

        for iteration in range(self.max_iter):

            # E-step: Estimate state and variance using current 'a', 'l', and 'q'
            ukf = self.e_step(params, returns)


            # Handle failure case: process failed, apply random perturbation
            if ukf.LL == -1e10:
                print(f"Iteration {iteration}: UKF failed, applying random perturbation and retrying.")
                params = self.random_perturbation(params, mu, returns)
                continue

            # Log likelihood improvement check
            LL_diff = np.abs(ukf.LL - self.best_LL)
            print(
                f"Log-likelihood: {ukf.LL:.6f}, Best log-likelihood so far: {self.best_LL:.6f}, Improvement: {LL_diff:.6e}")

            # Check for convergence based on log-likelihood difference
            if LL_diff < self.tol:
                print(
                    f"Convergence reached after {iteration + 1} iterations with log-likelihood improvement < tol ({self.tol}).")
                self.best_LL = ukf.LL
                self.best_params = params.copy()

                # Continue searching for potential better solutions
                for restart in range(self.restart_attempts):
                    print(f"Restart attempt {restart + 1}/{self.restart_attempts}")
                    params = self.random_perturbation(self.best_params, mu, returns)
                    ukf = self.e_step(params, returns)
                    if ukf.LL > self.best_LL:
                        self.best_LL = ukf.LL
                        self.best_params = params.copy()
                        print(f"Found better log-likelihood after random perturbation: {self.best_LL:.6f}")

                # Continue iterating instead of breaking to ensure more updates
                continue

            # Update best parameters and best log-likelihood
            if ukf.LL > self.best_LL:
                print(f"New best log-likelihood found: {ukf.LL:.6f}, updating best parameters.")
                self.best_LL = ukf.LL
                self.best_params = params.copy()

            ukf = self.e_step(params, returns)
            if ukf.state_estimation is None:
                params = self.random_perturbation(params, mu, returns)
                continue
            else:
                # M-step Phase 2: Update 'q'
                q_new = self.update_q(a, ukf.state_estimation)
                print(f"Updated 'q': {q_new:.6f}")

                # Recalculate 'l' based on new 'q'
                l_new = self.update_l(mu, q_new, a)
                print(f"Updated 'l': {l_new:.6f}")

                # M-step Phase 1: Update 'a' using OLS
                a_new = np.clip(self.update_a_with_ols(ukf.state_estimation, a, l_new), 0.5, 0.99)


            if params[0] == a_new:
                params = self.random_perturbation(self.best_params, mu, returns)
            else:
                # Update parameters
                params[0] = a_new
                params[1] = l_new
                params[2] = q_new



        # The final print statement remains the same
        print(
            f"\nEM algorithm completed. Best parameters: a={self.best_params[0]:.6f}, l={self.best_params[1]:.6f}, q={self.best_params[2]:.6f}")
        print(f"Best log-likelihood: {self.best_LL:.6f}")

        return self.best_params, self.best_LL


if __name__ == "__main__":
    # Define the test parameters for a simple example
    a = 0.95
    l = 0
    q = 0.2
    init_correl = l  # Initial log correlation, which can be set as the long-term mean
    init_var = q     # Set init_var equal to q as per your earlier requirements
    n_steps = 1000    # Number of time steps

    # Example: Generate synthetic data using the DataGeneration class
    data_gen = DataGeneration(theta=a, mu=l, sigma=q, n_steps=n_steps)
    X, vol, Y = data_gen.generate_process_volatility_returns()  # Generate OU process

    q_test = 0.1
    l_test = 0.5
    a_test = 0.99

    # Initialize and run the optimizer
    optimizer = VolOptimizer(a_test, l_test, q_test, max_iter=1000, tol=1e-6)
    params, LL = optimizer.em_algorithm(Y)
    print(params)
    print(LL)

    ukf = KalmanFilterVolEstimation(a, l, q, l, q, n_steps, Y)
    print(ukf.LL)
