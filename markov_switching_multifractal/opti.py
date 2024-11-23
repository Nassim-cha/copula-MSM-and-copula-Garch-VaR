from markov_switching_multifractal.calc_prob import ProbEstimation
from markov_switching_multifractal.generate_data import GenerateData
import numpy as np
import concurrent.futures
import time


class Optimizer:
    def __init__(self, returns, k, max_iter=100, tol=1e-6, basin_iter=100, step_size=0.2, temperature=1.0,
                 gamma_weight=0, b_weight=0):
        self.returns = returns
        self.k = k
        self.max_iter = max_iter
        self.tol = tol
        self.basin_iter = basin_iter  # Number of basin-hopping iterations
        self.step_size = step_size  # Step size for random perturbation
        self.temperature = temperature  # Temperature for Metropolis criterion
        self.sample_variance = np.var(returns)  # Estimate variance of returns
        self.gamma_weight = gamma_weight
        self.b_weight = b_weight
        self.b_values = np.linspace(1.0, 50.0, 10)  # Predefined set of b values to test
        self.num_workers = 8  # Number of processes to use in multiprocessing
        self.likelihood_cache = {}

    def estimate_sigma(self, m_0=0.5):
        factor = (m_0 ** 2 - 2 * m_0 + 2) ** (self.k / 2)
        return np.sqrt(self.sample_variance) / factor

    def reinitialize_near_bounds(self, params, bounds):
        for i, (param, (lower_bound, upper_bound)) in enumerate(zip(params, bounds)):
            if param <= lower_bound + 0.01 * (upper_bound - lower_bound) or param >= upper_bound - 0.01 * (
                    upper_bound - lower_bound):
                params[i] = np.random.uniform(lower_bound + 0.1 * (upper_bound - lower_bound),
                                              upper_bound - 0.1 * (upper_bound - lower_bound))
        return params

    def likelihood(self, params):
        params_tuple = tuple(np.round(params, decimals=6))  # Round to avoid precision issues

        # Return cached likelihood if available
        if params_tuple in self.likelihood_cache:
            return self.likelihood_cache[params_tuple]

        m_0, b, gamma = params
        sigma = self.estimate_sigma(m_0)
        prob_model = ProbEstimation(self.k, m_0, sigma, b, gamma, self.returns)
        likelihood_val = -prob_model.calc_likelihood()

        # Regularization terms
        gamma_regularization = len(self.returns) * (gamma - 0.5) ** 2
        b_regularization = len(self.returns) * (1.0 / b) ** 2

        likelihood_val += self.gamma_weight * gamma_regularization + self.b_weight * b_regularization
        self.likelihood_cache[params_tuple] = likelihood_val  # Cache the result

        return likelihood_val

    def perturb_parameters(self, params, bounds, step_size):
        perturbed_params = np.copy(params)

        # Apply random noise perturbation
        for i in range(len(params)):
            lower_bound, upper_bound = bounds[i]
            range_size = upper_bound - lower_bound

            # Random noise perturbation (Gaussian noise)
            perturbation = np.random.randn() * step_size * range_size

            # Apply the perturbation and clip the parameter within bounds
            perturbed_params[i] += perturbation
            perturbed_params[i] = np.clip(perturbed_params[i], lower_bound, upper_bound)

        return perturbed_params

    def basin_hopping(self, initial_params, bounds):
        current_params = np.copy(initial_params)
        current_likelihood = self.likelihood(current_params)
        step_size = self.step_size
        patience = 10
        improvement_count = 0

        for iteration in range(self.basin_iter):
            # Randomly perturb the parameters
            new_params = self.perturb_parameters(current_params, bounds, step_size)
            new_likelihood = self.likelihood(new_params)

            # If new parameters improve likelihood, accept them
            if new_likelihood < current_likelihood:
                current_params = new_params
                current_likelihood = new_likelihood
                step_size *= 0.9  # Decrease step size to fine-tune
                improvement_count = 0
            else:
                improvement_count += 1
                if improvement_count >= patience:
                    step_size *= 1.1  # Increase step size to explore more
                    improvement_count = 0
                    current_params = self.reinitialize_near_bounds(current_params, bounds)

        optimal_sigma = self.estimate_sigma(current_params[0])
        prob_model = ProbEstimation(self.k, current_params[0], optimal_sigma, current_params[1], current_params[2],
                                     self.returns)
        optimal_likelihood = prob_model.calc_likelihood()

        return current_params[0], current_params[1], current_params[2], optimal_sigma, optimal_likelihood

    def evaluate_b(self, b_value, initial_params, bounds):
        initial_params[1] = b_value  # Fix `b` to one of the predefined values
        m_0, b, gamma, sigma, global_likelihood = self.basin_hopping(np.copy(initial_params), bounds)
        return (m_0, b, gamma, sigma, global_likelihood)

    def optimize(self, initial_params=np.array([0.5, 10, 0.5])):
        bounds = [(0.2, 0.8), (1.0, 50.0), (0.05, 0.95)]
        best_likelihood = np.inf
        best_params = None

        # Start the timer
        start_time = time.time()

        # Parallelize over the range of predefined b values
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [executor.submit(self.evaluate_b, b, np.copy(initial_params), bounds) for b in self.b_values]

            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                m_0, b, gamma, sigma, global_likelihood = result
                if global_likelihood < best_likelihood:
                    best_likelihood = global_likelihood
                    best_params = [m_0, b, gamma, sigma]

        # End the timer
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Optimization completed in {elapsed_time:.2f} seconds")

        print(
            f"Best result: m_0={best_params[0]}, b={best_params[1]}, gamma={best_params[2]}, sigma={best_params[3]}, LL={best_likelihood}")

        return best_params


if __name__ == "__main__":
    # Example usage:
    sigma = 0.05
    k = 4
    m_0 = 0.3
    gamma = 0.5
    b = 18
    N = 5000

    # Generate data
    generate = GenerateData(sigma, k, m_0, gamma, b, N)
    vol_cp = generate.vol_cp
    vol = generate.vol
    returns = generate.returns

    optimizer = Optimizer(returns, k)

    # Perform optimization
    optimal_params = optimizer.optimize()

    print(f'Optimal Parameters: m_0={optimal_params[0]}, b={optimal_params[1]}, gamma={optimal_params[2]}, sigma={optimal_params[3]}')

    prob = ProbEstimation(k, m_0, sigma, b, gamma, returns)
    print(prob.calc_likelihood())
