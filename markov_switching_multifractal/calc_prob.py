import numpy as np
import itertools
from numba import jit
from scipy.stats import norm


@jit(nopython=True)
def calc_state_prob_numba(cond_prob_vect, transi_probs, N, nb_states):
    state_probs = np.zeros((N, nb_states))

    # Initialize previous state probabilities with equal probabilities
    equi_prob = 1 / nb_states
    prev_state = np.full(nb_states, equi_prob)

    for i in range(N):
        cond_prob = cond_prob_vect[i, :]
        bayesian_upd = calc_bayes_upd_numba(prev_state, cond_prob, transi_probs, nb_states)

        # Manually check for the -1.0 value in bayesian_upd
        error_flag = False
        for val in bayesian_upd:
            if val == -1.0:
                error_flag = True
                break

        if error_flag:
            return np.full((N, nb_states), -1.0)  # Return an array with invalid values to indicate failure

        state_probs[i] = bayesian_upd
        prev_state = bayesian_upd

    return state_probs


@jit(nopython=True)
def calc_likelihood_numba(state_probs, cond_prob_vect, transi_probs, N):
    L = 0.0
    for i in range(1, N):

        likelihood_term = np.dot(np.dot(transi_probs, state_probs[i-1, :]), cond_prob_vect[i, :])

        if likelihood_term <= 0:
            return -np.inf

        L += np.log(likelihood_term)

    return L


@jit(nopython=True)
def calc_bayes_upd_numba(prev_state, cond_prob, transi_probs, nb_states):
    scaling_factor = 0.0
    transition_prob = np.zeros(nb_states)
    state_prob = np.zeros(nb_states)

    for i in range(nb_states):
        transition_prob[i] = np.sum(transi_probs[i, :] * prev_state)

    for k in range(nb_states):
        prob = transition_prob[k] * cond_prob[k]
        scaling_factor += prob
        state_prob[k] = prob

    if scaling_factor == 0:
        return None

    state_prob /= scaling_factor

    return state_prob


class ProbEstimation:
    def __init__(self, k, m_0, sigma, b, gamma, returns):
        self.k = k
        self.m_0 = m_0
        self.sigma = sigma
        self.b = b
        self.gamma = gamma
        self.returns = returns
        self.nb_states = 2 ** self.k
        self.N = len(self.returns)
        self.transi_mat = self.calc_vect_combinations()
        self.transi_probs = self.calc_transition_probabilities()
        self.vol_states = self.calc_vol_states()

    def calc_vect_combinations(self):
        # Use a list to store combinations
        combinations = list(itertools.product([self.m_0, 2 - self.m_0], repeat=self.k))
        return np.array(combinations)

    def calc_transition_probabilities(self):
        # Precompute gamma_k, p_values, and q_values
        gamma_k = 1 - (1 - self.gamma) ** (self.b ** np.arange(self.transi_mat.shape[1]))
        p_values = 1 - gamma_k / 2
        q_values = 1 - p_values

        # Use broadcasting and vectorized operations to calculate transition probabilities
        transi_probs_array = np.prod(np.where(self.transi_mat[:, None, :] == self.transi_mat[None, :, :],
                                              p_values, q_values), axis=2)

        return transi_probs_array

    def calc_vol_states(self):
        vol_states_array = np.zeros(self.nb_states)
        for i in range(self.nb_states):
            product = np.prod(self.transi_mat[i])
            vol_states_array[i] = np.sqrt(product) * self.sigma
        return vol_states_array

    def calc_state_prob(self):
        # Vectorized calculation of the conditional probability matrix
        sigma_matrix = self.vol_states[None, :]  # Shape (1, nb_states)
        returns_matrix = self.returns[:, None]  # Shape (N, 1)

        # Using the formula for norm.pdf directly, avoiding the loop
        cond_prob_vect = (1 / (sigma_matrix * np.sqrt(2 * np.pi))) * np.exp(
            -0.5 * (returns_matrix / sigma_matrix) ** 2)

        # Call the Numba-optimized function
        return calc_state_prob_numba(cond_prob_vect, self.transi_probs, self.N, self.nb_states), cond_prob_vect

    def calc_state_marginals(self):
        # Vectorized calculation of the conditional probability matrix
        sigma_matrix = self.vol_states[None, :]  # Shape (1, nb_states)
        returns_matrix = self.returns[:, None]  # Shape (N, 1)

        eps = returns_matrix / sigma_matrix
        # Using the formula for norm.pdf directly, avoiding the loop
        cond_marg_vect = norm.cdf(eps)

        # Call the Numba-optimized function
        return cond_marg_vect, eps

    def calc_likelihood(self):
        state_probs, cond_prob_vect = self.calc_state_prob()

        # Check if state_probs contains the error value (-1.0), indicating failure
        if np.any(state_probs == -1.0):
            return -np.inf

        # Call the Numba-optimized likelihood calculation function
        return calc_likelihood_numba(state_probs, cond_prob_vect, self.transi_probs, self.N)
