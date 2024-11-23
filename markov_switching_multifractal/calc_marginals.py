from markov_switching_multifractal.calc_prob import ProbEstimation
from markov_switching_multifractal.generate_data import GenerateData
import numpy as np
from matplotlib import pyplot as plt


def calc_marginals(k, m_0, sigma, b, gamma, returns):

    prob = ProbEstimation(k, m_0, sigma, b, gamma, returns)

    state_prob_t, _ = prob.calc_state_prob()
    cond_marg_vect, cond_eps = prob.calc_state_marginals()

    eps = np.sum(state_prob_t * cond_eps, axis=1)
    state_prob_t = state_prob_t[1:, :]
    cond_marg_vect = cond_marg_vect[:-1, :]

    return np.sum(state_prob_t * cond_marg_vect, axis=1), eps, prob.vol_states


def calc_densities(k, m_0, sigma, b, gamma, returns):

    prob = ProbEstimation(k, m_0, sigma, b, gamma, returns)

    state_prob_t, cond_prob_vect = prob.calc_state_prob()

    state_prob_t = state_prob_t[1:, :]
    cond_prob_vect = cond_prob_vect[:-1, :]

    return np.sum(state_prob_t * cond_prob_vect, axis=1)


def calc_forecasts(k, m_0, sigma, b, gamma, returns):
    prob = ProbEstimation(k, m_0, sigma, b, gamma, returns)

    state_prob_t, _ = prob.calc_state_prob()

    return state_prob_t[-1, :]


if __name__ == "__main__":
    # Parameters
    sigma = 0.05
    k = 3
    m_0 = 0.5
    gamma = 0.5
    b = 2
    N = 500

    # Generate data
    generate = GenerateData(sigma, k, m_0, gamma, b, N)
    vol_cp = generate.vol_cp
    vol = generate.vol
    returns = generate.returns

    marginals, eps, vol_states = calc_marginals(k, m_0, sigma, b, gamma, returns)
    densities = calc_densities(k, m_0, sigma, b, gamma, returns)

    innovations = generate.eps

    # Create subplots
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))

    # First subplot: Marginals
    axs[0].plot(marginals, label='Marginals')
    axs[0].set_title('Marginals')
    axs[0].set_xlabel('Index')
    axs[0].set_ylabel('Marginal Value')
    axs[0].legend()

    # Second subplot: eps and Innovations
    axs[1].plot(eps, label='eps', color='blue')
    axs[1].plot(innovations, label='Innovations', color='orange', linestyle='--')
    axs[1].set_title('eps and Innovations')
    axs[1].set_xlabel('Index')
    axs[1].set_ylabel('Value')
    axs[1].legend()

    # Show the plots
    plt.tight_layout()
    plt.show()

    print(calc_forecasts(k, m_0, sigma, b, gamma, returns))
