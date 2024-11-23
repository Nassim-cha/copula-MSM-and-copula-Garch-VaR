from markov_switching_multifractal.calc_prob import ProbEstimation


def calc_prob_last_state(k, m_0, sigma, b, gamma, returns):

    prob = ProbEstimation(k, m_0, sigma, b, gamma, returns)

    state_prob_t, _ = prob.calc_state_prob()


    return prob.vol_states
