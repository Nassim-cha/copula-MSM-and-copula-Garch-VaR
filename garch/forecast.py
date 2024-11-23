from garch.estimation import ProbEstimation
import numpy as np


def calc_forecast(omega, alpha_vect, beta_vect, returns):

    prob = ProbEstimation(returns, omega, alpha_vect, beta_vect)
    # Calculate the conditional variances
    sigma2 = prob.calculate_conditional_variances()

    p = len(alpha_vect)
    q = len(beta_vect)

    last_returns = returns[-p:]
    last_sig2 = sigma2[-q:]

    forecast = omega + np.sum(alpha_vect * last_returns**2) + np.sum(beta_vect * last_sig2)

    return np.sqrt(forecast)