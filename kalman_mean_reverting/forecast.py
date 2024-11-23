from kalman_mean_reverting.estimate import KalmanFilterVolEstimation
import numpy as np


def calc_forecast(returns, a, l, q):
    n = len(returns)

    # Example: Run the Kalman Filter correlation estimation
    kalman_filter = KalmanFilterVolEstimation(a, l, q, l, q, n, returns)
    forecasts = kalman_filter.forecasts

    return np.exp(forecasts)
