from markov_switching_multifractal.opti import Optimizer
from markov_switching_multifractal.calc_prob import ProbEstimation
import yfinance as yf
import numpy as np


if __name__ == '__main__':
    # Define the ticker symbol for S&P 500
    ticker = "^GSPC"  # S&P 500 index symbol

    # Download historical data for the S&P 500 from 2000 to 2020
    data = yf.download(ticker, start="2009-04-15", end="2015-10-12")

    # Calculate the daily returns
    data['Daily Returns'] = data['Adj Close'].pct_change()

    # Drop the missing values (since the first return will be NaN)
    data = data.dropna()

    # Display the data
    print(data[['Adj Close', 'Daily Returns']])

    # Retrieve only the daily returns as a 1D NumPy array (N,)
    returns = data['Daily Returns'].values

    # Determine the number of values in the returns
    n_values = len(returns)

    # Create two sub-returns: one with the last 500 values, the other with the rest
    if n_values >= 500:
        sub_returns_last_500 = returns[-500:]
        returns_data = returns[:-500]
    else:
        sub_returns_last_500 = returns
        returns_data = []

    mean_returns_data = np.mean(returns_data)

    returns_data = returns_data - mean_returns_data

    returns = returns - np.mean(returns)

    # Initialize the optimizer
    optimizer = Optimizer(returns, k=4)

    # Perform local minimization followed by basin-hopping
    optimal_params = optimizer.optimize()

    print(f"Optimal Parameters: {optimal_params}")

    m_0 = optimal_params[0]
    sig = optimal_params[3]
    b = optimal_params[1]
    gamma = optimal_params[2]
    k = 4

    prob = ProbEstimation(k, m_0, sig, b, gamma, sub_returns_last_500)
    L = prob.calc_likelihood()
    print(L)
