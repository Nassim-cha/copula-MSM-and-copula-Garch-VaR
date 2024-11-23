from opti import GarchOptimizer
from estimation import ProbEstimation
import yfinance as yf
from matplotlib import pyplot as plt


if __name__ == '__main__':
    # Define the ticker symbol for S&P 500
    ticker = "^GSPC"  # S&P 500 index symbol

    # Download historical data for the S&P 500 from 2000 to 2020
    data = yf.download(ticker, start="2000-01-01", end="2020-12-31")

    # Calculate the daily returns
    data['Daily Returns'] = data['Adj Close'].pct_change()

    # Drop the missing values (since the first return will be NaN)
    data = data.dropna()

    # Display the data
    print(data[['Adj Close', 'Daily Returns']])

    # Retrieve only the daily returns as a 1D NumPy array (N,)
    returns = data['Daily Returns'].values * 100

    # Initialize the optimizer
    optimizer = GarchOptimizer(returns, p_max=3, q_max=3)

    # Perform local minimization followed by basin-hopping
    result = optimizer.optimize()

    omega, alpha_vect, beta_vect = optimizer.unpack_garch_parameters(result)

    prob = ProbEstimation(returns, omega, alpha_vect, beta_vect)

    eps = prob.calculate_eps_t()

    # Plotting the epsilon values (residuals)
    plt.figure(figsize=(10, 6))
    plt.plot(eps, label="Eps_t (Residuals)", color='b')
    plt.title("Residuals (Eps_t) Over Time")
    plt.xlabel("Time")
    plt.ylabel("Eps_t")
    plt.legend()
    plt.grid(True)
    plt.show()

