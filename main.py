from utils.calc_var_class import ValueAtRiskCalcualtion
from utils.factory import ValueAtRiskCalculationFactory
import matplotlib.pyplot as plt


def plot_var_and_returns(msm_var, garch_var, portfolio_returns):
    n = len(msm_var)  # Assuming all arrays are the same length
    x = range(n)      # X-axis values as a range from 0 to n-1

    plt.figure(figsize=(10, 6))
    plt.plot(x, msm_var, label='MSM VaR', linestyle='-', marker='', alpha=0.8)
    plt.plot(x, garch_var, label='GARCH VaR', linestyle='--', marker='', alpha=0.8)
    plt.plot(x, portfolio_returns, label='Portfolio Returns', linestyle=':', marker='', alpha=0.8)

    plt.title("VaR and Portfolio Returns Over Time")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":

    start_date = '2009-04-15'
    end_date = '2015-10-12'
    # List of ticker symbols (e.g., S&P 500, NASDAQ, Dow Jones, etc.)
    tickers = ['^GSPC', '^IXIC']
    # User can calculate VaR for portfolios of more than two assets
    # ex ['^GSPC', '^IXIC', '^DJI']  # S&P 500, NASDAQ, Dow Jones
    # But needs to specify weights when initiating ValueAtRiskCalcualtion class

    # number of days of the insample period
    N = 1135

    # Choose the estimation method type
    estimation_type = 'garch'
    copula_type = 'student'

    # Create the var_calculator instance
    var_calculator = ValueAtRiskCalculationFactory.create_var_calculator(copula_type=copula_type,
                                                                         estimation_type=estimation_type)

    calc_VaR_student_garch = ValueAtRiskCalcualtion(tickers,
                                                    start_date,
                                                    N,
                                                    var_calculator,
                                                    end_date,
                                                    num_points=100)
    # use larger values for num_points for more precision
    # Add weights=np.array([1/3, 1/3, 1/3]) if calculation over 3 assets portfolios

    garch_var = calc_VaR_student_garch.calc_var()

    # Choose the estimation method type
    estimation_type = 'msm'
    copula_type = 'student'

    # Create the var_calculator instance
    var_calculator = ValueAtRiskCalculationFactory.create_var_calculator(copula_type=copula_type,
                                                                         estimation_type=estimation_type)

    calc_VaR_student_MSM = ValueAtRiskCalcualtion(tickers,
                                                  start_date,
                                                  N,
                                                  var_calculator,
                                                  end_date,
                                                  num_points=100,
                                                  k=4)

    msm_var = calc_VaR_student_MSM.calc_var()

    portfolio_returns = calc_VaR_student_garch.out_sample_data.mean(axis=1).to_numpy()

    plot_var_and_returns(msm_var, garch_var, portfolio_returns)
