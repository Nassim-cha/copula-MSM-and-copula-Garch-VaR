# Copula-MSM and Copula-Garch Value-at-Risk backtest
#_Backtest tool to assess VaR of portfolio using copula-MSM copula-Garch and Copula-Unscented Kalman Filter models_

Compute optimized model parameters during in-sample estimation.
Calculate the Value-at-Risk for a designed out-of-sample period using forecasted joined density of assets.
Models are replication of research papers : "Forecasting Market Risk of Portfolios:
Copula-Markov Switching Multifractal Approach" (M.Segnon M.Trede 2017) and "Improved unscented kalman smoothing for stock volatility estimation" (O.Zoeter, A.Ypma, T.Heskes 2004)


- **Copula MSM Model**: Implements multifractal stochastic volatility modeling.
- **Copula GARCH Model**: Supports GARCH-based volatility modeling.
- **Backtesting Framework**: Performs VaR backtesting using historical portfolio data.
- **Customizable Parameters**: Easily configure key parameters for MSM and GARCH models.
- **Parallelized Computation**: Optimized for high-performance computation using Python libraries.

## Features

- **dataclass** and cache to retrieve assets returns and in-sample / out-of-sample classification
- Create of efficient **optimizer** for each model and copula parameters retrieval using inference for margnins method 
- Usage of **compilers** to improve code efficiency with numba
- **Abstract Based Classes** for each model and copulas VaR calculation
- Creation of grids to approximate nested integration of joined density in a limited time frame 
- VaR solving using a bissection algorithm
- **parrallelization** of VaR resolution with joblib

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Contributing](#contributing)

## Installation

1. Clone the repository:  

   git clone https://github.com/your-username/project-name.git
   cd project-name

Install the dependencies and devDependencies and start the server.

```sh
pip install -r requirements.txt
```

## Usage
Run the main file.
Adapt ValueAtRiskCalcualtion by using different models and copulas and the function to plot results.


## Project Structure

```sh
├── Value-at-Risk backtest/ 
    |main.py # Execute code 
    ├── data_loader/ # Load data
    |    load_data.py
    ├── copulas/ # Copula inference and density calculation code 
        ├── gaussian/ 
        |    gaussian_estimation.py
        |    inference_for_margins.py
        |    opti.py
        ├── student/ 
        |    student.py
        |    inference_for_margins.py
        |    opti.py
        ├── plackett/ 
        |    plackett.py
        |    inference_for_margins.py
        |    opti.py    
    ├── garch/ # In-sample estimation for Garch model
    |    generate_data.py # generate test data to test optimizer
    |    prob_estimation.py # model estimation for given parameters
    |    forecast.py # calculate the volatility forecast
    |    optimize.py # find best parameters
    |    test.py # test model on market data
    ├── markov_switching_multifractal/ # In-sample estimation for Garch model
    |    generate_data.py # generate test data to test optimizer
    |    calc_prob.py # model estimation (hamilton filter probabilities) for given parameters 
    |    input_forecast.py # calculate the volatility forecasts
    |    optimize.py # find best parameters
    |    test.py # test model on market data
    ├── kalman_mean_reverting/ # In-sample estimation for Kalman model
    |    generate_data.py # generate test data to test optimizer
    |    estimate.py # model estimation for given parameters
    |    forecast.py # calculate the volatility forecast
    |    optimize.py # find best parameters
    ├── utils/
    |    calc_var_class.py # class to calculate the VaR with a ABC class model as argument
    |    calc_var_ABC.py # Abstract Base Class for all the models to retrieve the joined density
    |    factory.py # allows to link a copula ABC to an estimation-type ABC 
    |    normal_distributions.py 
        ├── copula/
        |    gaussian_estimation.py
        |    student_estimaiton.py
        |    plackett_estimation.py
        ├── model/
        |    garch_estimaiton.py
        |    mean_reverting.py
        |    msm_estimation.py
 ```

## Development

Contributions are welcome! Please:
1. Fork the repo.
2. Create a feature branch.
3. Submit a pull request.
