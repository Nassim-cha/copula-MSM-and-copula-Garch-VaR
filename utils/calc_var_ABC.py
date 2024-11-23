from abc import ABC, abstractmethod


class SharedCacheCopulaMSMVaR:
    """
    A shared cache for StudentCopulaMSMVaR / GaussianCopulaMSMVaR /
    """
    cache = {}


class SharedCacheCopulaGarchVaR:
    """
    A shared cache for StudentCopulaGarchVaR / GaussianCopulaGarchVaR
    """
    cache = {}


class SharedCacheCopulaMRVaR:
    """
    A shared cache for StudentCopulaGarchVaR / GaussianCopulaGarchVaR
    """
    cache = {}


class VaRCalculationMethod(ABC):
    """
    Abstract Base Class (ABC) for VaR Calculation Methods that define a standard
    interface for stochastic volatility models and copula-based risk modeling.

    Each concrete subclass must implement the following methods:
    - model_params_insample: Determine stochastic volatility parameters based on in-sample data.
    - calculate_marginals_and_densities_in_sample: Calculate marginals and densities based on the model's parameters.
    - copula_or_correl_params_insample: Estimate the copula or correlation parameters based on marginals and densities.
    """

    @abstractmethod
    def model_params_insample(self):
        """
        Determine stochastic volatility parameters for a specific model using the in-sample data.

        This method optimizes or estimates the parameters of the stochastic volatility model
        based on historical in-sample returns data. These parameters could include things like
        volatility components, persistence factors, and other statistical characteristics of the model.

        Parameters:
        - in_sample_dict: dict
            A dictionary containing the in-sample returns for each index (ticker symbol as key).
        - k: int
            The number of components or parameters in the model (e.g., number of volatility components).

        Returns:
        - results_dict: dict
            A dictionary where the keys are the index names (ticker symbols) and the values
            are the optimized model parameters for each index.
        """
        pass

    @abstractmethod
    def calculate_marginals_and_densities_in_sample(self):
        """
        Calculate the marginals and densities of the in-sample returns based on the model and its parameters.

        After determining the model's parameters (from model_params_insample), this method calculates
        the marginal distributions and densities for the in-sample data. These values are used in
        subsequent copula estimation or other risk analysis.

        Parameters:
        - in_sample_dict: dict
            A dictionary containing the in-sample returns for each index (ticker symbol as key).
        - k: int
            The number of components or parameters in the model (e.g., number of volatility components).
        - in_sample_params: dict
            A dictionary containing the optimized model parameters for each index, as returned
            by the model_params_insample method.

        Returns:
        - marginals_array: np.ndarray
            A single stacked array of marginals for all tickers (columns represent different tickers).
        - densities_array: np.ndarray
            A single stacked array of densities for all tickers (columns represent different tickers).
        """
        pass

    @abstractmethod
    def copula_or_correl_params_insample(self):
        """
        Estimate the copula or correlation parameters based on the marginals and densities.

        Once the marginals and densities have been computed for the in-sample data, this method estimates
        the copula (or correlation) parameters to capture the dependence structure between the assets (tickers).
        The copula parameters are crucial for modeling joint distributions and dependencies between multiple assets.

        Parameters:
        - marginals: np.ndarray
            A stacked array of the marginal distributions for each index.
        - densities: np.ndarray
            A stacked array of the density values for each index.

        Returns:
        - copula_params: dict or np.ndarray
            The estimated copula or correlation parameters for the multivariate distribution.
        """
        pass

    @abstractmethod
    def integration_params_retrieval(self):
        """

        :return:
        """
        pass
