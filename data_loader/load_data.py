from dataclasses import dataclass
import pandas as pd
import yfinance as yf
import numpy as np

# Shared cache for index returns and in-sample data
class SharedCacheIndexReturns:
    returns_cache = {}
    insample_cache = {}

@dataclass
class IndexReturnsRetriever:
    tickers: list  # List of index tickers (e.g., ['^GSPC', '^IXIC', '^DJI'])
    start_date: str  # Start date for in-sample estimation
    N: int  # Number of returns for in-sample estimation
    weights: np.array
    end_date: str = None  # End date for historical data, defaults to today

    def __post_init__(self):
        # Check if returns are already cached
        cache_key = (tuple(self.tickers), self.start_date, self.end_date)
        if cache_key in SharedCacheIndexReturns.returns_cache:
            print(f"Using cached returns data for tickers {self.tickers}")
            self.returns = SharedCacheIndexReturns.returns_cache[cache_key]
        else:
            # Download index returns automatically
            self.returns = self.get_index_returns(self.tickers, self.start_date, self.end_date)
            # Cache the returns data
            SharedCacheIndexReturns.returns_cache[cache_key] = self.returns
            print(f"Downloaded and cached returns data for tickers {self.tickers}")

        # Ensure the DataFrame is sorted by date and start_date is a valid date
        self.returns = self.returns.sort_index()
        self.start_date = pd.to_datetime(self.start_date)

        # Filter returns from start date onwards
        self.returns = self.returns[self.returns.index >= self.start_date]

        # Ensure only dates where all indexes have data are used
        self.returns = self.returns.dropna()

    def get_index_returns(self, tickers, start_date, end_date=None):
        """
        Retrieve index returns for the specified tickers from Yahoo Finance.

        Parameters:
        tickers: list of str
            List of ticker symbols to retrieve from Yahoo Finance.
        start_date: str
            The start date for the historical data.
        end_date: str, optional
            The end date for the historical data. If None, it defaults to today.

        Returns:
        returns_df: pd.DataFrame
            A DataFrame of daily returns for each ticker with dates as the index.
        """
        # Download the data from Yahoo Finance
        data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']

        # Drop rows with missing data (only keep dates where all tickers have data)
        data = data.dropna()

        # Calculate daily log returns
        log_returns = np.log(data / data.shift(1)).dropna() * 100

        return log_returns

    def get_insample_data(self):
        """
        Retrieve in-sample and out-of-sample data based on the specified start date and N returns.

        Returns:
        in_sample_dict: dict
            A dictionary where the keys are the index names (ticker symbols)
            and the values are arrays of centered in-sample returns.
        rolling_windows_dict: dict of dict
            A dictionary where the keys are the rolling window end dates,
            and the values are dictionaries with ticker names as keys and their corresponding
            centered rolling window returns as values.
        mean_returns: dict
            A dictionary where the keys are the index names (ticker symbols)
            and the values are the mean of the in-sample returns for each index.
        end_date: pd.Timestamp
            The end date of the in-sample estimation period.
        out_sample_data: pd.DataFrame
            The returns that are not in the in-sample period.
        remaining_out_sample: int
            The number of returns in the out-sample period.
        dim: int
            The number of indexes (tickers) being analyzed.
        """
        # Check if in-sample data is already cached
        cache_key = (tuple(self.tickers), self.start_date, self.N)
        if cache_key in SharedCacheIndexReturns.insample_cache:
            print(f"Using cached in-sample data for tickers {self.tickers}")
            return SharedCacheIndexReturns.insample_cache[cache_key]

        # Check if there are enough returns for in-sample estimation
        if len(self.returns) < self.N:
            raise ValueError(
                f"Not enough returns after the start date for in-sample estimation. "
                f"Required: {self.N}, Available: {len(self.returns)}")

        # Get the in-sample data (first N returns)
        in_sample_data = self.returns.iloc[:self.N]

        dim = self.returns.shape[1]

        # Calculate the mean returns for each index during the in-sample period
        mean_returns = in_sample_data.mean(axis=0)

        ptf_mean = np.sum(mean_returns.values * self.weights)

        # Center the in-sample returns by subtracting the mean
        in_sample_centered = in_sample_data - mean_returns

        # Create the in-sample dictionary with tickers as keys and centered returns as values
        in_sample_dict = {ticker: in_sample_centered[ticker].values for ticker in self.returns.columns}

        end_date = in_sample_data.index[-1]

        # Get the out-of-sample data (returns that are not in the in-sample period)
        out_sample_data = self.returns.iloc[self.N:]

        # Calculate the number of remaining out-sample returns
        remaining_out_sample = len(out_sample_data)

        # Generate a dictionary of rolling windows of centered in-sample returns for each ticker
        rolling_windows_dict = {}
        for i in range(remaining_out_sample):
            rolling_window = self.returns.iloc[i:i + self.N]
            rolling_window_centered = rolling_window - mean_returns  # Center the rolling window by subtracting the mean
            rolling_window_end_date = rolling_window.index[-1]  # Get the last date for the key
            rolling_windows_dict[rolling_window_end_date] = {
                ticker: rolling_window_centered[ticker].values for ticker in self.returns.columns
            }

        # Cache the in-sample data, now including all 7 values
        SharedCacheIndexReturns.insample_cache[cache_key] = (
            in_sample_dict,
            rolling_windows_dict,
            mean_returns.to_dict(),
            end_date,
            out_sample_data,
            remaining_out_sample,
            dim,
            ptf_mean
        )
        print(f"Cached in-sample data for tickers {self.tickers}")

        return (in_sample_dict,
                rolling_windows_dict,
                mean_returns.to_dict(),
                end_date,
                out_sample_data,
                remaining_out_sample,
                dim,
                ptf_mean)


if __name__ == "__main__":
    # List of ticker symbols (e.g., S&P 500, NASDAQ, Dow Jones, etc.)
    tickers = ['^GSPC', '^IXIC', '^DJI']  # S&P 500, NASDAQ, Dow Jones

    # Initialize the IndexReturnsRetriever (automatically downloads returns data)
    retriever = IndexReturnsRetriever(tickers=tickers, start_date='2020-01-01', N=50)

    # Retrieve in-sample and out-of-sample data
    in_sample_dict, rolling_windows_dict, mean_returns, end_date = retriever.get_insample_data()

    print(f"In-sample centered returns by index: {in_sample_dict}")
    print(f"Mean returns by index: {mean_returns}")
    print(f"End date of in-sample estimation: {end_date}")
    print(f"First rolling window centered returns: \n{rolling_windows_dict[next(iter(rolling_windows_dict))]}")
