from scipy.special import erf
import numpy as np

def norm_cdf_array(x, mean=0, std=1):
    """
    Compute the CDF of a normal distribution for an n-dimensional array of values.

    Parameters:
    - x: np.ndarray, array of values to compute the CDF at (can be any shape).
    - mean: float, mean of the distribution (default is 0).
    - std: float, standard deviation of the distribution (default is 1).

    Returns:
    - np.ndarray: CDF values of the normal distribution at each element of x, with the same shape as x.
    """
    # Standardize the input array
    z = (x - mean) / std

    # Calculate the CDF using the error function (erf)
    result = 0.5 * (1 + erf(z / np.sqrt(2)))

    return result

def norm_pdf_array(x, mean=0, std=1):
    """
    Compute the PDF of a normal distribution for an n-dimensional array of values without using external libraries.

    Parameters:
    - x: np.ndarray, array of values to compute the PDF at (can be any shape).
    - mean: float, mean of the distribution (default is 0).
    - std: float, standard deviation of the distribution (default is 1).

    Returns:
    - np.ndarray: PDF values of the normal distribution at each element of x, with the same shape as x.
    """
    # Standardize the input array
    z = (x - mean) / std

    # Calculate the PDF using the normal distribution formula
    result = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * z**2)

    return result


# Function to manually reshape arrays
def manual_reshape(
        arr,
        new_shape
):
    """
    Manually reshapes a 1D array to a 2D array.

    Parameters:
    - arr: np.ndarray
        Array to reshape.
    - new_shape: tuple
        Desired shape for the array.

    Returns:
    - np.ndarray
        Reshaped array.
    """
    rows, cols = new_shape
    reshaped = np.zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
            reshaped[i, j] = arr[i * cols + j]
    return reshaped