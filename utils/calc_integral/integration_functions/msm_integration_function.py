import numpy as np
from utils.utils import manual_reshape, norm_cdf_array


def integrated_function(
        grids,
        step_sizes,
        copula_params,
        integrations_params_i,
        integrations_params_static,
        copula_density,
        unpack_copula_params
):
    """
    Computes the integrand using provided parameters.

    Parameters:
    - flattened_grids: np.ndarray
        The grid points to evaluate the function on.
    - params: np.ndarray
        1D array containing all necessary parameters for the integrand function.

    Returns:
    - np.ndarray
        Evaluated density for each grid point.
    """

    forecasts_by_states = integrations_params_i[0]
    forecasts = integrations_params_i[1]
    nu, corr_matrix = unpack_copula_params(copula_params)
    num_points = grids.shape[0]
    unique_vol_states = integrations_params_static

    x = grids[:, :, np.newaxis] / unique_vol_states[np.newaxis, :, :]

    cdf = np.sum(forecasts_by_states * norm_cdf_array(x), axis=2)

    params = {"cdf": cdf, "nu": nu, "corr_matrix": corr_matrix}

    # Compute the Student copula density using nu and rho
    student_copula_density = copula_density(**params)

    student_copula_density_array = manual_reshape(student_copula_density, (num_points, 1))
    # Sum over result points and multiply by step sizes to get the final integral result for all i
    integral_results = np.sum(student_copula_density_array * step_sizes, axis=0) * forecasts

    return integral_results
