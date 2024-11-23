import numpy as np
from utils.utils import norm_cdf_array, norm_pdf_array, manual_reshape


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
    forecasted_vol = integrations_params_i
    nu, corr_matrix = unpack_copula_params(copula_params)
    num_points = grids.shape[0]

    x = grids / forecasted_vol

    cdf = norm_cdf_array(x)

    params = {"cdf": cdf, "nu": nu, "corr_matrix": corr_matrix}

    # Compute the Student copula density using nu and rho
    pdf = norm_pdf_array(x) / forecasted_vol

    density_prod = np.prod(pdf, axis=1)

    # Compute the Student copula density using nu and rho
    student_copula_density = copula_density(**params)

    density_func = manual_reshape(student_copula_density * density_prod, (num_points, 1))

    density_func = np.nan_to_num(density_func)

    # Sum over result points and multiply by step sizes to get the final integral result for all i
    integral_results = density_func * step_sizes

    return integral_results
