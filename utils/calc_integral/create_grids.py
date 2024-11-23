import numpy as np
from numba import njit


@njit
def create_nested_grid(num_points, dim, g, var, lower_bound, upper_bound, grids_generation_params, weights):
    """
    Generates a nested grid and step sizes for multidimensional integration.

    Parameters:
    - num_points: int
        Number of points in each dimension.
    - dim: int
        Dimensionality of the grid.
    - grid: np.ndarray
        Array to store generated grid points.
    - step_size: np.ndarray
        Array to store the step sizes for integration.
    - g: function
        Function defining the dynamic bounds for the innermost dimension.
    - var: float
        Variable used in the function g to determine bounds.
    - lower_bound, upper_bound: float
        Bounds for the outer dimensions of the grid.
    """

    densities, x_values, step_size_array, params = grids_generation_params
    grid, step_size = create_grids_with_dynamic_inner_bounds(
        lower_bound,
        upper_bound,
        num_points,
        dim,
        g,
        var,
        x_values,
        densities,
        step_size_array,
        params,
        weights
    )

    return grid, step_size


# Function to recursively generate grids and step sizes for multidimensional integration
@njit
def generate_grids_and_deltas(
        current_dim,
        current_bounds,
        previous_points,
        lower_bound,
        upper_bound,
        num_points,
        dim,
        g,
        grids_creator,
        delta_product,
        point_index,
        var,
        x_values,
        densities,
        step_size_array,
        params,
        weights
):
    """
    Recursively generates grids and their corresponding delta products (step sizes).

    Parameters:
    - current_dim: int
        Current dimension being processed.
    - current_bounds: tuple
        Bounds for the current dimension.
    - previous_points: np.ndarray
        Array holding points from the previous dimensions.
    - current_delta_product: float
        Product of deltas from previous dimensions.
    - lower_bound, upper_bound: float
        Bounds for the integration in the outer dimensions.
    - num_points: int
        Number of grid points per dimension.
    - dim: int
        Total number of dimensions.
    - g: function
        Function defining the dynamic bounds for the innermost dimension.
    - grids_creator: np.ndarray
        Array to store generated grid points.
    - delta_product: np.ndarray
        Array to store delta products (step sizes).
    - point_index: np.ndarray
        Current index for the grid point being processed.
    - var: float
        Variable used in the function g to determine bounds.
    - x_values: np.ndarray
        Array of 1D grid values for each dimension.
    - densities: np.ndarray
        Array of densities corresponding to each grid value in x_values.
    - params: np.ndarray
        Array of parameters to select density indices dynamically.
    """

    if current_dim == dim - 1:
        # Get dynamic bound from function g
        dynamic_upper_bound = g(previous_points, var[1], weights).item()
        dynamic_lower_bound = g(previous_points, var[0], weights).item()
        dynamic_lower_bound = max(dynamic_lower_bound, lower_bound)
        # Filter x_values within [adjusted_lower, dynamic_bound]
        valid_indices = np.where((x_values > dynamic_lower_bound) & (x_values <= dynamic_upper_bound))[0]
        inner_points = x_values[valid_indices]
        deltas_points = step_size_array[valid_indices]

        for i in range(len(valid_indices)):

            # Use valid inner points if available; otherwise, repeat the last valid point or set a default
            idx = valid_indices[i]
            grids_creator[point_index[0], current_dim] = inner_points[i]

            for j in range(len(params)):
                k = int(params[j, current_dim])

                delta_product[point_index[0], j] *= densities[current_dim - 1, k, idx] * deltas_points[i]

            point_index[0] += 1

    else:
        # Get x_values within [adjusted_lower, current_bounds[1]]
        valid_indices = np.where((x_values >= lower_bound) & (x_values <= current_bounds[1]))[0]
        outer_points = x_values[valid_indices]
        delta_points = step_size_array[valid_indices]

        for i in range(len(valid_indices)):

            # Use valid inner points if available; otherwise, repeat the last valid point or set a default
            idx = valid_indices[i]
            for j in range(point_index[0], point_index[0] + len(valid_indices) ** (dim - current_dim - 1)):
                if j >= grids_creator.shape[0]:  # Ensure we don’t go out of bounds
                    break
                grids_creator[j, current_dim] = outer_points[i]

                for l in range(params.shape[0]):
                    # Multiply current delta product by density value at this index
                    k = int(params[l, current_dim])
                    delta_product[j, l] *= densities[current_dim - 1, k, idx] * delta_points[i]

                # Define bounds for recursion
                next_bounds = (lower_bound, upper_bound)

            # Recurse to the next dimension
            generate_grids_and_deltas(
                current_dim + 1,
                next_bounds,
                np.append(previous_points, outer_points[i]),
                lower_bound,
                upper_bound,
                num_points,
                dim,
                g,
                grids_creator,
                delta_product,
                point_index,
                var,
                x_values,
                densities,
                step_size_array,
                params,
                weights
            )

            for j in range(point_index[0], point_index[0] + len(valid_indices) ** (dim - current_dim - 1)):
                for l in range(params.shape[0]):
                    delta_product[j, l] = 1


# Function to create the grid with dynamic inner bounds
@njit
def create_grids_with_dynamic_inner_bounds(
        lower_bound,
        upper_bound,
        num_points,
        dim,
        g,
        var,
        x_values,
        densities,
        step_size_array,
        params,
        weights
):
    """
    Creates a grid for a nested integral in multiple dimensions with dynamic bounds.

    Parameters:
    - lower_bound, upper_bound: float
        Fixed bounds for the outer dimensions.
    - num_points: int
        Number of grid points per dimension.
    - dim: int
        Total number of dimensions.
    - g: function
        Function that defines the upper bound for the innermost dimension.
    - var: float
        Variable used in the function g to determine bounds.

    Returns:
    - grids_creator: np.ndarray
        Generated grid points.
    - delta_product: np.ndarray
        Step sizes for each grid point.
    """
    total_points = num_points ** dim
    grids_creator = np.zeros((total_points, dim))
    N = params.shape[0]
    delta_product = np.ones((total_points, N))
    point_index = np.array([0])

    generate_grids_and_deltas(
        current_dim=0,
        current_bounds=(lower_bound, upper_bound),
        previous_points=np.zeros(0),
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        num_points=num_points,

        dim=dim,
        g=g,
        grids_creator=grids_creator,
        delta_product=delta_product,
        point_index=point_index,
        var=var,
        x_values=x_values,
        densities=densities,
        step_size_array=step_size_array,
        params=params,
        weights=weights
    )

    sliced_grids_creator = grids_creator[:point_index[0], :]
    sliced_delta_product = delta_product[:point_index[0], :]

    return sliced_grids_creator, sliced_delta_product