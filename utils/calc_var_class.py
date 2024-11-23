from data_loader.load_data import IndexReturnsRetriever
from utils.calc_integral.calc_integral import calc_grids_and_integrals_results
from utils.calc_integral.integration_algo import var_function
import numpy as np
import time


class ValueAtRiskCalcualtion:
    def __init__(
            self,
            tickers,
            start_date,
            in_sample_data_num,
            VaRCalculationMethod,
            end_date=None,
            num_points=100,
            weights=np.array([0.5, 0.5]),
            *args,
            **kwargs
    ):

        # input params retrieval
        self.num_points = num_points
        self.tickers = tickers
        self.start_date = start_date
        self.in_sample_data_num = in_sample_data_num
        self.VaRCalculationMethod = VaRCalculationMethod
        self.end_date = end_date
        self.weights = weights

        # data loading
        (self.in_sample_dict, self.rolling_windows_dict, self.mean_returns, self.end_date, self.out_sample_data,
         self.out_sample_N, self.dim, self.ptf_mean) = self.get_in_sample_data()

        # VaR params
        self.in_sample_params = self.retrieve_param_in_sample(*args, **kwargs)
        self.marginals, self.densities, self.vol_states_array = self.calc_marg_and_densities(*args, **kwargs)
        self.copula_params = self.calc_copula_params()
        self.integrations_params_t, self.integrations_params_static, self.grids_generations_params = (
            self.integration_params_retrieval())

        # VaR functions
        self.copula_function = VaRCalculationMethod.copula_density
        self.unpack_copula_params = VaRCalculationMethod.unpack_copula_params
        self.integrated_function = VaRCalculationMethod.integrated_function

    def get_in_sample_data(self):
        # Initialize the IndexReturnsRetriever (automatically downloads returns data)
        retriever = IndexReturnsRetriever(
            tickers=self.tickers,
            start_date=self.start_date,
            N=self.in_sample_data_num,
            weights=self.weights,
            end_date=self.end_date
        )

        # Retrieve in-sample and out-of-sample data
        return retriever.get_insample_data()

    def retrieve_param_in_sample(self, *args, **kwargs):
        # Call model_params_insample with in_sample_dict and any provided args and kwargs
        in_sample_params = self.VaRCalculationMethod.model_params_insample(self.in_sample_dict, *args, **kwargs)
        return in_sample_params

    def calc_marg_and_densities(self, *args, **kwargs):
        # Call calculate_marginals_and_densities_in_sample with in_sample_dict, in_sample_params, and any args/kwargs
        marginals, densities, vol_states_array = (
            self.VaRCalculationMethod.calculate_marginals_and_densities_in_sample(
                self.in_sample_dict,
                self.in_sample_params,
                *args,
                **kwargs
            )
        )
        return marginals, densities, vol_states_array

    def calc_copula_params(self):
        best_fit_copula = self.VaRCalculationMethod.copula_or_correl_params_insample(
            self.marginals,
            self.densities)
        copula_params = self.VaRCalculationMethod.copula_integrations_params(best_fit_copula)
        return copula_params

    def integration_params_retrieval(self):
        (integrations_params_t, integrations_params_static, grids_generations_params) = (
            self.VaRCalculationMethod.integration_params_retrieval(
            self.dim,
            self.rolling_windows_dict,
            self.in_sample_params,
            self.num_points,
            self.vol_states_array
        ))
        return integrations_params_t, integrations_params_static, grids_generations_params

    def calc_var(self, obj_var=0.05, first_guess=-3, second_guess=(-3.5, -2)):
        """
        Calculate the Value-at-Risk (VaR) as a vector of nested integrals.
        This function leverages nested intervals to approximate -inf bounds by -100 and
        iteratively builds on previously computed integrals to simplify calculations.

        Parameters:
        - obj_var: Objective Value-at-Risk threshold, used in bisection testing.
        - first_guess: Float, initial guess for bounds (default is -1.5).
        - second_guess: Tuple of two floats, bounds used for nested integration adjustment (default is (-2, -1)).

        Returns:
        - nested_results: List of computed results for each nested interval.
        """
        start_time = time.time()  # Start timer

        min_var_value = -7.5
        max_var_value = 0

        lower = -100
        upper = first_guess  # Using first_guess as the initial upper bound
        bounds = np.column_stack((lower * np.ones(self.out_sample_N), upper * np.ones(self.out_sample_N)))

        # Step 1: Calculate the initial integral over bounds -100 to first_guess
        results = self.compute_integral(
            bounds=bounds
        )

        # Step 2: where results < obj_var compute nested integral for first_guess,
        # and where results > obj_var for second_guess
        new_lower = np.where(results >= obj_var, second_guess[0],
                             first_guess)  # Use second_guess[0] where results >= obj_var, else use first_guess
        new_upper = np.where(results < obj_var, second_guess[1],
                             first_guess)  # Use second_guess[1] where results < obj_var, else use first_guess
        bounds = np.column_stack((new_lower, new_upper))

        # Create prev_upper array based on new_lower values
        prev_upper = np.where(new_lower == second_guess[0], second_guess[0], first_guess)

        new_result = self.compute_integral(
            bounds=bounds,
        )

        result_current = self.adjust_integral(
            new_result,
            results,
            bounds,
            upper * np.ones(self.out_sample_N))

        upper = bounds[:, 1]

        # Update bounds based on initial results and set up bisection bounds for each case
        bisection_bounds = np.empty((self.out_sample_N, 2))
        bisection_bounds[result_current > obj_var, 0] = min_var_value
        bisection_bounds[result_current > obj_var, 1] = second_guess[0]
        bisection_bounds[(result_current < obj_var) & (upper == first_guess), 0] = second_guess[0]
        bisection_bounds[(result_current < obj_var) & (upper == first_guess), 1] = first_guess
        bisection_bounds[(result_current < obj_var) & (upper == second_guess[1]), 0] = second_guess[1]
        bisection_bounds[(result_current < obj_var) & (upper == second_guess[1]), 1] = max_var_value
        bisection_bounds[(result_current > obj_var) & (upper == second_guess[1]), 0] = first_guess
        bisection_bounds[(result_current > obj_var) & (upper == second_guess[1]), 1] = second_guess[1]

        upper = bisection_bounds[:, 1]

        # Initialize upper_stack as True where upper is not equal to elements in second_guess, False otherwise
        upper_stack = ~np.isin(upper, list(second_guess))

        # Run bisection algorithm vectorized for each point's bounds
        final_results = self.bisection_algorithm(
            obj_var,
            bisection_bounds,
            result_current,
            upper_stack,
            prev_upper
        )

        var = final_results + self.ptf_mean

        end_time = time.time()  # End timer
        elapsed_time = end_time - start_time
        print(f"calc_var function execution time: {elapsed_time:.4f} seconds")

        return var

    def compute_integral(self, bounds):
        """
        Compute the integral for specified bounds.

        Parameters:
        - lower, upper: Bounds for integration.
        - Other parameters are model inputs.

        Returns:
        - result: Computed integral result for the specified bounds.

        """

        unique_var_values, unique_indices = np.unique(bounds, axis=0, return_inverse=True)

        result = calc_grids_and_integrals_results(
            T=self.out_sample_N,
            unique_var_values=unique_var_values,
            unique_indices=unique_indices,
            num_points=self.num_points,
            dim=self.dim,
            var_function=var_function,
            lower_bound=-5,
            upper_bound=5,
            grids_generations_params=self.grids_generations_params,
            integrations_params_t=self.integrations_params_t,
            integrations_params_static=self.integrations_params_static,
            copula_params=self.copula_params,
            integrated_function=self.integrated_function,
            copula_density=self.copula_function,
            unpack_copula_params=self.unpack_copula_params,
            weights=self.weights
        )
        return result

    def adjust_integral(
            self,
            new_result,
            prev_results,
            bounds,
            prev_upper
    ):
        """
        Adjust the current integral result based on new bounds calculation.

        Parameters:
        - new_result: Array of computed integral values for the current bounds.
        - prev_results: Array of previously computed integral values.
        - bounds: 2D array where each row represents the [lower, upper] bounds for the current interval.
        - prev_upper: Previous upper bound, used to determine if results should be added or subtracted.

        Returns:
        - adjusted_result: Array of adjusted integral values, representing the integral from -inf to the upper value of bounds.
        """
        # Initialize the adjusted result array with the same shape as prev_results
        adjusted_result = np.zeros_like(prev_results)

        # Loop over each entry to determine whether to add or subtract based on the bounds
        for i in range(len(prev_results)):
            lower, upper = bounds[i]

            # Use np.array_equal to check if lower bound matches the previous upper bound at index i
            if lower == prev_upper[i]:
                # Add new_result[i] to prev_results[i] if lower matches prev_upper
                adjusted_result[i] = prev_results[i] + new_result[i]
            else:
                # Otherwise, subtract new_result[i] from prev_results[i]
                adjusted_result[i] = prev_results[i] - new_result[i]

        return adjusted_result

    def bisection_algorithm(
            self,
            obj_var,
            bisection_bounds,
            prev_result,
            upper_stack,
            prev_upper,
            tolerance=1e-6
    ):
        """
        Perform vectorized bisection to refine integral calculations.

        Parameters:
        - obj_var: Target VaR threshold.
        - bisection_bounds: Array of shape (T, 2) for lower and upper bounds of each point.
        - T: Number of points.
        - prev_result: Array of previously computed integral results.
        - upper_stack: Boolean array to determine whether to stack with upper values or lower values.
        - tolerance: Tolerance for convergence.

        Returns:
        - final_values: Array of refined integral results after bisection.
        """
        # Initialize arrays for lower and upper bounds
        lower = bisection_bounds[:, 0]
        upper = bisection_bounds[:, 1]

        # Loop until all points are within tolerance
        while np.any(upper - lower > tolerance):
            mid = (lower + upper) / 2

            # Use upper_stack to determine bounds for each point
            bounds = np.where(upper_stack[:, None], np.column_stack((lower, mid)), np.column_stack((mid, upper)))

            # Calculate mid_result for each point based on the selected bounds
            mid_result = self.compute_integral(
                bounds=bounds,
            )

            # Adjust result based on whether we're adding or subtracting the new integral
            result_current = self.adjust_integral(mid_result, prev_result, bounds, prev_upper)

            # Check if result_current is all zeros and break if so
            if np.all(result_current == 0):
                print("Breaking loop as result_current is all zeros.")
                break

            # Update bounds and `upper_stack` based on obj_var threshold
            upper_stack = result_current < obj_var
            lower = np.where(~upper_stack, lower, mid)
            upper = np.where(upper_stack, upper, mid)

            # Update previous results to reflect current results for the next iteration
            prev_result = result_current
            prev_upper = mid

        # Final values after bisection convergence
        final_values = (lower + upper) / 2

        return final_values
