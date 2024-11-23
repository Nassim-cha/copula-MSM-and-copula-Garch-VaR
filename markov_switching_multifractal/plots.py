from markov_switching_multifractal.generate_data import GenerateData
from markov_switching_multifractal.calc_prob import ProbEstimation
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def regroup_and_plot(results):
    """
    Plots total_error as a function of (b, gamma) for all (b_weight, gamma_weight) combinations
    in the same plot, using different colors for different (b_weight, gamma_weight) pairs.
    """
    # Convert the results to a DataFrame for easy handling
    df = pd.DataFrame(results)

    # Create a 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Get all unique combinations of b_weight and gamma_weight
    weight_combinations = df[['b_weight', 'gamma_weight']].drop_duplicates()

    # Define a colormap for distinguishing the different weight combinations
    colors = plt.cm.jet(np.linspace(0, 1, len(weight_combinations)))

    # Loop over each unique weight combination and plot the total error
    for idx, (b_weight, gamma_weight) in enumerate(weight_combinations.values):
        # Filter the dataframe for this particular combination of weights
        df_filtered = df[(df['b_weight'] == b_weight) & (df['gamma_weight'] == gamma_weight)]

        # Extract b, gamma, and total_error values for plotting
        X = df_filtered['b'].values
        Y = df_filtered['gamma'].values
        Z = df_filtered['total_error'].values

        # Plot the data with a specific color for this weight combination
        ax.scatter(X, Y, Z, color=colors[idx], label=f'b_weight={b_weight}, gamma_weight={gamma_weight}')

    # Set axis labels and title
    ax.set_xlabel('b')
    ax.set_ylabel('gamma')
    ax.set_zlabel('Total Error')
    ax.set_title('Total Error vs b and gamma for all weight combinations')

    # Add a legend to distinguish different weight combinations
    ax.legend()

    # Show the plot
    plt.show()

def plots(vol_cp, vol, returns, state_prob_t):
    # Plot each column of vol_cp in a separate subplot
    plt.figure(figsize=(15, 10))

    # Number of columns
    num_columns = vol_cp.shape[1]

    for i in range(num_columns):
        plt.subplot(num_columns, 1, i + 1)  # Create a subplot for each column
        plt.plot(range(1, N + 1), vol_cp[1:, i], label=f'Column {i + 1}')
        plt.xlabel('Time (1 to N)')
        plt.ylabel('Values')
        plt.title(f'Evolution of Column {i + 1} of vol_cp Over Time')
        plt.grid(True)
        plt.legend(loc='upper right')

    plt.tight_layout()  # Adjust subplots to fit into the figure area neatly
    plt.show()

    # Display the actual volatility values from row 1 to N
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, N + 1), vol, label='Actual Volatility (Rows 1 to N)', color='blue')
    plt.xlabel('Time (1 to N)')
    plt.ylabel('Actual Volatility')
    plt.title('Actual Volatility Over Time (Rows 1 to N)')
    plt.grid(True)
    plt.legend()
    plt.show()

    # Display the centered returns values from row 1 to N
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, N + 1), returns, label='Generated Returns (Rows 1 to N)', color='blue')
    plt.xlabel('Time (1 to N)')
    plt.ylabel('Generated Returns')
    plt.title('Generated Returns Over Time (Rows 1 to N)')
    plt.grid(True)
    plt.legend()
    plt.show()

    # Generate the subplots
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))

    # 1. Area plot for state probabilities (state_prob_t)
    axes[0].stackplot(range(len(state_prob_t)), state_prob_t.T)
    axes[0].set_title('State Probabilities (state_prob_t)')
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('Probability')

    # 2. Plot for generated combination at each step (generated_combination)
    axes[1].plot(generated_combination, color='blue')
    axes[1].set_title('Generated Combination at Each Step')
    axes[1].set_xlabel('Time')
    axes[1].set_ylabel('Combination Index')

    # 3. Plot for returns
    axes[2].plot(returns, color='green')
    axes[2].set_title('Returns')
    axes[2].set_xlabel('Time')
    axes[2].set_ylabel('Returns')

    # Adjust layout and show plot
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Parameters
    sigma = 0.05
    k = 3
    m_0 = 0.5
    gamma = 0.5
    b = 2
    N = 500

    # Generate data
    generate = GenerateData(sigma, k, m_0, gamma, b, N)
    vol_cp = generate.vol_cp
    vol = generate.vol
    returns = generate.returns

    prob = ProbEstimation(k, m_0, sigma, b, gamma, returns)
    transi_mat = prob.transi_mat
    transi_prob = prob.transi_probs
    state_prob_t, _ = prob.calc_state_prob()

    generated_combination = generate.determine_combination_at_each_step()
    L = prob.calc_likelihood()
    print(L)

    plots(vol_cp, vol, returns, state_prob_t)
