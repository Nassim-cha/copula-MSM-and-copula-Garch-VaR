import numpy as np
import matplotlib.pyplot as plt


class DataGeneration:
    def __init__(self, theta, mu, sigma, n_steps):
        self.theta = theta      # Mean reversion speed
        self.mu = mu            # Long-term mean
        self.sigma = sigma      # Volatility
        self.x0 = mu            # Initial value
        self.n_steps = n_steps  # Number of steps
        self.Y = np.zeros(n_steps)  # Unbounded OU process
        self.vol = np.zeros(n_steps)
        self.X = np.zeros(n_steps)  # Bounded OU process [-1, 1]
        self.first_series = np.zeros(n_steps)  # First normal process
        self.second_series = np.zeros(n_steps)  # Second correlated normal process

    def generate_process_volatility_returns(self):
        """ Generate the Ornstein-Uhlenbeck process and apply a tanh transformation to bound it between -1 and 1 """
        self.X[0] = self.x0
        for t in range(1, self.n_steps):
            # OU process update: dY_t = theta * (mu - Y_t) dt + sigma * sqrt(dt) * N(0,1)
            self.X[t] = self.theta * (self.X[t-1] - self.mu) + self.mu + self.sigma * np.random.normal()

        # Apply exp transformation to generate centered returns
        self.vol = np.exp(self.X)

        normal_samples = np.random.normal(0, 1, self.n_steps)

        self.Y = self.vol * normal_samples

        return self.X, self.vol, self.Y

    def plot_processes(self):
        """ Plot the bounded OU process and the two correlated normal processes """
        fig, ax = plt.subplots(3, 1, figsize=(10, 12))

        # Plot the bounded OU process X[t]
        ax[0].plot(self.X, label='OU process', color='green')
        ax[0].set_title('OU process generating log-volatility')
        ax[0].set_xlabel('Time Steps')
        ax[0].set_ylabel('X[t]')
        ax[0].legend()

        # Plot the first normal process
        ax[1].plot(self.vol, label='Stochastic Volatility', color='blue')
        ax[1].set_title('Stochastic Volatility Process')
        ax[1].set_xlabel('Time Steps')
        ax[1].set_ylabel('Stochastic Volatility')
        ax[1].legend()

        # Plot the second correlated normal process
        ax[2].plot(self.Y, label='Centered Returns', color='red')
        ax[2].set_title('Centered Returns')
        ax[2].set_xlabel('Time Steps')
        ax[2].set_ylabel('Centered Returns')
        ax[2].legend()

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    theta = 0.95
    mu = 0
    sigma = 0.2
    n_steps = 500
    generate = DataGeneration(theta, mu, sigma, n_steps)
    generate.generate_process_volatility_returns()
    generate.plot_processes()
