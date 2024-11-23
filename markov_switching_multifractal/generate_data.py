import numpy as np
import random
import itertools

class GenerateData:
    def __init__(self, sigma, k, m_0, gamma, b, N):
        self.sigma = sigma
        self.k = k
        self.m_0 = m_0
        self.gamma = gamma
        self.b = b
        self.N = N
        self.gamma_vect = self.calculate_gamma_vect()
        self.vol_cp = self.generate_vol_cp()
        self.vol = self.generate_vol()
        self.returns, self.eps = self.genrate_centered_returns()

    def calculate_gamma_vect(self):
        # calculer le vecteur des probabilités de transition pour chaque composantes
        gamma_vect = 1 - (1 - self.gamma) ** (self.b ** np.arange(self.k))
        return gamma_vect

    def initialize_vol_cp(self):
        # initialisation des composantes de vol à t = 0
        # pas utilisés dans la générations des rendements
        return np.random.choice([self.m_0, 2 - self.m_0], self.k)

    def vol_cp_iteration(self, vol_cp_lag):
        # Simulate draws for components transition probabilities and determine components values
        return [m_k if random.random() < 1-gamma_k/2 else 2 - m_k for m_k, gamma_k in zip(vol_cp_lag, self.gamma_vect)]

    def generate_vol_cp(self):
        # Initialize the array with zeros
        vol_cp = np.zeros((self.N+1, self.k))

        # Fill the first row with random choices of m_0 or 2 - m_0
        vol_cp[0] = self.initialize_vol_cp()

        for i in range(1, self.N+1):
            cp_iter = self.vol_cp_iteration(vol_cp[i - 1])
            vol_cp[i] = cp_iter

        return vol_cp

    def generate_vol(self):
        return np.sqrt(self.sigma**2 * np.prod(self.vol_cp[1:], axis=1))

    def genrate_centered_returns(self):
        # Generate random draws from a normal distribution
        random_draws = np.random.normal(0, 1, self.N)  # N-1 because we're considering rows 1 to N

        # Calculate actual returns as the product of actual volatility, random draws, and adding the drift
        return self.vol * random_draws, random_draws

    def calc_vect_combinations(self):
        combinations = list(itertools.product([self.m_0, 2-self.m_0], repeat=self.k))

        # Create a dictionary with keys from 1 to len(combinations)
        vectors_dict = {i + 1: vec for i, vec in enumerate(combinations)}

        return vectors_dict

    def determine_combination_at_each_step(self):
        # Get all possible combinations of vol_cp values
        combinations = self.calc_vect_combinations()

        # Initialize an array to store the combination index at each step
        combination_steps = np.zeros(self.N, dtype=int)  # Array of zeros with size N+1 (for steps 0 to N)

        # Iterate through each step of vol_cp
        for t in range(0, self.N):
            # Get the current vol_cp vector at step t
            current_cp = tuple(self.vol_cp[t])

            # Find which combination it corresponds to
            for key, vec in combinations.items():
                if current_cp == vec:  # Check if current_cp matches any combination
                    combination_steps[t] = key  # Store the combination index in the array
                    break  # Exit the loop once the match is found

        return combination_steps
