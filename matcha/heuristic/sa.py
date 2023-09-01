import numpy as np


class SimulatedAnnealing:
    def __init__(self, temperature, max_iter):
        self.temperature = temperature
        self.max_iter = max_iter

        self.initialization = None
        self.fitness_calculation = None

        self.individual = None
        self.best_individual = None
        self.best_fitness = -np.inf

    def setup(self, fitness_calculation, initialization):
        self.fitness_calculation = fitness_calculation
        self.initialization = initialization

    def _add_disturbance(self):
        return self.individual + np.random.normal(size=self.individual.shape)

    def optimize(self):
        self.individual = self.initialization()

        individual_fitness = self.fitness_calculation(self.individual)
        n_iter = 0
        while n_iter < self.max_iter:
            candidate = self._add_disturbance()
            candidate_fitness = self.fitness_calculation(candidate)
            delta_fitness = candidate_fitness - individual_fitness

            probability = 1 / (1 + np.exp(delta_fitness / self.temperature))
            update_flag = np.random.choice([0, 1], p=[probability, 1 - probability])

            if delta_fitness < 0 and update_flag == 0:
                continue
            individual_fitness = candidate_fitness
            self.individual = candidate

            if individual_fitness > self.best_fitness:
                self.best_individual = self.individual
                self.best_fitness = individual_fitness

            n_iter += 1

        return self
