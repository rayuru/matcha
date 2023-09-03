import numpy as np


class EvolutionStrategy:
    def __init__(self, group_size, sigma, max_iter, learning_rate):
        self.group_size = group_size
        self.sigma = sigma
        self.max_iter = max_iter
        self.learning_rate = learning_rate

        self.fitness_calculation = None
        self.initialization = None

        self.generation = None
        self.fitness = None

    def setup(self, fitness_calculation, initialization):
        self.fitness_calculation = fitness_calculation
        self.initialization = initialization

    def _calculate_fitness(self, generation):
        return np.array([self.fitness_calculation(i) for i in generation])

    def _sample_search_direction(self):
        search_direction = np.random.normal(size=(self.group_size, self.generation.shape[1]))
        return search_direction

    def _estimate_gradient(self, search_direction):
        f1 = self._calculate_fitness(self.generation + self.sigma * search_direction)
        f2 = self._calculate_fitness(self.generation - self.sigma * search_direction)
        gradient = np.einsum("n,nd->d", f1 - f2, search_direction) / 2 / self.sigma / self.group_size
        return gradient[np.newaxis, ...]

    def _update_generation(self, gradient):
        new_generation = self.generation - self.learning_rate * gradient
        new_fitness = self._calculate_fitness(new_generation)
        if new_fitness < self.fitness:
            self.generation = new_generation
            self.fitness = new_fitness

    def optimize(self):
        self.generation = self.initialization()
        self.fitness = self._calculate_fitness(self.generation)
        n_iter = 0
        while n_iter < self.max_iter:
            search_direction = self._sample_search_direction()
            gradient = self._estimate_gradient(search_direction)
            self._update_generation(gradient)
            n_iter += 1
        return self


class GaussianProcessEvolutionStrategy:
    def __init__(self, mu, la, max_iter):
        self.max_iter = max_iter
        self.mu = mu
        self.la = la

        self.fitness_calculation = None
        self.mean_initialization = None
        self.variance_initialization = None

        self.mean = None
        self.variance = None

    def setup(self, fitness_calculation, mean_initialization, variance_initialization):
        self.fitness_calculation = fitness_calculation
        self.mean_initialization = mean_initialization
        self.variance_initialization = variance_initialization

    def optimize(self):
        mean = self.mean_initialization()
        variance = self.variance_initialization()
        n_iter = 0

        while n_iter < self.max_iter:
            group = np.random.multivariate_normal(mean, np.diag(variance), size=self.mu)

            fitness = np.array([self.fitness_calculation(i) for i in group])
            elite = group[np.argsort(fitness)[: self.la]]
            mean = elite.mean(axis=0)
            variance = np.var(elite, axis=0)
            n_iter += 1
        self.mean = mean
        self.variance = variance
        return self


class CovarianceMatrixAdaptionEvolutionStrategy:
    ...


if __name__ == "__main__":

    data = np.random.normal(size=(100, 10))
    weight = np.random.normal(size=10)
    bias = np.random.normal()
    target = data @ weight + bias

    def rmse(params):
        w = params[:10]
        b = params[10:]
        pred = data @ w + b
        res = ((target.T - pred) ** 2).mean()
        return res

    es = GaussianProcessEvolutionStrategy(1000, 30, max_iter=100)
    es.setup(
        fitness_calculation=rmse,
        mean_initialization=lambda: np.random.random(11),
        variance_initialization=lambda: np.random.random(11)
    )
    es.optimize()
    print(es.mean)
    print(weight, bias)

    print(es.variance)
