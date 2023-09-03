import numpy as np


class ParticularSwarmOptimization:
    def __init__(self, group_size: int, w_velocity: float, w_pbest: float, w_gbest: float, max_iter: int):
        """

        :param group_size:
        :param w_velocity: [0, 1] is recommended
        :param w_pbest: [0, 1] is recommended
        :param w_gbest: [0, 1] is recommended
        :param max_iter:
        """
        self.group_size = group_size
        self.w_velocity = w_velocity
        self.w_pbest = w_pbest
        self.w_gbest = w_gbest
        self.max_iter = max_iter
        self.n_iter = 0
        self.fitness_calculation = None
        self.position_initialization = None
        self.velocity_initialization = None
        self.position_validation = None

        self.curr_group_position = None
        self.best_group_position = None
        self.group_velocity = None
        self.curr_group_fitness = None
        self.best_group_fitness = None

    def setup(
            self,
            fitness_calculation,
            position_initialization,
            velocity_initialization,
            position_validation=None
    ):
        self.fitness_calculation = fitness_calculation
        self.position_initialization = position_initialization
        self.velocity_initialization = velocity_initialization
        self.position_validation = position_validation if position_validation is not None else lambda x: x
        return self

    def _validate_curr_group_position(self):
        self.curr_group_position = self.position_validation(self.curr_group_position)

    def _init_group_status(self):
        self.curr_group_position = np.vstack([
            self.position_initialization() for _ in range(self.group_size)])
        self._validate_curr_group_position()

        self.group_velocity = np.vstack([
            self.velocity_initialization() for _ in range(self.group_size)])
        v_quantile = np.quantile(np.abs(self.group_velocity), .9)
        self.group_velocity = np.clip(self.group_velocity, -v_quantile, v_quantile)

        self.best_group_position = self.curr_group_position
        self.best_group_fitness = -np.ones(shape=(self.group_size, 1)) * np.inf

    def _update_curr_group_velocity(self):
        pbest_random = np.random.random(size=self.group_velocity.shape)
        gbest_random = np.random.random(size=self.group_velocity.shape)

        v = self.w_velocity * (self.max_iter - self.n_iter) / self.max_iter * self.group_velocity
        p = self.w_pbest * pbest_random * (self.best_group_position - self.curr_group_position)
        g = self.w_gbest * gbest_random * (self.best_position - self.curr_group_position)

        self.group_velocity = v + p + g

    def _update_curr_group_position(self):
        self.curr_group_position = self.curr_group_position + self.group_velocity
        self._validate_curr_group_position()

    def _update_best_group_position(self):
        self.best_group_position = np.where(
            self.curr_group_fitness > self.best_group_fitness,
            self.curr_group_position, self.best_group_position)

    def _update_curr_group_fitness(self):
        self.curr_group_fitness = np.array([self.fitness_calculation(i) for i in self.curr_group_position])[..., np.newaxis]

    def _update_best_group_fitness(self):
        self.best_group_fitness = np.where(
            self.curr_group_fitness > self.best_group_fitness,
            self.curr_group_fitness, self.best_group_fitness)

    @property
    def best_position(self):
        return self.best_group_position[np.argmax(self.best_group_fitness)][np.newaxis, ...]

    @property
    def best_fitness(self):
        return self.best_group_fitness.max()

    def optimize(self):

        self._init_group_status()

        while self.n_iter < self.max_iter:
            self._update_curr_group_position()
            self._update_curr_group_fitness()
            self._update_best_group_position()
            self._update_best_group_fitness()
            self._update_curr_group_velocity()
            if self.n_iter % 100 == 0:
                print(self.n_iter, self.w_velocity * (self.max_iter - self.n_iter) / self.max_iter, self.best_fitness)
            self.n_iter += 1

        return self


if __name__ == "__main__":
    N_SAMPLE = 100
    N_FEATURE = 10
    data = np.random.normal(size=(N_SAMPLE, N_FEATURE))
    w = np.random.normal(size=(N_FEATURE))
    b = np.random.normal(size=1)
    target = data @ w + b

    def rmse(weights):
        w = weights[:N_FEATURE]
        b = weights[N_FEATURE:]
        predict = data @ w + b
        return -np.mean((target - predict) ** 2)

    pso = ParticularSwarmOptimization(100, 1, 1, 1, 100)
    pso.setup(
        fitness_calculation=rmse,
        position_initialization=lambda: np.random.random((1, 11)),
        velocity_initialization=lambda: np.random.random((1, 11))
    )
    pso.optimize()
    print(pso.best_position, pso.best_fitness)
    print(w, b)
