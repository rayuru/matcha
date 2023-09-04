import random
import numpy as np
from copy import deepcopy


class Fish:
    def __init__(self, scorer, init_p, init_d, visual, retry):
        self.scorer = scorer
        self.init_p = init_p
        self.init_d = init_d
        self.visual = visual
        self.retry = retry

        self.position = self.init_p()
        self.speed = None
        self.fitness = self.scorer(self.position)

    def __sub__(self, other):
        return np.sqrt(np.square(self.position - other.position).mean())

    def get_visible_swarm(self, swarm):
        distances = [self - fish for fish in swarm]
        visible_swarm = []
        for d, fish in zip(distances, swarm):
            if 0 < d < self.visual:
                visible_swarm.append(fish)
        return visible_swarm

    def _move(self, target_position=None, direction=None):
        if target_position is not None:
            if np.abs(target_position - self.position).sum() == 0:
                return self.position
            direction = target_position - self.position
        direction /= np.linalg.norm(direction, 2)
        position = self.position + direction * self.speed

        return position

    def _prey(self, visible_swarm):
        if not visible_swarm:
            return
        for _ in range(self.retry):
            candidate = random.choice(visible_swarm)
            if candidate.fitness > self.fitness:
                return self._move(candidate.position)

    def _swarm(self, visible_swarm):
        if not visible_swarm:
            return
        if np.mean([fish.fitness for fish in visible_swarm]) < self.fitness:
            return
        visible_center = np.mean([fish.position for fish in visible_swarm], axis=0)
        return self._move(visible_center)

    def _follow(self, visible_swarm, swarm):
        if not visible_swarm:
            return
        best_fish = max(visible_swarm, key=lambda x: x.fitness)
        best_fish_visible_swarm = best_fish.get_visible_swarm(swarm)
        if not best_fish_visible_swarm:
            return
        visible_fitness = np.mean([fish.fitness for fish in best_fish_visible_swarm])
        if visible_fitness < self.fitness:
            return
        visible_center = np.mean([fish.position for fish in best_fish_visible_swarm], axis=0)
        return self._move(visible_center)

    def _random_move(self):
        return self._move(direction=self.init_d())

    def step(self, swarm, speed):
        self.speed = speed
        visible_swarm = self.get_visible_swarm(swarm)
        next_position = max([
            self._prey(visible_swarm),
            self._swarm(visible_swarm),
            self._follow(visible_swarm, swarm)
        ], key=lambda x: self.scorer(x) if x is not None else -np.inf)
        if next_position is None:
            self.position = self._random_move()
        else:
            self.position = next_position
        self.fitness = self.scorer(self.position)


class ArtificialFishSwarm:
    def __init__(self, visual, retry, swarm_size, max_iter):
        self.visual = visual
        self.swarm_size = swarm_size
        self.retry = retry
        self.max_iter = max_iter

        self.fitness_calculation = None
        self.position_initialization = None
        self.direction_initialization = None
        self.speed_initialization = None

        self.best_fish = None

    def setup(
            self,
            fitness_calculation,
            position_initialization,
            direction_initialization,
            speed_initialization
    ):
        self.fitness_calculation = fitness_calculation
        self.position_initialization = position_initialization
        self.direction_initialization = direction_initialization
        self.speed_initialization = speed_initialization

    def optimize(self):
        swarm = self._initialize_swarm()

        n_iter = 0
        while n_iter < self.max_iter:
            new_swarm = deepcopy(swarm)

            for fish in new_swarm:
                fish.step(swarm, self.speed_initialization() * (1 - n_iter / self.max_iter))
                if self.best_fish is None or self.best_fish.fitness < fish.fitness:
                    self.best_fish = deepcopy(fish)
            if n_iter % 10 == 0: print(n_iter, max(fish.fitness for fish in new_swarm))
            swarm = new_swarm
            n_iter += 1
        return self

    def _initialize_swarm(self):
        swarm = []
        for _ in range(self.swarm_size):
            fish = Fish(
                scorer=self.fitness_calculation,
                init_p=self.position_initialization,
                init_d=self.direction_initialization,
                visual=self.visual,
                retry=self.retry
            )
            swarm.append(fish)
        return swarm


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

    afs = ArtificialFishSwarm(1, 10, 100, 200)
    afs.setup(
        fitness_calculation=rmse,
        position_initialization=lambda: np.random.normal(size=11),
        direction_initialization=lambda: np.random.normal(size=11),
        speed_initialization=lambda: 1
    )
    afs.optimize()
    print(afs.best_fish.position, afs.best_fish.fitness)
    print(w, b)
