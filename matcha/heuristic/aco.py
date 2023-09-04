import numpy as np


class Ant:
    def __init__(self, graph):
        self.graph = graph
        self.spots = np.arange(graph.shape[0])
        self.pheromone = None
        self.mask = np.zeros(shape=(1, graph.shape[0]))
        self.current_spot = None
        self.path = []
        self.distance = 0
        self.move_count = 0

    def set_pheromone(self, pheromone):
        self.pheromone = pheromone

    def _initialize(self, start_spot):
        self.distance = 0
        self.move_count = 0
        # self.current_spot = np.random.choice(np.arange(self.graph.shape[0]))
        self.current_spot = start_spot
        self.path = [self.current_spot]
        self.mask = np.zeros(self.graph.shape[0])
        self.mask[self.current_spot] -= 1e4

    def _choose_next_spot(self):
        # use pheromone as logits for probability calculation
        # use mask to push down logits and probability
        # use safe softmax to avoid overflow
        logits = self.pheromone[self.current_spot, :] + self.mask
        safe_exp = np.exp(logits - logits.max())
        prob = safe_exp / safe_exp.sum()
        next_spot = np.random.choice(self.spots, p=prob.reshape(-1))
        return next_spot

    def _update_status(self, next_spot):
        self.mask[next_spot] -= 1e4
        self.distance += self.graph[self.current_spot, next_spot]
        self.path.append(next_spot)
        self.current_spot = next_spot
        self.move_count += 1

    def search_path(self, start_spot):
        self._initialize(start_spot)
        while self.move_count < self.graph.shape[0] - 1:
            next_spot = self._choose_next_spot()
            self._update_status(next_spot)

        return self.path


class AntColonyOptimization:
    def __init__(self, graph, group_size, max_iter, decay=.9):
        self.graph = graph
        self.pheromone = np.zeros_like(graph)
        self.ant_group = [Ant(self.graph) for i in range(group_size)]
        self.max_iter = max_iter
        self.decay = decay
        self.best_path = None
        self.best_distance = np.inf

    def optimize(self, start_spot=0):
        n_iter = 0
        while n_iter < self.max_iter:
            # initialize pheromone matrix for current iteration
            temp_pheromone = np.zeros_like(self.pheromone)
            for ant in self.ant_group:
                # set historical pheromone matrix for current ant
                ant.set_pheromone(self.pheromone)
                path = ant.search_path(start_spot)

                # update current pheromone matrix
                for i, j in zip(path, path[1:]):
                    temp_pheromone[i, j] += 1 / ant.distance

                # update best record
                if ant.distance < self.best_distance:
                    self.best_distance = ant.distance
                    self.best_path = ant.path
            n_iter += 1

            # decay historical pheromone and update with current pheromone
            self.pheromone = self.pheromone * self.decay + temp_pheromone
        return self
