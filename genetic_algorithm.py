import numpy as np
from typing import List


class Population:
    def __init__(self, population: List[object], mutation_rate: float = 0.05):
        self.population = population
        self._num_alive = len(self.population)
        self.mutation_rate = mutation_rate
        self.generation = 1
        self.max_fitness = 0
        self.max_fitness_history = []

    @property
    def best_member(self):
        best_member = None
        best_fitness = 0

        for member in self.population:
            if member.fitness > best_fitness:
                best_member = member
                best_fitness = member.fitness

        return best_member

    @property
    def num_alive(self):
        count = 0
        for member in self.population:
            count += member.alive

        self._num_alive = count
        return self._num_alive

    def evaluate(self):
        population_fitness = np.array(
            [member.fitness for member in self.population])

        self.max_fitness = population_fitness.max()
        self.max_fitness_history.append(self.max_fitness)

        for member in self.population:
            parentA = None
            parentB = None

            while parentA is None:
                potential_parent = self.population[np.random.randint(
                    len(self.population))]
                if potential_parent != member:
                    parentA = self.select_parent(potential_parent)

            while parentB is None:
                potential_parent = self.population[np.random.randint(
                    len(self.population))]
                if potential_parent != parentA and potential_parent != member:
                    parentB = self.select_parent(potential_parent)

            member.crossover(parentA, parentB, self.mutation_rate)

        for member in self.population:
            member.apply()
            member.reset()

        self.generation += 1

    def select_parent(self, parent):
        if (np.random.uniform(0, 1) < parent.fitness / self.max_fitness):
            return parent
        return None
