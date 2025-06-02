import torch
import numpy as np

from model import Linear_Network
from game import Game
import copy
from crossover_mutation import crossover, mutate
from draw_game import Draw


class Individual:
    def __init__(self, n_actions, input_dims, mutation_rate=0.1):
        self.mutation_rate = mutation_rate
        self.fitness = 0
        self.score = 0

        self.model = Linear_Network(input_dims, [20, 12], n_actions, genetic=True)

    def calculate_fitness(self, steps, score):
        self.fitness = steps + ((2 ** score) + (score ** 2.1) * 500) - (
                ((0.25 * steps) ** 1.3) * (score ** 1.2))
        self.score = score

    def choose_action(self, observation):
        state = torch.tensor(observation, dtype=torch.float).to(self.model.device)
        dist = self.model(state)
        action = torch.argmax(dist)
        return action

def roulette_selection(agents, num_parents):
    selection = []
    fitness_sum = sum(agent.fitness for agent in agents)
    for _ in range(num_parents):
        pick = np.random.uniform(0, fitness_sum)
        current = 0
        for agent in agents:
            current += agent.fitness
            if current > pick:
                selection.append(agent)
                break
    return selection


if __name__ == '__main__':
    num_generations = 1000
    num_parents = 20
    num_individuals = 100

    game = Game((10, 10), num_apples=1)
    individuals = [Individual(3, 32) for _ in range(num_individuals)]
    # for individual in individuals:
    #     individual.model.load("ga_best_model.pth")
    best_fitness = 0

    for generation in range(num_generations):
        for individual in individuals:
            game.reset()
            while True:
                state = game.get_state()
                state = torch.tensor(state, dtype=torch.float).to(individual.model.device)
                dist = individual.model(state)
                action = torch.argmax(dist)

                _, done, _ = game.step(action)
                if done:
                    break
            individual.calculate_fitness(game.snake.steps, game.snake.score)

        individuals.sort(key=lambda x: x.fitness, reverse=True)
        best_individuals = individuals[:num_parents]
        print(
            f'Generation {generation}, Best fitness: {best_individuals[0].fitness}, Score: {best_individuals[0].score}')
        if best_individuals[0].fitness >= best_fitness:
            best_fitness = best_individuals[0].fitness
            best_individuals[0].model.save('ga_best_model.pth')
        for i in range(num_parents):
            individuals[i] = best_individuals[i]

        for i in range(num_parents, num_individuals, 2):
            parent1, parent2 = roulette_selection(best_individuals, 2)

            parent1_weights, parent1_biases = parent1.model.get_weights_biases()
            parent2_weights, parent2_biases = parent2.model.get_weights_biases()

            child1_weights, child1_biases, child2_weights, child2_biases = [], [], [], []
            for j in range(len(parent1_weights)):
                child1_weight, child1_bias, child2_weight, child2_bias = crossover(parent1_weights[j],
                                                                                   parent1_biases[j],
                                                                                   parent2_weights[j],
                                                                                   parent2_biases[j])

                child1_weight, child1_bias = mutate(child1_weight, child1_bias, parent1.mutation_rate)
                child2_weight, child2_bias = mutate(child2_weight, child2_bias, parent2.mutation_rate)
                child1_weights.append(child1_weight)
                child1_biases.append(child1_bias)
                child2_weights.append(child2_weight)
                child2_biases.append(child2_bias)

            individuals[i].model.set_weights_biases(child1_weights, child1_biases)
            individuals[i + 1].model.set_weights_biases(child2_weights, child2_biases)
