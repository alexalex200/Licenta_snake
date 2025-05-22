ETA = 100
import random
import numpy as np
import copy


def simulated_binary_crossover(parent1, parent2):
    rand = np.random.rand(*parent1.shape)
    gamma = np.empty(parent1.shape)
    gamma[rand <= 0.5] = (2 * rand[rand <= 0.5]) ** (1.0 / (ETA + 1))
    gamma[rand > 0.5] = (1.0 / (2.0 * (1.0 - rand[rand > 0.5]))) ** (1.0 / (ETA + 1))

    child1 = 0.5 * ((1 + gamma) * parent1 + (1 - gamma) * parent2)
    child2 = 0.5 * ((1 - gamma) * parent1 + (1 + gamma) * parent2)

    return child1, child2


def single_point_binary_crossover(parent1, parent2):
    # Save original shapes to reshape later
    original_shape = parent1.shape

    # Normalize to 2D if 1D (e.g., bias vectors)
    if len(parent1.shape) == 1:
        parent1 = parent1.reshape(1, -1)
        parent2 = parent2.reshape(1, -1)

    # Copy parents
    child1 = parent1.copy()
    child2 = parent2.copy()

    # Get shape
    rows, cols = parent1.shape

    # Random crossover point
    row = np.random.randint(0, rows)
    col = np.random.randint(0, cols)

    # Perform crossover
    child1[:row, :] = parent2[:row, :]
    child2[:row, :] = parent1[:row, :]
    child1[row, :col + 1] = parent2[row, :col + 1]
    child2[row, :col + 1] = parent1[row, :col + 1]

    # Reshape back to original shape
    if len(original_shape) == 1:
        child1 = child1.flatten()
        child2 = child2.flatten()

    return child1, child2


def crossover(parent1_weights, parent1_biases, parent2_weights, parent2_biases):
    if np.random.rand() < 0.5:
        child1_weights, child2_weights = single_point_binary_crossover(parent1_weights, parent2_weights)
        child1_biases, child2_biases = single_point_binary_crossover(parent1_biases, parent2_biases)
    else:
        child1_weights, child2_weights = simulated_binary_crossover(parent1_weights, parent2_weights)
        child1_biases, child2_biases = simulated_binary_crossover(parent1_biases, parent2_biases)

    return child1_weights, child1_biases, child2_weights, child2_biases


def gaussian_mutation(child, mutation_rate, scale=0.2):
    mutation_array = np.random.random(child.shape) < mutation_rate
    gaussian = np.random.normal(size=child.shape) * scale
    child[mutation_array] += gaussian[mutation_array]


def random_uniform_mutation(child, mutation_rate, low=-1, high=1):
    mutation_array = np.random.random(child.shape) < mutation_rate
    random_values = np.random.uniform(low, high, child.shape)
    child[mutation_array] = random_values[mutation_array]


def mutate(child_weights, child_biases, mutation_rate):
    if np.random.rand() < 1:
        gaussian_mutation(child_weights, mutation_rate)
        gaussian_mutation(child_biases, mutation_rate)
    else:
        random_uniform_mutation(child_weights, mutation_rate)
        random_uniform_mutation(child_biases, mutation_rate)

    return child_weights, child_biases