import torch
import numpy as np
from model import Generic_QNet
from game import Game
import copy
from crossover_mutation import crossover, mutate
from draw_game import Draw


class Agent_genetic:
    def __init__(self, index_snake, model=None):
        self.index_snake = index_snake
        self.mutation_rate = 0.1
        self.fitness = 0
        self.score = 0
        self.steps = 0

        self.model = Generic_QNet(32, [20, 12], 3)
        if model is not None:
            self.model.load_state_dict(torch.load(model))
            self.model.eval()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def get_vision(self, game):
        snake = game.snakes[self.index_snake]
        w, a, b = [], [], []
        direction = snake.direction
        for _ in range(8):
            dis_to_wall, is_apple, is_body = look_in_direction(snake, direction, game)
            w.append(dis_to_wall)
            a.append(is_apple)
            b.append(is_body)
            direction = next_direction(direction)

        h = [0, 0, 0, 0]
        if snake.direction == (-1, 0):
            h[0] = 1
        elif snake.direction == (1, 0):
            h[1] = 1
        elif snake.direction == (0, -1):
            h[2] = 1
        elif snake.direction == (0, 1):
            h[3] = 1

        t = [0, 0, 0, 0]
        if len(snake.body) > 1:
            if snake.body[-1][1] == (-1, 0):
                t[0] = 1
            elif snake.body[-1][1] == (1, 0):
                t[1] = 1
            elif snake.body[-1][1] == (0, -1):
                t[2] = 1
            elif snake.body[-1][1] == (0, 1):
                t[3] = 1

        vision = list(np.concatenate((w, a, b, h, t)))
        return np.array(vision, dtype=int)

    def calculate_fitness(self):
        self.fitness = self.steps + ((2 ** self.score) + (self.score ** 2.1) * 500) - (
                ((0.25 * self.steps) ** 1.3) * (self.score ** 1.2))


def look_in_direction(snake, direction, game, distance=False):
    occupied_positions = game.get_occupied_positions()
    dx, dy = direction
    x, y = snake.body[0][0][0] + dx, snake.body[0][0][1] + dy
    dis_to_wall = 1
    is_apple = 0
    is_body = 0

    while 0 <= x < game.board_size[0] and 0 <= y < game.board_size[1]:
        if (x, y) in occupied_positions and is_body == 0:
            is_body = dis_to_wall
        for apple in game.apples:
            if (x, y) == (apple.x, apple.y) and is_apple == 0:
                is_apple = dis_to_wall
        dis_to_wall += 1
        x += dx
        y += dy

    if is_body != 0:
        if distance:
            is_body = 1.0 / is_body
        else:
            is_body = 1
    if is_apple != 0:
        if distance:
            is_apple = 1.0 / is_apple
        else:
            is_apple = 1
    return 1.0/dis_to_wall, is_apple, is_body


def next_direction(direction):
    # Map the direction to a right turn
    right_turns = {
        (0, 1): (1, 1),
        (1, 1): (1, 0),
        (1, 0): (1, -1),
        (1, -1): (0, -1),
        (0, -1): (-1, -1),
        (-1, -1): (-1, 0),
        (-1, 0): (-1, 1),
        (-1, 1): (0, 1)
    }
    return right_turns[direction]


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


def train():
    num_generations = 1000
    num_parents = 500
    num_agents = 1000
    num_of_layers = 3

    game = Game((10, 10), num_snakes=1, num_apples=1)
    agents = [Agent_genetic(0) for _ in range(num_agents)]
    best_score = -np.inf
    # draw = Draw(game)

    for generation in range(num_generations):
        for agent in agents:
            agent.steps = 0
            agent.score = 0
            game.reset()
            while True:
                # draw.draw()
                state = agent.get_vision(game)
                state = torch.tensor(state, dtype=torch.float).to(agent.device)
                move = [0, 0, 0]
                move[torch.argmax(agent.model(state)).item()] = 1
                game.change_direction(game.snakes[0], move)
                _, game_overs, scores = game.update()
                agent.steps += 1
                agent.score = scores[0]
                if game_overs[0] == True and all(x == game_overs[0] for x in game_overs):
                    break
            agent.calculate_fitness()

        agents.sort(key=lambda x: x.fitness, reverse=True)
        best_agents = agents[:num_parents]
        print(f'Generation {generation}, Best fitness: {best_agents[0].fitness}, Score: {best_agents[0].score}')
        if best_agents[0].fitness > best_score:
            best_score = best_agents[0].fitness
            best_agents[0].model.save('GA_model.pth')
        for i in range(num_parents):
            agents[i] = best_agents[i]

        for i in range(num_parents, num_agents, 2):
            parent1, parent2 = roulette_selection(best_agents, 2)

            parent1_weights, parent1_biases = parent1.model.get_weights_biases()
            parent2_weights, parent2_biases = parent2.model.get_weights_biases()

            child1_weights, child1_biases, child2_weights, child2_biases = [], [], [], []
            for j in range(num_of_layers):
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

            agents[i].model.set_weights_biases(child1_weights, child1_biases)
            agents[i + 1].model.set_weights_biases(child2_weights, child2_biases)


if __name__ == "__main__":
    train()
