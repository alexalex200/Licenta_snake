import torch
import random
import numpy as np
import math
from collections import deque
from game import Game, direction_to_right, direction_to_left
from model import Linear_QNet, QTrainer
from draw_game import Draw
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')  # Use a non-GUI backend for matplotlib
from IPython import display

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.0005

plt.ion()

def plot(scores, mean_scores):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores)
    plt.plot(mean_scores)
    plt.ylim(ymin=0)
    plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
    plt.show(block=False)
    plt.pause(.1)

class Agent:
    def __init__(self, index_snake, model=None):
        self.index_snake = index_snake
        self.n_games = 0
        self.epsilon = 0
        self.gamma = 0.9
        self.memory = deque(maxlen=MAX_MEMORY)

        self.model = Linear_QNet(11, 256, 3)
        if model is not None:
            self.model.load_state_dict(torch.load(model))
            self.model.eval()

        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def get_state(self, game):
        snake = game.snakes[self.index_snake]
        state = []

        state.append(game.is_collision(snake, snake.direction))
        state.append(game.is_collision(snake, direction_to_right(snake.direction)))
        state.append(game.is_collision(snake, direction_to_left(snake.direction)))

        state.append(snake.direction == (0, -1))
        state.append(snake.direction == (0, 1))
        state.append(snake.direction == (-1, 0))
        state.append(snake.direction == (1, 0))

        closest_apple = None
        closest_distance = float('inf')
        for apple in game.apples:
            if math.sqrt(
                    (apple.x - snake.body[0][0][0]) ** 2 + (apple.y - snake.body[0][0][1]) ** 2) < closest_distance:
                closest_distance = math.sqrt(
                    (apple.x - snake.body[0][0][0]) ** 2 + (apple.y - snake.body[0][0][1]) ** 2)
                closest_apple = apple

        state.append(closest_apple.x < snake.body[0][0][0])
        state.append(closest_apple.x > snake.body[0][0][0])
        state.append(closest_apple.y < snake.body[0][0][1])
        state.append(closest_apple.y > snake.body[0][0][1])

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        for state, action, reward, next_state, done in mini_sample:
            self.train_short_memory(state, action, reward, next_state, done)
        # states = np.array([item[0] for item in mini_sample])
        # actions = np.array([item[1] for item in mini_sample])
        # rewards = np.array([item[2] for item in mini_sample])
        # next_states = np.array([item[3] for item in mini_sample])
        # dones = np.array([item[4] for item in mini_sample])
        #
        # self.train_short_memory(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        self.epsilon = 100 - self.n_games
        final_move = [0, 0, 0]
        if random.randint(0, 100) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float).to(self.device)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
        return final_move


def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent(0)
    game = Game(board_size=(10, 10), num_snakes=1, num_apples=1)

    while True:
        state_old = agent.get_state(game)
        final_move = agent.get_action(state_old)

        game.change_direction(game.snakes[0], final_move)

        rewards, dones, scores = game.update()
        reward, done, score = rewards[0], dones[0], scores[0]

        state_new = agent.get_state(game)
        agent.train_short_memory(state_old, final_move, reward, state_new, done)
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()
            if score > record:
                record = score
                # Save the model
                agent.model.save()
            print(f'Game {agent.n_games}, Score: {score}, Record: {record}')
            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            # plot(plot_scores, plot_mean_scores)


if __name__ == "__main__":
    train()

