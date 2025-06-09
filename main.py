import pygame
from game import Game
from draw_game import Draw
from agent import Agent
from agent_genetic import Individual
from plots import Plot
import torch

if __name__ == "__main__":

    game = Game(board_size=(6, 6), num_apples=1)

    plot = Plot()

    agent = Agent(3, 32)
    agent.load_models("ppo_agent")
    agent.actor.categorical = False

    individual = Individual(3, 32)
    individual.model.load("ga_best_model_copy.pth")
    individual.model.categorical = False

    #draw = Draw(game, individual.model)

    n = 0
    nn = 0
    running = True
    while running:
        game.reset()
        while True:
            #draw.draw()
            state = game.get_state()
            state = torch.tensor(state, dtype=torch.float).to(individual.model.device)
            action = agent.actor(state)
            action = torch.argmax(action, -1)
            _, done,score = game.step(action)
            if done:
                if score == 33:
                    nn += 1
                    plot.add_ppo(game.snake.score, game.snake.steps)
                break
        n += 1
        if nn == 1000:
            print(n)
            break
    plot.save_data()

