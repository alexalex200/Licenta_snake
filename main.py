import pygame
from game import Game
from draw_game import Draw
from agent import Agent
from agent_genetic import Individual
from plots import Plot
import keyboard
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


    draw = Draw(game, agent.actor, individual.model)

    running = True
    while running:
        game.reset()
        while True:
            draw.draw()
            state = game.get_state()
            state = torch.tensor(state, dtype=torch.float).to(individual.model.device)
            if draw.model_switch:
                action = agent.actor(state)
            else:
                action = individual.model(state)
            action = torch.argmax(action, -1)
            _, done,score = game.step(action)
            if done:
                if score == game.board_size[0] * game.board_size[1] - 3:
                    draw.win_flicker()
                break
    #plot.save_data()

