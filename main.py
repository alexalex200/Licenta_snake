import pygame
from game import Game
from draw_game import Draw
from agent import Agent
from agent_genetic import Individual
import torch

if __name__ == "__main__":

    game = Game(board_size=(6, 6), num_apples=1)
    draw = Draw(game)

    agent = Agent(3, 32)
    agent.actor.load("ppo_agent_actor.pth")
    agent.critic.load("ppo_agent_critic.pth")

    individual = Individual(3, 32)
    individual.model.load("ga_best_model.pth")

    running = True
    while running:
        game.reset()
        while True:
            draw.draw()
            state = game.get_state()
            action,_,_ = agent.choose_action(state)
            _, done, _ = game.step(action)
            if done:
                break
