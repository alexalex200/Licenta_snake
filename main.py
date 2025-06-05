import pygame
from game import Game
from draw_game import Draw
from agent import Agent
from agent_genetic import Individual
import torch

if __name__ == "__main__":

    game = Game(board_size=(6, 6), num_apples=1)

    agent = Agent(3, 32)
    agent.actor.load("ppo_agent_actor.pth")
    agent.actor.categorical = False
    agent.critic.load("ppo_agent_critic.pth")

    individual = Individual(3, 32)
    individual.model.load("ga_best_model.pth")
    individual.model.categorical = False

    draw = Draw(game, agent.actor)

    running = True
    while running:
        game.reset()
        while True:
            draw.draw()
            state = game.get_state()
            state = torch.tensor(state, dtype=torch.float).to(agent.actor.device)
            action = agent.actor(state)
            action = torch.argmax(action, -1)
            _, done,score  = game.step(action)
            if done:
                break
