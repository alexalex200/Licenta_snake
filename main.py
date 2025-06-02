import pygame

from agent_genetic import Agent_genetic
from game import Game
from draw_game import Draw
from agent import Agent
import torch

from model import Linear_QNet

pygame.init()

game = Game(board_size=(6, 6), num_snakes=1, num_apples=1)
agents = []
for i in range(game.num_snakes):
    agents.append(Agent(i, model='model/model.pth'))
for agent in agents:
    agent.n_games = 100

draw = Draw(game)
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    draw.draw()
    for i, agent in enumerate(agents):
        if game.snakes[i].dead:
            continue
        state = agent.get_vision(game)
        state = torch.tensor(state, dtype=torch.float).to(agent.device)
        prediction = agent.model(state)
        dist = torch.distributions.Categorical(prediction)
        action = dist.sample().item()
        move = [0, 0, 0]
        move[action] = 1
        game.change_direction(game.snakes[i], move)

    _,game_overs,_ = game.update()

    if game_overs[0] == True and all(x == game_overs[0] for x in game_overs):
        game.reset()

pygame.quit()