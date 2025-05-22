import pygame

from agent_genetic import Agent_genetic
from game import Game
from draw_game import Draw
from agent import Agent
import torch

from model import Linear_QNet

pygame.init()

game = Game(board_size=(10, 10), num_snakes=1, num_apples=1)
agents = []
for i in range(game.num_snakes):
    agents.append(Agent_genetic(i, model='model/GA_model_best.pth'))
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
        #move = agent.get_action(state)
        move = [0, 0, 0]
        move[torch.argmax(agent.model(state)).item()] = 1
        game.change_direction(game.snakes[i], move)

    _,game_overs,_ = game.update()

    if game_overs[0] == True and all(x == game_overs[0] for x in game_overs):
        game.reset()

pygame.quit()