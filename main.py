import pygame
from game import Game
from draw_game import Draw
from agent import Agent
import torch

pygame.init()

game = Game(board_size=(10, 10), num_snakes=1)
agent = Agent(0)
model = torch.load('model/model_10x10_39.pth')

draw = Draw(game)

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    state = agent.get_state(game)
    move = model(state)

    game.change_direction(game.snakes[0], move)
    game.update()

    # Draw the game
    draw.draw()
pygame.quit()