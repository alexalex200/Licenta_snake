import pygame
from dataclasses import dataclass
from images import *
import random

BACKGROUND_COLOR = (0, 0, 0)
GRID_COLOR = (255, 255, 255)
cell_size = 32

class Draw:
    def __init__(self,game):
        self.game = game
        self.width = game.board_size[0] * cell_size
        self.height = game.board_size[1] * cell_size
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.fps = pygame.time.Clock()
        pygame.display.set_caption("Snake")

        self.apple, self.head, self.body, self.tail, self.bent = load_images()


    def draw_grid(self):
        for i in range(0, self.width, cell_size):
            pygame.draw.line(self.screen, GRID_COLOR, (i, 0), (i, self.height))
        for i in range(0, self.height, cell_size):
            pygame.draw.line(self.screen, GRID_COLOR, (0, i), (self.width, i))


    def draw_snake(self):

        tail = False
        for snake in self.game.snakes:

            head_img = get_head_direction(self.head, snake.direction[0], snake.direction[1])
            self.screen.blit(pygame.image.frombuffer(head_img.tobytes(), head_img.shape[1::-1], "BGR"), (snake.body[0][0][0]*cell_size, snake.body[0][0][1]*cell_size))

            for i,body_part in enumerate(snake.body):
                if i != 0 and i != len(snake.body)-1:
                    if body_part[1] != snake.body[i-1][1]:
                        body_img = get_bent_direction(self.bent, body_part[1][0], body_part[1][1], snake.body[i-1][1][0], snake.body[i-1][1][1])
                        if i == len(snake.body)-2:
                            tail = True
                    else:
                        body_img = get_body_direction(self.body, body_part[1][0], body_part[1][1])
                    self.screen.blit(pygame.image.frombuffer(body_img.tobytes(), body_img.shape[1::-1], "BGR"), (body_part[0][0]*cell_size, body_part[0][1]*cell_size))

            if tail:
                tail_image = get_tail_direction(self.tail, snake.body[-2][1][0], snake.body[-2][1][1])
                tain = False
            else:
                tail_image = get_tail_direction(self.tail, snake.body[-2][1][0], snake.body[-2][1][1])
            self.screen.blit(pygame.image.frombuffer(tail_image.tobytes(), tail_image.shape[1::-1], "BGR"), (snake.body[-1][0][0]*cell_size, snake.body[-1][0][1]*cell_size))


    def draw_apple(self):
        for apple in self.game.apples:
            self.screen.blit(pygame.image.frombuffer(self.apple.tobytes(), self.apple.shape[1::-1], "BGR"), (apple.x*cell_size, apple.y*cell_size))


    def draw(self):
        self.screen.fill(BACKGROUND_COLOR)
        self.draw_snake()
        self.draw_apple()
        pygame.display.update()
        self.fps.tick(10)


# pygame.init()
#
# SCREEN_WIDTH = 800
# SCREEN_HEIGHT = 800
# BLOCK_SIZE = 32
#
# screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
# fps = pygame.time.Clock()
#
# @dataclass
# class Body_part:
#     x: int
#     y: int
#     dir_x: int
#     dir_y: int
#
# snake_head = Body_part(BLOCK_SIZE*6, BLOCK_SIZE*6, 0, 1)
# snake_body = [Body_part(BLOCK_SIZE*6, BLOCK_SIZE*5, 0, 1), Body_part(BLOCK_SIZE*6,BLOCK_SIZE*4, 0, 1), Body_part(BLOCK_SIZE*6,BLOCK_SIZE*3, 0, 1)]
# snake_tail = Body_part(BLOCK_SIZE*6,BLOCK_SIZE*2, 0, 1)
#
# snake_apple = Body_part(random.randint(1,24)*32,random.randint(1,24)*32, 0, 0)
#
# dir_x,dir_y = 0,1
#
# def move_snake():
#     global snake_head, snake_body, snake_tail, dir_x, dir_y
#     snake_tail.dir_x = snake_body[-1].dir_x
#     snake_tail.dir_y = snake_body[-1].dir_y
#     snake_tail.x = snake_body[-1].x
#     snake_tail.y = snake_body[-1].y
#
#     for i in range(len(snake_body)-1, 0, -1):
#         snake_body[i].dir_x = snake_body[i-1].dir_x
#         snake_body[i].dir_y = snake_body[i-1].dir_y
#         snake_body[i].x = snake_body[i-1].x
#         snake_body[i].y = snake_body[i-1].y
#
#     snake_body[0].dir_x = snake_head.dir_x
#     snake_body[0].dir_y = snake_head.dir_y
#     snake_body[0].x = snake_head.x
#     snake_body[0].y = snake_head.y
#
#     snake_head.x += dir_x*BLOCK_SIZE
#     snake_head.y += dir_y*BLOCK_SIZE
#     snake_head.dir_x = dir_x
#     snake_head.dir_y = dir_y
#
#
#
# apple, head, body, tail, bent = load_images()
#
# run = True
# while run:
#
#     screen.fill((0,0,0))
#     head_img = get_head_direction(head, snake_head.dir_x, snake_head.dir_y)
#     screen.blit(pygame.image.frombuffer(head_img.tobytes(), head_img.shape[1::-1], "BGR"), (snake_head.x, snake_head.y))
#
#     for i,body_part in enumerate(snake_body):
#         if i==0:
#             if body_part.dir_x != snake_head.dir_x or body_part.dir_y != snake_head.dir_y:
#                 body_img = get_bent_direction(bent, body_part.dir_y, body_part.dir_x, snake_head.dir_y, snake_head.dir_x)
#             else:
#                 body_img = get_body_direction(body, body_part.dir_x, body_part.dir_y)
#         else:
#             if body_part.dir_x != snake_body[i-1].dir_x or body_part.dir_y != snake_body[i-1].dir_y:
#                 body_img = get_bent_direction(bent, body_part.dir_y, body_part.dir_x, snake_body[i-1].dir_y, snake_body[i-1].dir_x)
#             else:
#                 body_img = get_body_direction(body, body_part.dir_x, body_part.dir_y)
#         screen.blit(pygame.image.frombuffer(body_img.tobytes(), body_img.shape[1::-1], "BGR"), (body_part.x, body_part.y))
#
#     tail_image = get_tail_direction(tail, snake_body[-1].dir_x, snake_body[-1].dir_y)
#     screen.blit(pygame.image.frombuffer(tail_image.tobytes(), tail_image.shape[1::-1], "BGR"), (snake_tail.x, snake_tail.y))
#
#     screen.blit(pygame.image.frombuffer(apple.tobytes(), apple.shape[1::-1], "BGR"), (snake_apple.x, snake_apple.y))
#
#
#     for event in pygame.event.get():
#         if event.type == pygame.KEYDOWN:
#             if event.key == pygame.K_UP:
#                 dir_x, dir_y = 0, -1
#             elif event.key == pygame.K_DOWN:
#                 dir_x, dir_y = 0, 1
#             elif event.key == pygame.K_LEFT:
#                 dir_x, dir_y = -1, 0
#             elif event.key == pygame.K_RIGHT:
#                 dir_x, dir_y = 1, 0
#         if event.type == pygame.QUIT:
#             run = False
#     move_snake()
#     if snake_head.x == snake_apple.x and snake_head.y == snake_apple.y:
#         print("Apple eaten")
#         snake_body.append(Body_part(snake_tail.x, snake_tail.y, snake_tail.dir_x, snake_tail.dir_y))
#         snake_apple = Body_part(random.randint(1,24)*32,random.randint(1,24)*32, 0, 0)
#
#     pygame.display.update()
#     fps.tick(10)
# pygame.quit()