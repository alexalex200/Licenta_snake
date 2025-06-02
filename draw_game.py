import pygame

import game
from images import *

BACKGROUND_COLOR = (34, 34, 34)
GRID_COLOR = (60, 61, 55)


class Draw:
    def __init__(self, game, model):
        pygame.init()

        self.game = game
        self.model = model
        self.width = 1280
        self.height = 720
        self.cell_size = 600 // self.game.board_size[0]
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.fps = pygame.time.Clock()
        pygame.display.set_caption("Snake")

        self.apple, self.grass, self.mud, self.head, self.body, self.tail, self.bent = load_images(self.cell_size)

    def draw_grid(self, start_x=0, start_y=0):
        for i in range(0, self.game.board_size[0] + 1):
            pygame.draw.line(self.screen, GRID_COLOR, (i*self.cell_size + start_x, 0 + start_y), (i*self.cell_size + start_x, self.game.board_size[1] * self.cell_size + start_y))
        for i in range(0, self.game.board_size[1] + 1):
            pygame.draw.line(self.screen, GRID_COLOR, (0 + start_x, i * self.cell_size + start_y), (self.game.board_size[0] * self.cell_size + start_x, i * self.cell_size + start_y))

    def draw_snake(self, start_x=0, start_y=0):

        tail = False
        snake = self.game.snake
        head_img = get_head_direction(self.head, snake.direction[0], snake.direction[1])
        self.screen.blit(pygame.image.frombuffer(head_img.tobytes(), head_img.shape[1::-1], "RGBA"),
                         (start_x + snake.body[0][0][0] * self.cell_size,start_y + snake.body[0][0][1] * self.cell_size))

        for i, body_part in enumerate(snake.body):
            if i != 0 and i != len(snake.body) - 1:
                if body_part[1] != snake.body[i - 1][1]:
                    body_img = get_bent_direction(self.bent, body_part[1][0], body_part[1][1],
                                                  snake.body[i - 1][1][0], snake.body[i - 1][1][1])
                    if i == len(snake.body) - 2:
                        tail = True
                else:
                    body_img = get_body_direction(self.body, body_part[1][0], body_part[1][1])

                self.screen.blit(pygame.image.frombuffer(body_img.tobytes(), body_img.shape[1::-1], "RGBA"),
                                 (start_x + body_part[0][0] * self.cell_size,start_y + body_part[0][1] * self.cell_size))

            if tail:
                tail_image = get_tail_direction(self.tail, snake.body[-2][1][0], snake.body[-2][1][1])
                tail = False
            else:
                tail_image = get_tail_direction(self.tail, snake.body[-2][1][0], snake.body[-2][1][1])
            self.screen.blit(pygame.image.frombuffer(tail_image.tobytes(), tail_image.shape[1::-1], "RGBA"),
                             (start_x + snake.body[-1][0][0] * self.cell_size,start_y + snake.body[-1][0][1] * self.cell_size))

    def draw_apple(self, start_x=0, start_y=0):
        for apple in self.game.apples:
            self.screen.blit(pygame.image.frombuffer(self.apple.tobytes(), self.apple.shape[1::-1], "RGBA"),
                             (start_x + apple.x * self.cell_size,start_y + apple.y * self.cell_size))

    # def draw_background(self):
    #     for i in range(self.width // cell_size):
    #         for j in range(self.height // cell_size):
    #             self.screen.blit(pygame.image.frombuffer(self.grass.tobytes(), self.grass.shape[1::-1], "RGBA"),
    #                              (i * cell_size, j * cell_size))

    def draw_network(self):
        layer_gap = 680// (len(self.model.dimensions) + 1)
        node_gap = self.height // (max(self.model.dimensions) + 2)
        for i in range(1, len(self.model.dimensions)):
            for j in range(self.model.dimensions[i - 1]):
                for k in range(self.model.dimensions[i]):
                    pygame.draw.line(self.screen, (255, 255, 255),
                                     (i * layer_gap, j * node_gap + (self.height - node_gap * self.model.dimensions[i - 1]) // 2),
                                     ((i + 1) * layer_gap, k * node_gap + (self.height - node_gap * self.model.dimensions[i]) // 2), 1)
        for i in range(1,len(self.model.dimensions) + 1):
            advantage = (self.height - node_gap * self.model.dimensions[i - 1])//2
            for j in range(self.model.dimensions[i - 1]):
                pygame.draw.circle(self.screen, (255,255,255), (i * layer_gap, j * node_gap + advantage), node_gap // 3 + 1, 0)
                pygame.draw.circle(self.screen, (255, 0, 0), (i * layer_gap, j * node_gap + advantage ), node_gap // 3, 0)


    def draw(self):

        pygame.event.get()
        # self.draw_background()
        # self.draw_puddle()
        self.screen.fill(BACKGROUND_COLOR)
        self.draw_grid(680, 0)
        self.draw_snake(680, 0)
        self.draw_apple(680, 0)
        self.draw_network()
        pygame.display.update()
        self.fps.tick(10)

    def quit(self):
        pygame.quit()

