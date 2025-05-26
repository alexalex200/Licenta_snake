import pygame
from images import *

BACKGROUND_COLOR = (34, 34, 34)
GRID_COLOR = (60, 61, 55)
cell_size = 40


class Draw:
    def __init__(self, game):
        pygame.init()

        self.game = game
        self.width = game.board_size[0] * cell_size
        self.height = game.board_size[1] * cell_size
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.fps = pygame.time.Clock()
        pygame.display.set_caption("Snake")

        self.apple, self.grass, self.mud, self.head, self.body, self.tail, self.bent = load_images(cell_size)

    def draw_grid(self):
        for i in range(0, self.width, cell_size):
            pygame.draw.line(self.screen, GRID_COLOR, (i, 0), (i, self.height))
        for i in range(0, self.height, cell_size):
            pygame.draw.line(self.screen, GRID_COLOR, (0, i), (self.width, i))

    def draw_snake(self):

        tail = False
        for snake in self.game.snakes:

            if len(snake.body) == 0:
                continue

            head_img = get_head_direction(self.head, snake.direction[0], snake.direction[1])
            if len(snake.colors) != 0:
                head_img = change_colors(head_img, snake.colors[0], snake.colors[1], snake.colors[2])

            self.screen.blit(pygame.image.frombuffer(head_img.tobytes(), head_img.shape[1::-1], "RGBA"),
                             (snake.body[0][0][0] * cell_size, snake.body[0][0][1] * cell_size))

            for i, body_part in enumerate(snake.body):
                if i != 0 and i != len(snake.body) - 1:
                    if body_part[1] != snake.body[i - 1][1]:
                        body_img = get_bent_direction(self.bent, body_part[1][0], body_part[1][1],
                                                      snake.body[i - 1][1][0], snake.body[i - 1][1][1])
                        if i == len(snake.body) - 2:
                            tail = True
                    else:
                        body_img = get_body_direction(self.body, body_part[1][0], body_part[1][1])

                    if len(snake.colors) != 0:
                        body_img = change_colors(body_img, snake.colors[0], snake.colors[1], snake.colors[2])
                    self.screen.blit(pygame.image.frombuffer(body_img.tobytes(), body_img.shape[1::-1], "RGBA"),
                                     (body_part[0][0] * cell_size, body_part[0][1] * cell_size))

            if tail:
                tail_image = get_tail_direction(self.tail, snake.body[-2][1][0], snake.body[-2][1][1])
                tail = False
            else:
                tail_image = get_tail_direction(self.tail, snake.body[-2][1][0], snake.body[-2][1][1])
            if len(snake.colors) != 0:
                    tail_image = change_colors(tail_image, snake.colors[0], snake.colors[1], snake.colors[2])
            self.screen.blit(pygame.image.frombuffer(tail_image.tobytes(), tail_image.shape[1::-1], "RGBA"),
                             (snake.body[-1][0][0] * cell_size, snake.body[-1][0][1] * cell_size))

    def draw_apple(self):
        for apple in self.game.apples:
            self.screen.blit(pygame.image.frombuffer(self.apple.tobytes(), self.apple.shape[1::-1], "RGBA"),
                             (apple.x * cell_size, apple.y * cell_size))

    def draw_puddle(self):
        for puddle in self.game.puddles:
            for m in puddle.mud:
                self.screen.blit(pygame.image.frombuffer(self.mud.tobytes(), self.mud.shape[1::-1], "RGBA"),
                                 (m[0] * cell_size, m[1] * cell_size))

    def draw_background(self):
        for i in range(self.width // cell_size):
            for j in range(self.height // cell_size):
                self.screen.blit(pygame.image.frombuffer(self.grass.tobytes(), self.grass.shape[1::-1], "RGBA"),(i*cell_size, j*cell_size))

    def draw(self):

        # self.draw_background()
        # self.draw_puddle()
        self.screen.fill(BACKGROUND_COLOR)
        self.draw_grid()
        self.draw_snake()
        self.draw_apple()
        pygame.display.update()
        self.fps.tick(20)

    def quit(self):
        pygame.quit()