import pygame
import torch
from images import *

BACKGROUND_COLOR = (34, 34, 34)
GRID_COLOR = (60, 61, 55)
GREEN =(32, 194, 14)
RED = (128, 0, 0)
YELLOW = (179, 179, 0)
WHITE = (211, 211, 211)

# BACKGROUND_COLOR = (221, 221, 221)
# GRID_COLOR = (60, 61, 55)
# GREEN = (60, 61, 55)
# RED = (255, 0, 0)
# YELLOW = (0, 255, 0)
# WHITE = (34, 34, 34)


def cap_in_bounds(value, min_value, max_value):
    return max(min_value, min(value, max_value))


class Draw:
    def __init__(self, game, rl_model, ga_model):
        pygame.init()
        self.font = pygame.font.Font(None, 30)

        self.game = game
        self.ga_model = ga_model
        self.rl_model = rl_model
        self.model = rl_model
        self.model_switch = True
        self.width = 1600
        self.height = 900
        self.cell_size = 600 // self.game.board_size[0]
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.fps = pygame.time.Clock()
        pygame.display.set_caption("Snake")

        self.apple, self.grass, self.mud, self.head, self.body, self.tail, self.bent = load_images(self.cell_size)

        self.vision = False

    def draw_grid(self, start_x=0, start_y=0):
        for i in range(0, self.game.board_size[0] + 1):
            pygame.draw.line(self.screen, GRID_COLOR, (i*self.cell_size + start_x, 0 + start_y), (i*self.cell_size + start_x, self.game.board_size[1] * self.cell_size + start_y), 2)
        for i in range(0, self.game.board_size[1] + 1):
            pygame.draw.line(self.screen, GRID_COLOR, (0 + start_x, i * self.cell_size + start_y), (self.game.board_size[0] * self.cell_size + start_x, i * self.cell_size + start_y), 2)

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

    def draw_vision(self, start_x=0, start_y=0):
        self.game.get_state()
        head_x, head_y = self.game.snake.body[0][0]
        head_x = start_x + head_x * self.cell_size + self.cell_size // 2
        head_y = start_y + head_y * self.cell_size + self.cell_size // 2
        for apple_x, apple_y, x, y, is_wall in self.game.vision_for_draw:
            x = start_x + x * self.cell_size + self.cell_size // 2
            y = start_y + y * self.cell_size + self.cell_size // 2

            if is_wall:
                pygame.draw.line(self.screen, WHITE,
                                 (head_x, head_y),
                                 (x, y), 2)
                if x - start_x == 0 or x - start_x == self.game.board_size[0] * self.cell_size:
                    pygame.draw.line(self.screen, WHITE, (x, y - self.cell_size // 4),
                                     (x, y + self.cell_size // 4), 2)
                if y - start_y == 0 or y - start_y == self.game.board_size[1] * self.cell_size:
                    pygame.draw.line(self.screen, WHITE, (x - self.cell_size // 4, y),
                                     (x + self.cell_size // 4, y), 2)

                pygame.draw.line(self.screen, BACKGROUND_COLOR, (start_x - 1, start_y),
                                 (start_x - self.cell_size//4, start_y), 2)

                pygame.draw.line(self.screen, BACKGROUND_COLOR, (start_x - 1, start_y + self.cell_size * self.game.board_size[1]),
                                 (start_x - self.cell_size // 4, start_y + self.cell_size * self.game.board_size[1]), 2)

                pygame.draw.line(self.screen, BACKGROUND_COLOR,
                                 (start_x , start_y + self.cell_size * self.game.board_size[1] + 1),
                                 (start_x, start_y + self.cell_size * self.game.board_size[1] + self.cell_size // 4), 2)

                pygame.draw.line(self.screen, BACKGROUND_COLOR,
                                 (start_x + self.cell_size * self.game.board_size[0], start_y + self.cell_size * self.game.board_size[1] + 1),
                                 (start_x + self.cell_size * self.game.board_size[0], start_y + self.cell_size * self.game.board_size[1] + self.cell_size // 4), 2)

            else:
                pygame.draw.line(self.screen, YELLOW,(head_x, head_y), (x, y), 2)
                pygame.draw.polygon(self.screen, YELLOW, [(x - self.cell_size // 2, y - self.cell_size // 2),
                                                               (x + self.cell_size // 2, y - self.cell_size // 2),
                                                               (x + self.cell_size // 2, y + self.cell_size // 2),
                                                               (x - self.cell_size // 2, y + self.cell_size // 2)], 2)
                pygame.draw.polygon(self.screen, BACKGROUND_COLOR,
                                    [(x - self.cell_size // 2 + 2, y - self.cell_size // 2 + 2),
                                     (x + self.cell_size // 2 - 1, y - self.cell_size // 2 + 2),
                                     (x + self.cell_size // 2 - 1, y + self.cell_size // 2 - 1),
                                     (x - self.cell_size // 2 + 2, y + self.cell_size // 2 - 1)])

            if (apple_x, apple_y) != (-1, -1):
                x = start_x + apple_x * self.cell_size + self.cell_size // 2
                y = start_y + apple_y * self.cell_size + self.cell_size // 2
                pygame.draw.line(self.screen, RED,
                                 (head_x, head_y),
                                 (x,y), 2)
                pygame.draw.polygon(self.screen, RED,[(x - self.cell_size // 2, y - self.cell_size // 2),
                                                               (x + self.cell_size // 2, y - self.cell_size // 2),
                                                               (x + self.cell_size // 2, y + self.cell_size // 2),
                                                               (x - self.cell_size // 2, y + self.cell_size // 2)], 2)
                pygame.draw.polygon(self.screen, BACKGROUND_COLOR,
                                    [(x - self.cell_size // 2 + 2, y - self.cell_size // 2 + 2),
                                     (x + self.cell_size // 2 - 1, y - self.cell_size // 2 + 2),
                                     (x + self.cell_size // 2 - 1, y + self.cell_size // 2 - 1),
                                     (x - self.cell_size // 2 + 2, y + self.cell_size // 2 - 1)])

    def draw_network(self, input_layer, layers, weights):

        layer_gap = (self.width - 600) // (len(self.model.dimensions) + 1)
        node_gap = self.height // (max(self.model.dimensions) + 2)

        layers = [input_layer] + layers
        for i in range(len(layers[-1])):
            if layers[-1][i] != max(layers[-1]):
                layers[-1][i] = 0

        texts = ["wall", "apple", "snake", "head left", "head right", "head up", "head down", "tail left", "tail right", "tail up", "tail down", "left turn", "straight", "right turn", "score", "steps", "direction", "length"]

        for i in range(1, len(self.model.dimensions)):
            for j in range(self.model.dimensions[i - 1]):
                for k in range(self.model.dimensions[i]):
                    if layers[i - 1][j] > 0 and layers[i][k] > 0:
                        pygame.draw.line(self.screen, GREEN,
                                     (i * layer_gap + node_gap//3, j * node_gap + (self.height - node_gap * self.model.dimensions[i - 1]) // 2),
                                     ((i + 1) * layer_gap - node_gap//3 , k * node_gap + (self.height - node_gap * self.model.dimensions[i]) // 2), 1)

        for i in range(1,len(self.model.dimensions) + 1):
            advantage = (self.height - node_gap * self.model.dimensions[i - 1])//2
            for j in range(self.model.dimensions[i - 1]):
                pygame.draw.circle(self.screen, GREEN, (i * layer_gap, j * node_gap + advantage), node_gap // 3 + 2, 0)
                if layers[i - 1][j] > 0:
                    if i == 1:
                        self.screen.blit(self.font.render(texts[j//8 + max(0,j-24)], False, GREEN),
                                         ((i-0.7) * layer_gap, j * node_gap + advantage - 10))
                    pygame.draw.circle(self.screen, GREEN, (i * layer_gap, j * node_gap + advantage), node_gap // 3, 0)
                    if i == len(self.model.dimensions):
                        self.screen.blit(self.font.render(texts[j + 11], False, GREEN),
                                         ((i+0.2) * layer_gap, j * node_gap + advantage - 10))
                else:
                    if i == 1:
                        self.screen.blit(self.font.render(texts[j//8 + max(0,j-24)], False, GRID_COLOR),
                                         ((i-0.7) * layer_gap, j * node_gap + advantage - 10))
                    if i == len(self.model.dimensions):
                        self.screen.blit(self.font.render(texts[j + 11], False, GRID_COLOR),
                                         ((i+0.2) * layer_gap, j * node_gap + advantage - 10))
                    pygame.draw.circle(self.screen, BACKGROUND_COLOR, (i * layer_gap, j * node_gap + advantage ), node_gap // 3, 0)

    def vision_toggle(self):

        position_x = self.width - 599
        position_y = 610
        pygame.draw.rect(self.screen, GRID_COLOR, (position_x - 2, position_y - 2, 27, 27))
        pygame.draw.rect(self.screen, GREEN if self.vision else BACKGROUND_COLOR, (position_x , position_y , 23, 23))
        font = pygame.font.Font(None, 25)
        self.screen.blit(font.render("Vision ON/OFF", False, GREEN if self.vision else GRID_COLOR), (position_x + 29, position_y + 4))

    def model_toggle(self):

        position_x = self.width - 325
        position_y = 610

        font = pygame.font.Font(None, 25)
        pygame.draw.rect(self.screen, GRID_COLOR, (position_x - 2, position_y - 2, 150, 27))
        pygame.draw.rect(self.screen, GREEN if self.model_switch else BACKGROUND_COLOR, (position_x, position_y, 146, 23))
        self.screen.blit(font.render("RL", False, BACKGROUND_COLOR if self.model_switch else GRID_COLOR), (position_x + 65, position_y + 4))

        pygame.draw.rect(self.screen, GRID_COLOR, (position_x + 150, position_y - 2, 150, 27))
        pygame.draw.rect(self.screen, GREEN if not self.model_switch else BACKGROUND_COLOR, (position_x + 152, position_y, 146, 23))
        self.screen.blit(font.render("GA", False, BACKGROUND_COLOR if not self.model_switch else GRID_COLOR), (position_x + 215, position_y + 4))

    def draw_info(self):

        position_x = self.width - 510
        position_y = 700

        font = pygame.font.Font(None, 40)
        self.screen.blit(font.render("Score: " + str(self.game.snake.score) + " | Steps: " + str(self.game.snake.steps) + " | Energy: " + str(self.game.snake.energy), False, (255,255,255)), (position_x, position_y))


    def win_flicker(self):
        position_x = self.width - 510
        position_y = 700
        font = pygame.font.Font(None, 40)
        pygame.draw.rect(self.screen, BACKGROUND_COLOR, (position_x - 2, position_y - 2, 600, 50))

        for i in range(10):
            self.screen.blit(font.render("Score: " + str(self.game.snake.score) + " | Steps: " + str(
                self.game.snake.steps) + " | Energy: " + str(self.game.snake.energy), False, (255, 255, 255)),
                             (position_x, position_y))
            pygame.display.update()
            pygame.time.wait(250)
            self.screen.blit(font.render("Score: " + str(self.game.snake.score) + " | Steps: " + str(
                self.game.snake.steps) + " | Energy: " + str(self.game.snake.energy), False, GRID_COLOR),
                             (position_x, position_y))
            pygame.display.update()
            pygame.time.wait(250)

    def draw(self):

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.quit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                if pygame.Rect(self.width - 600, 610, 23, 23).collidepoint(event.pos):
                    self.vision = not self.vision
                if pygame.Rect(self.width - 325, 610, 146, 23).collidepoint(event.pos) and not self.model_switch:
                    self.model = self.rl_model
                    self.model_switch = True
                    self.game.reset()
                elif pygame.Rect(self.width - 175, 610, 146, 23).collidepoint(event.pos) and self.model_switch:
                    self.model = self.ga_model
                    self.model_switch = False
                    self.game.reset()

        self.screen.fill(BACKGROUND_COLOR)
        self.draw_grid((self.width - 601), 1)
        if self.vision:
            self.draw_vision((self.width - 601), 1)
        self.draw_snake((self.width - 601), 1)
        self.draw_apple((self.width - 601), 1)
        weights, _ = self.model.get_weights_biases()
        state = self.game.get_state()
        state_tensor = torch.tensor(state, dtype=torch.float).to(self.model.device)
        self.draw_network(state, self.model.get_values_of_layers(state_tensor), weights)

        self.vision_toggle()
        self.model_toggle()
        self.draw_info()

        pygame.display.update()
        self.fps.tick(10)

    def quit(self):
        pygame.quit()

