import random
import math
import numpy as np
from draw_game import Draw
import pygame


class Snake:
    def __init__(self, start_pos, direction, length=3):
        self.body = []
        self.direction = direction
        self.score = 0
        self.steps = 0
        self.energy = 0
        self.dead = False
        x, y = start_pos
        dx, dy = direction
        for i in range(length):
            self.body.append(((x - i * dx, y - i * dy), direction))

    def move(self):
        self.body.pop()
        x, y = self.body[0][0]
        dx, dy = self.direction
        self.body.insert(0, ((x + dx, y + dy), self.direction))

    def grow(self):
        x, y = self.body[-1][0]
        dx, dy = self.body[-1][1]
        self.body.append(((x - dx, y - dy), (dx, dy)))


class Apple:
    def __init__(self, board_size, snake):
        self.x, self.y = self.spawn_new_apple(board_size, snake)

    def spawn_new_apple(self, board_size, snake):
        while True:
            x = random.randint(0, board_size[0] - 1)
            y = random.randint(0, board_size[1] - 1)
            if not collision_with_snake(snake.body, (x, y)):
                return x, y


def collision_with_snake(snake_body, position):
    for pos, _ in snake_body:
        if pos == position:
            return True
    return False


def direction_to_right(direction):
    # Map the direction to a right turn
    right_turns = {
        (1, 0): (0, -1),
        (0, -1): (-1, 0),
        (-1, 0): (0, 1),
        (0, 1): (1, 0)
    }
    return right_turns[direction]


def direction_to_left(direction):
    # Map the direction to a left turn
    left_turns = {
        (1, 0): (0, 1),
        (0, -1): (1, 0),
        (-1, 0): (0, -1),
        (0, 1): (-1, 0)
    }
    return left_turns[direction]


def next_direction(direction):
    # Map the direction to a right turn
    right_turns = {
        (0, 1): (1, 1),
        (1, 1): (1, 0),
        (1, 0): (1, -1),
        (1, -1): (0, -1),
        (0, -1): (-1, -1),
        (-1, -1): (-1, 0),
        (-1, 0): (-1, 1),
        (-1, 1): (0, 1)
    }
    return right_turns[direction]


class Game:
    def __init__(self, board_size=(10, 10), num_apples=2):
        self.board_size = board_size
        self.num_apples = num_apples
        self.snake = Snake((board_size[0] // 2, board_size[1] // 2), (1, 0))
        self.snake.energy = board_size[0] * board_size[1]
        self.apples = [Apple(board_size, self.snake) for _ in range(num_apples)]

    def reset(self):
        self.snake = Snake((self.board_size[0] // 2, self.board_size[1] // 2), (1, 0))
        self.snake.energy = self.board_size[0] * self.board_size[1]
        self.apples = [Apple(self.board_size, self.snake) for _ in range(self.num_apples)]

    def snake_out_of_bounds(self):
        if self.snake.body[0][0][0] < 0 or self.snake.body[0][0][0] >= self.board_size[0] or \
                self.snake.body[0][0][1] < 0 or self.snake.body[0][0][1] >= self.board_size[1]:
            return True
        return False

    def closer_to_apple(self, apple):
        if math.sqrt((apple.x - self.snake.body[0][0][0]) ** 2 + (apple.y - self.snake.body[0][0][1]) ** 2) < \
                math.sqrt((apple.x - self.snake.body[1][0][0]) ** 2 + (apple.y - self.snake.body[1][0][1]) ** 2):
            return True
        return False

    def look_in_direction(self, direction):
        dx, dy = direction
        x, y = self.snake.body[0][0][0] + dx, self.snake.body[0][0][1] + dy
        len = 1
        is_wall = 0
        is_apple = 0
        is_body = 0

        while 0 <= x < self.board_size[0] and 0 <= y < self.board_size[1]:
            if collision_with_snake(self.snake.body, (x, y)):
                is_body = 1.0 / len
                return is_wall, is_apple, is_body
            for apple in self.apples:
                if (x, y) == (apple.x, apple.y):
                    is_apple = 1.0
            x += dx
            y += dy
            len += 1
        is_wall = 1.0 / len
        return is_wall, is_apple, is_body

    def get_state(self):
        w, a, b = [], [], []
        direction = self.snake.direction
        for _ in range(8):
            dis_to_wall, is_apple, is_body = self.look_in_direction(direction)
            w.append(dis_to_wall)
            a.append(is_apple)
            b.append(is_body)
            direction = next_direction(direction)
        h = [0, 0, 0, 0]
        if self.snake.direction == (-1, 0):
            h[0] = 1
        elif self.snake.direction == (1, 0):
            h[1] = 1
        elif self.snake.direction == (0, -1):
            h[2] = 1
        elif self.snake.direction == (0, 1):
            h[3] = 1

        t = [0, 0, 0, 0]
        if self.snake.body[-1][1] == (-1, 0):
            t[0] = 1
        elif self.snake.body[-1][1] == (1, 0):
            t[1] = 1
        elif self.snake.body[-1][1] == (0, -1):
            t[2] = 1
        elif self.snake.body[-1][1] == (0, 1):
            t[3] = 1

        vision = list(np.concatenate((w, a, b, h, t)))
        return np.array(vision, dtype=int)

    def step(self, action):

        reward = 0
        game_over = False

        if action == 0:
            self.snake.direction = direction_to_right(self.snake.direction)
        elif action == 2:
            self.snake.direction = direction_to_left(self.snake.direction)

        self.snake.move()
        self.snake.steps += 1
        self.snake.energy -= 1

        if collision_with_snake(self.snake.body[1:],
                                self.snake.body[0][0]) or self.snake_out_of_bounds() or self.snake.energy <= 0:
            game_over = True
            reward = -200
            return reward, game_over, self.snake.score

        if len(self.snake.body) == self.board_size[0] * self.board_size[1] - 1:
            game_over = True
            reward = 1000
            return reward, game_over, self.snake.score

        for apple in self.apples:
            if self.closer_to_apple(apple):
                reward += 5

            if self.snake.body[0][0] == (apple.x, apple.y):
                self.snake.grow()
                self.snake.score += 1
                reward += 100 - self.snake.energy * 0.5
                self.snake.energy = self.board_size[0] * self.board_size[1]
                apple.x, apple.y = apple.spawn_new_apple(self.board_size, self.snake)

        return reward, game_over, self.snake.score
