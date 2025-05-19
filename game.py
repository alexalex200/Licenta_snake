import random
import math


class Snake:
    def __init__(self, start_pos, direction, length=3):
        self.body = []
        self.direction = direction
        self.score = 0
        self.frame_iteration = 0
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
    def __init__(self, board_size, occupied_positions):
        self.x, self.y = self.spawn_new_apple(board_size, occupied_positions)

    def spawn_new_apple(self, board_size, occupied_positions):
        while True:
            x = random.randint(0, board_size[0] - 1)
            y = random.randint(0, board_size[1] - 1)
            if (x, y) not in occupied_positions:
                return x, y


def dir_to_center(x, y, board_size):
    center_x = board_size[0] / 2
    center_y = board_size[1] / 2

    dx = 1 if x <= center_x else -1
    dy = 1 if y <= center_y else -1

    # Prefer moving along x-axis if possible
    return (dx, 0) if dx != 0 else (0, dy)


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


def direction_index_to_center(x, y, board_width, board_height):
    center_x = board_width // 2
    center_y = board_height // 2

    dx = 0
    dy = 0

    if x < center_x:
        dx = 1
    elif x > center_x:
        dx = -1

    if y < center_y:
        dy = 1
    elif y > center_y:
        dy = -1

    direction_map = {
        (-1, -1): 0,
        (0, -1): 1,
        (1, -1): 2,
        (1, 0): 3,
        (1, 1): 4,
        (0, 1): 5,
        (-1, 1): 6,
        (-1, 0): 7,
        (0, 0): -1  # Special case: already at center
    }

    return direction_map[(dx, dy)]


def in_radius(apple_x, apple_y, snake_x, snake_y, radius):
    if (apple_x - snake_x) ** 2 + (apple_y - snake_y) ** 2 <= radius ** 2:
        return 1  # - ((apple_x - snake_x) ** 2 + (apple_y - snake_y) ** 2) / radius ** 2
    else:
        return 0


class Game:
    def __init__(self, board_size=(20, 20), num_snakes=2, num_apples=1):
        self.board_size = board_size
        self.snakes = []
        self.num_snakes = num_snakes
        self.apples = []
        self.num_apples = num_apples

        self.reset()

    def reset(self):
        self.snakes = []
        self.apples = []
        nx, ny = math.ceil(math.sqrt(self.num_snakes)), math.ceil(math.sqrt(self.num_snakes))
        for i in range(nx):
            prev_x = i * (self.board_size[0] // nx)
            x = (i + 1) * (self.board_size[0] // nx)
            for j in range(ny):
                prev_y = j * (self.board_size[1] // ny)
                y = (j + 1) * (self.board_size[1] // ny)

                self.snakes.append(Snake(((x + prev_x) // 2, (y + prev_y) // 2),
                                         dir_to_center((x + prev_x) // 2, (y + prev_y) // 2, self.board_size)))
                if len(self.snakes) >= len(self.snakes):
                    break
            else:
                continue
            break

        self.apples = [Apple(self.board_size, self.get_occupied_positions()) for _ in range(self.num_apples)]

    def change_direction(self, snake, action):

        if action[0] == 1:
            snake.direction = direction_to_right(snake.direction)
        elif action[2] == 1:
            snake.direction = direction_to_left(snake.direction)

    def is_snake_dead(self, snake, ocupied_positions):
        head_pos = snake.body[0][0]
        if head_pos[0] < 0 or head_pos[0] >= self.board_size[0] or head_pos[1] < 0 or head_pos[1] >= self.board_size[1]:
            return True
        if head_pos in ocupied_positions:
            return True
        return False

    def is_collision(self, snake, direction):
        ocupied_positions = self.get_occupied_positions()
        head_pos = snake.body[0][0]
        dx, dy = direction
        new_head_pos = (head_pos[0] + dx, head_pos[1] + dy)
        if new_head_pos[0] < 0 or new_head_pos[0] >= self.board_size[0] or new_head_pos[1] < 0 or new_head_pos[1] >= \
                self.board_size[1] or new_head_pos in ocupied_positions:
            return True
        return False

    def get_occupied_positions(self):
        occupied_positions = []
        for snake in self.snakes:
            for pos, _ in snake.body:
                occupied_positions.append(pos)
        return occupied_positions

    def update(self):

        occupied_positions = self.get_occupied_positions()
        rewards = [0 for _ in range(self.num_snakes)]
        game_over = [False for _ in range(self.num_snakes)]
        for i, snake in enumerate(self.snakes):

            snake.move()
            snake.frame_iteration += 1

            if self.is_snake_dead(snake, occupied_positions) or snake.frame_iteration > 100 * len(snake.body):
                game_over[i] = True
                rewards[i] = -10
                continue

            for apple in self.apples:
                if snake.body[0][0] == (apple.x, apple.y):
                    snake.grow()
                    snake.score += 1
                    rewards[i] = 10
                    apple.x, apple.y = apple.spawn_new_apple(self.board_size, self.get_occupied_positions())

        scores = [snake.score for snake in self.snakes]
        return rewards, game_over, scores
