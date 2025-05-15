import random
import math
import numpy as np

class Snake:
    def __init__(self,start_pos, direction, length=3):
        self.body = []
        self.direction = direction
        self.dead = False
        self.score = 0
        x,y = start_pos
        dx,dy = direction
        for i in range(length):
            self.body.append(((x - i*dx,y - i*dy),direction))

    def move(self):
        self.body.pop()
        x,y = self.body[0][0]
        dx,dy = self.direction
        self.body.insert(0,((x+dx,y+dy),self.direction))

    def grow(self):
        x,y = self.body[-1][0]
        dx,dy = self.body[-1][1]
        self.body.append(((x-dx,y-dy),(dx,dy)))

class Apple:
    def __init__(self, board_size, occupied_positions):
        self.x, self.y = self.spawn_new_apple(board_size, occupied_positions)


    def spawn_new_apple(self, board_size, occupied_positions):
        while True:
            x = random.randint(0, board_size[0] - 1)
            y = random.randint(0, board_size[1] - 1)
            if (x,y) not in occupied_positions:
                return (x,y)

def dir_to_center(x,y,board_size):
    center_x = board_size[0] / 2
    center_y = board_size[1] / 2
    dx = 0
    dy = 0

    if x <= center_x:
        dx = 1
    elif x > center_x:
        dx = -1

    if y <= center_y:
        dy = 1
    elif y > center_y:
        dy = -1

    if dx != 0:
        return dx, 0
    elif dy != 0:
        return 0, dy
    else:
        return 0, 0

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
        ( 0, -1): 1,
        ( 1, -1): 2,
        ( 1,  0): 3,
        ( 1,  1): 4,
        ( 0,  1): 5,
        (-1,  1): 6,
        (-1,  0): 7,
        ( 0,  0): -1  # Special case: already at center
    }

    return direction_map[(dx, dy)]


def in_radius(apple_x, apple_y, snake_x, snake_y, radius):
    if (apple_x - snake_x) ** 2 + (apple_y - snake_y) ** 2 <= radius ** 2:
        return 1 - ((apple_x - snake_x) ** 2 + (apple_y - snake_y) ** 2) / radius ** 2
    else:
        return 0

class Game:
    def __init__(self, board_size=(20, 20), num_snakes=2):
        self.board_size = board_size
        self.snakes = []
        self.num_snakes = num_snakes

        nx,ny = math.ceil(math.sqrt(num_snakes)), math.ceil(math.sqrt(num_snakes))
        for i in range(nx):
            prev_x = i * (board_size[0] // nx)
            x = (i + 1) * (board_size[0] // nx)
            for j in range(ny):
                prev_y = j * (board_size[1] // ny)
                y = (j + 1) * (board_size[1] // ny)

                self.snakes.append(Snake(((x+prev_x)//2,(y+prev_y)//2),dir_to_center((x+prev_x)//2,(y+prev_y)//2,board_size)))
                if len(self.snakes) >= num_snakes:
                    break
            else:
                continue
            break

        self.apples = [Apple(board_size, self.get_occupied_positions()) for _ in range(num_snakes)]


    def change_direction(self, snake, direction):
        if snake.direction[0] != direction[0] and snake.direction[1] != direction[1]:
            snake.direction = direction

    def is_snake_dead(self, snake, ocupied_positions):
        head_pos = snake.body[0][0]
        if head_pos[0] < 0 or head_pos[0] >= self.board_size[0] or head_pos[1] < 0 or head_pos[1] >= self.board_size[1]:
            return True
        if head_pos in ocupied_positions:
            return True
        return False

    def dead_next(self, snake_head, occupied_positions):
        if snake_head[0] < 0 or snake_head[0] >= self.board_size[0] or snake_head[1] < 0 or snake_head[1] >= self.board_size[1]:
            return True
        if snake_head in occupied_positions:
            return True
        return False

    def get_occupied_positions(self):
        occupied_positions = []
        for snake in self.snakes:
            for pos,_ in snake.body:
                occupied_positions.append(pos)
        return occupied_positions

    def update(self):

        occupied_positions = self.get_occupied_positions()
        for i,snake in enumerate(self.snakes):
            if snake.dead:
                continue

            snake.move()
            snake.score += 0.1

            if self.is_snake_dead(snake, occupied_positions):
                self.snakes[i].dead = True
                self.snakes[i].score -= 10
                self.snakes[i].body = []
                continue

            for apple in self.apples:
                if snake.body[0][0] == (apple.x, apple.y):
                    snake.grow()
                    snake.score += 10
                    apple.x, apple.y = apple.spawn_new_apple(self.board_size, self.get_occupied_positions())

    def reset(self):
        self.snakes = []
        self.apples = []
        nx,ny = math.ceil(math.sqrt(self.num_snakes)), math.ceil(math.sqrt(self.num_snakes))
        for i in range(nx):
            prev_x = i * (self.board_size[0] // nx)
            x = (i + 1) * (self.board_size[0] // nx)
            for j in range(ny):
                prev_y = j * (self.board_size[1] // ny)
                y = (j + 1) * (self.board_size[1] // ny)

                self.snakes.append(Snake(((x+prev_x)//2,(y+prev_y)//2),dir_to_center((x+prev_x)//2,(y+prev_y)//2,self.board_size)))
                if len(self.snakes) >= len(self.snakes):
                    break
            else:
                continue
            break

        self.apples = [Apple(self.board_size, self.get_occupied_positions()) for _ in range(len(self.snakes))]

    def preprocess_state(self, agent_index):

        if self.snakes[agent_index].dead:
            return np.zeros(4 + 8 + 4, dtype=np.float32)
        processed_state = []
        processed_state.extend(self.snakes[agent_index].body[0][0])

        directions = [0 for _ in range(4)]
        if self.snakes[agent_index].direction[0] == 1:
            directions[0] = 1
        if self.snakes[agent_index].direction[0] == -1:
            directions[1] = 1
        if self.snakes[agent_index].direction[1] == 1:
            directions[2] = 1
        if self.snakes[agent_index].direction[1] == -1:
            directions[3] = 1

        processed_state.extend(directions)


        food = [0 for _ in range(8)]
        for apple in self.apples:
            index = direction_index_to_center(apple.x, apple.y, self.board_size[0], self.board_size[1])
            score = in_radius(apple.x, apple.y, self.snakes[agent_index].body[0][0][0], self.snakes[agent_index].body[0][0][1], self.board_size[0] // 2)
            if food[index] < score:
                food[index] = score

        processed_state.extend(food)


        danger = [0 for _ in range(4)]
        if self.dead_next((self.snakes[agent_index].body[0][0][0] , self.snakes[agent_index].body[0][0][1] - 1), self.get_occupied_positions()):
            danger[0] = 1
        if self.dead_next((self.snakes[agent_index].body[0][0][0] - 1, self.snakes[agent_index].body[0][0][1] ), self.get_occupied_positions()):
            danger[1] = 1
        if self.dead_next((self.snakes[agent_index].body[0][0][0] , self.snakes[agent_index].body[0][0][1] + 1), self.get_occupied_positions()):
            danger[2] = 1
        if self.dead_next((self.snakes[agent_index].body[0][0][0] + 1, self.snakes[agent_index].body[0][0][1] ), self.get_occupied_positions()):
            danger[3] = 1

        processed_state.extend(danger)

        return np.array(processed_state, dtype=np.float32)

