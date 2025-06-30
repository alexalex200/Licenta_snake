from game import Game
from draw_game import Draw
from plots import Plot
def choose_action(x, y, board_size):
    if x == 0:
        return 0
    if x == board_size[0] - 2 and y != board_size[1] - 1 and y != 0:
        return 2

    if x == board_size[0] - 1 and y == board_size[1] - 1:
        return 0

    if x == board_size[0] - 1 and y == 0:
        return 0

    return 1

if __name__ == "__main__":
    game = Game((6, 6), num_apples=1)
    draw = Draw(game, None, None)
    steps = 0
    n = 0

    running = True
    while running:
        game.reset()
        steps = 0
        while True:
            draw.draw()
            steps += 1
            action = choose_action(game.snake.body[0][0][0], game.snake.body[0][0][1], game.board_size)
            reward, done, score = game.step(action)
            print(reward)
            if done:
                break


