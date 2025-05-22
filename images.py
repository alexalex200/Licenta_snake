import cv2 as cv
import random
import copy

def load_images(size = 32):
    apple = cv.imread('images/apple.png', cv.IMREAD_UNCHANGED)
    apple = cv.resize(apple, (size, size), interpolation=cv.INTER_NEAREST)
    apple = cv.cvtColor(apple, cv.COLOR_BGRA2RGBA)

    grass = cv.imread('images/grass.png', cv.IMREAD_UNCHANGED)
    grass = cv.resize(grass, (size, size), interpolation=cv.INTER_NEAREST)
    grass = cv.cvtColor(grass, cv.COLOR_BGRA2RGBA)

    mud = cv.imread('images/mud.png', cv.IMREAD_UNCHANGED)
    mud = cv.resize(mud, (size, size), interpolation=cv.INTER_NEAREST)
    mud = cv.cvtColor(mud, cv.COLOR_BGRA2RGBA)

    head_down = cv.imread('images/snake_head.png', cv.IMREAD_UNCHANGED)
    head_down = cv.resize(head_down, (size, size), interpolation=cv.INTER_NEAREST)
    head_down = cv.cvtColor(head_down, cv.COLOR_BGRA2RGBA)
    head_up = cv.rotate(head_down, cv.ROTATE_180)
    head_left = cv.rotate(head_down, cv.ROTATE_90_CLOCKWISE)
    head_right = cv.rotate(head_down, cv.ROTATE_90_COUNTERCLOCKWISE)
    head = [head_down, head_up, head_left, head_right]

    body_down = cv.imread('images/snake_body.png', cv.IMREAD_UNCHANGED)
    body_down = cv.resize(body_down, (size, size), interpolation=cv.INTER_NEAREST)
    body_down = cv.cvtColor(body_down, cv.COLOR_BGRA2RGBA)
    body_up = cv.rotate(body_down, cv.ROTATE_180)
    body_left = cv.rotate(body_down, cv.ROTATE_90_CLOCKWISE)
    body_right = cv.rotate(body_down, cv.ROTATE_90_COUNTERCLOCKWISE)
    body = [body_down, body_up, body_left, body_right]

    tail_down = cv.imread('images/snake_tail.png', cv.IMREAD_UNCHANGED)
    tail_down = cv.resize(tail_down, (size, size), interpolation=cv.INTER_NEAREST)
    tail_down = cv.cvtColor(tail_down, cv.COLOR_BGRA2RGBA)
    tail_up = cv.rotate(tail_down, cv.ROTATE_180)
    tail_left = cv.rotate(tail_down, cv.ROTATE_90_CLOCKWISE)
    tail_right = cv.rotate(tail_down, cv.ROTATE_90_COUNTERCLOCKWISE)
    tail = [tail_down, tail_up, tail_left, tail_right]

    bent_down_left = cv.imread('images/snake_body_bent.png', cv.IMREAD_UNCHANGED)
    bent_down_left = cv.resize(bent_down_left, (size, size), interpolation=cv.INTER_NEAREST)
    bent_down_left = cv.cvtColor(bent_down_left, cv.COLOR_BGRA2RGBA)
    bent_down_right = cv.flip(bent_down_left, 1)
    bent_up_left = cv.flip(bent_down_left, 0)
    bent_up_right = cv.flip(bent_down_left, -1)
    bent_left_up = cv.rotate(bent_down_left, cv.ROTATE_90_CLOCKWISE)
    bent_left_down = cv.flip(bent_left_up, 0)
    bent_right_down = cv.rotate(bent_down_left, cv.ROTATE_90_COUNTERCLOCKWISE)
    bent_right_up = cv.flip(bent_right_down, 0)

    bent = [bent_down_left, bent_down_right, bent_up_left, bent_up_right, bent_left_up, bent_left_down, bent_right_up, bent_right_down]
    return apple, grass, mud, head, body, tail, bent


def get_head_direction(head, x, y):
    if x == 1:
        return head[3]
    elif x == -1:
        return head[2]
    elif y == 1:
        return head[0]
    elif y == -1:
        return head[1]
    return None


def get_body_direction(body, x, y):
    if x == 1:
        return body[3]
    elif x == -1:
        return body[2]
    elif y == 1:
        return body[0]
    elif y == -1:
        return body[1]
    return None


def get_tail_direction(tail, x, y):
    if x == 1:
        return tail[3]
    elif x == -1:
        return tail[2]
    elif y == 1:
        return tail[0]
    elif y == -1:
        return tail[1]
    return None


def get_bent_direction(bent, x, y, next_x, next_y):
    if x == 1 and next_y == 1:
        return bent[0]
    elif x == -1 and next_y == 1:
        return bent[1]
    elif x == 1 and next_y == -1:
        return bent[2]
    elif x == -1 and next_y == -1:
        return bent[3]
    elif y == 1 and next_x == -1:
        return bent[4]
    elif y == -1 and next_x == -1:
        return bent[5]
    elif y == 1 and next_x == 1:
        return bent[6]
    elif y == -1 and next_x == 1:
        return bent[7]
    return None


def get_random_color():
    original = tuple(random.sample(range(0, 256), 3))
    while sum(original) < 100:
        original = tuple(random.sample(range(0, 256), 3))
    darker = tuple(int(x * 0.8) for x in original)
    brighter = tuple(int(x * 1.2) if int(x * 1.2) <= 255 else 255 for x in original)

    return original, darker, brighter

#168 230 29 original
#34 177 76 darker
#211 249 188 brighter

def change_colors(img, original, darker, brighter):
    img = copy.deepcopy(img)
    height, width = img.shape[:2]
    for i in range(height):
        for j in range(width):
            r, g, b, a = img[i, j]
            if r == 168 and g == 230 and b == 29:
                img[i, j] = [*original, a]
            elif r == 34 and g == 177 and b == 76:
                img[i, j] = [*darker, a]
            elif r == 211 and g == 249 and b == 188:
                img[i, j] = [*brighter, a]
    return img
