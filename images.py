import cv2 as cv

def load_images():
    apple = cv.imread('images/apple.png')

    head_down = cv.imread('images/snake_head.png')
    head_up = cv.rotate(head_down, cv.ROTATE_180)
    head_left = cv.rotate(head_down, cv.ROTATE_90_CLOCKWISE)
    head_right = cv.rotate(head_down, cv.ROTATE_90_COUNTERCLOCKWISE)
    head = [head_down, head_up, head_left, head_right]

    body_down = cv.imread('images/snake_body.png')
    body_up = cv.rotate(body_down, cv.ROTATE_180)
    body_left = cv.rotate(body_down, cv.ROTATE_90_CLOCKWISE)
    body_right = cv.rotate(body_down, cv.ROTATE_90_COUNTERCLOCKWISE)
    body = [body_down, body_up, body_left, body_right]

    tail_down = cv.imread('images/snake_tail.png')
    tail_up = cv.rotate(tail_down, cv.ROTATE_180)
    tail_left = cv.rotate(tail_down, cv.ROTATE_90_CLOCKWISE)
    tail_right = cv.rotate(tail_down, cv.ROTATE_90_COUNTERCLOCKWISE)
    tail = [tail_down, tail_up, tail_left, tail_right]

    bent_down_left = cv.imread('images/snake_body_bent.png')
    bent_down_right = cv.flip(bent_down_left, 1)
    bent_up_left = cv.flip(bent_down_left, 0)
    bent_up_right = cv.flip(bent_down_left, -1)
    bent_left_up = cv.rotate(bent_down_left, cv.ROTATE_90_CLOCKWISE)
    bent_left_down = cv.flip(bent_left_up, 0)
    bent_right_down = cv.rotate(bent_down_left, cv.ROTATE_90_COUNTERCLOCKWISE)
    bent_right_up = cv.flip(bent_right_down, 0)

    bent = [bent_down_left, bent_down_right, bent_up_left, bent_up_right, bent_left_up, bent_left_down, bent_right_up, bent_right_down]
    return apple, head, body, tail, bent


def get_head_direction(head, x, y):
    if x == 1:
        return head[3]
    elif x == -1:
        return head[2]
    elif y == 1:
        return head[0]
    elif y == -1:
        return head[1]


def get_body_direction(body, x, y):
    if x == 1:
        return body[3]
    elif x == -1:
        return body[2]
    elif y == 1:
        return body[0]
    elif y == -1:
        return body[1]


def get_tail_direction(tail, x, y):
    if x == 1:
        return tail[3]
    elif x == -1:
        return tail[2]
    elif y == 1:
        return tail[0]
    elif y == -1:
        return tail[1]


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

# _,_,_,_,bent = load_images()
# for b in bent:
#     cv.imshow('bent', b)
#     cv.waitKey(0)
#     cv.destroyAllWindows()