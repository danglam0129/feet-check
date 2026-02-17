import cv2


def draw_point(img, x, y, color):
    cv2.circle(img, (int(x), int(y)), 6, color, -1)
