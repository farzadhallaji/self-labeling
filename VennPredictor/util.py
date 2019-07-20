import numpy as np


# Some elementary functions to speak the same language as the paper
# (at some point we'll just replace the occurrence of the calls with the function body itself)
def push(x, stack):
    stack.append(x)


def pop(stack):
    return stack.pop()


def top(stack):
    return stack[-1]


def nextToTop(stack):
    return stack[-2]


# perhaps inefficient but clear implementation
def nonleftTurn(a, b, c):
    d1 = b - a
    d2 = c - b
    return np.cross(d1, d2) <= 0


def nonrightTurn(a, b, c):
    d1 = b - a
    d2 = c - b
    return np.cross(d1, d2) >= 0


def slope(a, b):
    ax, ay = a
    bx, by = b
    return (by - ay) / (bx - ax)


def notBelow(t, p1, p2):
    p1x, p1y = p1
    p2x, p2y = p2
    tx, ty = t
    m = (p2y - p1y) / (p2x - p1x)
    b = (p2x * p1y - p1x * p2y) / (p2x - p1x)
    return ty >= tx * m + b
