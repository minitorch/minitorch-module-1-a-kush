"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


# [done]: Implement for Task 0.1.
def mul(x: float, y: float):
    return 1.0 * x * y

def id(x: float):
    return 1.0 * x

def add(x: float, y: float):
    return 1.0 * x + y

def neg(x: float):
    return -1.0 * x

def lt(x: float, y: float):
    return float(x < y)

def eq(x: float, y: float):
    return float(x == y)

def max(x: float, y: float):
    if x >= y:
        return 1.0 * x
    return 1.0 * y


def is_close(x: float, y: float, thresh: float = 1e-5):
    return float(abs(x - y) <= thresh)

def sigmoid(x: float):
    if x >= 0:
        return 1.0 / (1 + math.exp(-x))
    else:
        return 1.0 * math.exp(x) / (1 + math.exp(x))
    
def relu(x: float):
    return 1.0 * max(0.0, x)

def log(x: float):
    return 1.0 * math.log(x)

def exp(x: float):
    return 1.0 * math.exp(x)

def inv(x: float):
    return 1.0 / x

def log_back(x: float, k: float):
    return 1.0 * k * inv(x) 

def inv_back(x: float, k: float):
    return -1.0 * k * inv(x) * inv(x)

def relu_back(x: float, k: float):
    if x > 0:
        return 1.0 * k
    else:
        return 0.0



# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


# [done]: Implement for Task 0.3.
def map(f: Callable, x: Iterable):
    return [f(elem) for elem in x]

def zipWith(f: Callable, x: Iterable, y: Iterable):
    return [f(elem_x, elem_y) for elem_x, elem_y in zip(x, y)]

def reduce(f: Callable, x: Iterable) -> float:
    res = 0.0
    for elem in x:
        if res is 0.0:
            res = elem
        else:
           res = f(res, elem) 
    return res

def negList(x: Iterable):
    return map(neg, x)

def addLists(x: Iterable, y: Iterable):
    return zipWith(add, x, y)

def sum(x: Iterable):
    return reduce(add, x)

def prod(x: Iterable):
    return reduce(mul, x)
        