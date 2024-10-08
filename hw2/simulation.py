import math
import numpy as np
import random

def bernoulli(p) -> bool:
    '''Returns True with probability p and False with probability 1-p'''
    r = random.random()
    return r < p

def markov_expectation(n=10, k=10000, seed=True):
    p = 0.125
    q = 0.25

    distributions = [(int(k/2), int(k/2))]

    if seed: random.seed(0)
    for i in range(n):
        state1, state2 = distributions[-1]
        new_state1, new_state2 = 0, 0
        for _ in range(state1):
            if bernoulli(p):
                new_state2 += 1
            else:
                new_state1 += 1
        for _ in range(state2):
            if bernoulli(q):
                new_state1 += 1
            else:
                new_state2 += 1
        print((new_state1, new_state2))
        distributions.append((new_state1, new_state2))

    final_state1, final_state2 = distributions[-1]
    return (final_state1 + 2*final_state2) / k

def markov_matrix(n=10):
    matrix = np.array([[0.875, 0.25], [0.125, 0.75]])
    prod = matrix[:]
    for i in range(n - 1):
        prod = matrix.dot(prod)
    return prod

def monte_carlo_gaussian(n=1000, seed=False):
    if seed: random.seed(0)
    return sum(random.gauss(0, 1) for _ in range(n)) / n

def monte_carlo_sqrt_2(n=1000, seed=False):
    if seed: random.seed(0)
    f = lambda x: 1 if x**2 < 2 else 0
    uniforms = (random.random() * 2 for _ in range(n))
    integrands = (f(x) for x in uniforms)
    return 2*sum(integrands) / n
