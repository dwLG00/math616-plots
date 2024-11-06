import numpy as np
import math

def integrate(function, start, end, steps=100):
    if start == end:
        return 0
    stepsize = (end - start) / steps
    xs = np.arange(start, end, stepsize)
    ys = np.array(list(map(function, xs)))
    return np.trapz(ys, xs)

def multiplicative_noise_moments(a, b, c, f, stepsize=0.1, T=100):
    # compute first - fourth moments using trapezoidal integration

    xs = np.arange(0, T, stepsize)

    # nth moment functions
    moment1_memoize = {}
    moment2_memoize = {}
    moment3_memoize = {}
    moment4_memoize = {}

    def moment1(t):
        if moment1_memoize.get(t): return moment1_memoize.get(t)
        m1 = math.e**(-a*t) + integrate(lambda s: f*math.e**(-a*(t - s)), 0, t)
        moment1_memoize[t] = m1
        return m1

    # <x(0)^n> = 1, n >= 2
    def moment2(t):
        if moment2_memoize.get(t): return moment2_memoize.get(t)
        m2 = math.e**((b**2 - 2*a)*t) + integrate(lambda s: math.e**((b**2 - 2*a)*(t - s)) * (2*f*moment1(s) + c**2), 0, t)
        moment2_memoize[t] = m2
        return m2

    def moment3(t):
        if moment3_memoize.get(t): return moment3_memoize.get(t)
        m3 = math.e**((3*b**3 - 3*a)*t) + integrate(lambda s: math.e**((3*b**3 - 3*a)*(t - s)) * (3*f*moment2(s) + 3*c**2*moment1(s)), 0, t)
        moment3_memoize[t] = m3
        return m3

    def moment4(t):
        if moment4_memoize.get(t): return moment4_memoize.get(t)
        m4 = math.e**((6 * b**2 - 4*a)*t) + integrate(lambda s: math.e**((6 * b**2 - 4*a)*(t - s))*(4*f*moment3(s) + 6*c**2*moment2(s)), 0, t)
        moment4_memoize[t] = m4

    first_moment = np.array(list(map(moment1, xs)))
    second_moment = np.array(list(map(moment2, xs)))
    third_moment = np.array(list(map(moment3, xs)))
    fourth_moment = np.array(list(map(moment4, xs)))

    return first_moment, second_moment, third_moment, fourth_moment
