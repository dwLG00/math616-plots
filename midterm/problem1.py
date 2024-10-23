import numpy as np
import random
import matplotlib.pyplot as plt
import math

def euler_maruyama(a, b, x_0, dt=0.1, steps=10, return_timeseries=True):
    # Euler-Maruyama estimation for dx_t = a(x, t)dt + b(x, t)dW_t
    # with initial condition x_0 and time-increment dt

    x = x_0
    l = [(0, x_0)]
    for i in range(steps):
        t = i*dt
        t1 = (i + 1)*dt
        x = x + a(x, t) * dt + b(x, t) * math.sqrt(dt) * random.gauss(0, 1)
        l.append((t1, x))

    if return_timeseries:
        return l
    return l[-1]

def plot_timeseries():
    # Plots timeseries of one particular path of differential equation
    random.seed(0)
    #a = lambda x, t: -x
    #b = lambda x, t: 1
    a = lambda x, t: -x
    b = lambda x, t: t
    timeseries = euler_maruyama(a, b, 1, steps=1000)
    ts = [v[0] for v in timeseries]
    xs = [v[1] for v in timeseries]
    plt.plot(ts, xs)
    plt.xlabel('time')
    plt.ylabel('x value')
    plt.title('Euler-Maruyama simulation of dx_t = -x_t*dt + dW_t, x_0 = 1')
    plt.show()

def variance_estimate(n=10000, dt=0.1, steps=1000, plot_graphs=True, return_timeseries=False):
    #random.seed(0)
    #a = lambda x, t: -x
    #b = lambda x, t: 1
    a = lambda x, t: -x
    b = lambda x, t: t
    timeserieses = [np.array([v[1] for v in euler_maruyama(a, b, 1, dt=dt, steps=steps)]) for _ in range(n)]
    stack = np.stack(timeserieses)
    mean_stack_sq = np.sum(stack ** 2, axis = 0) / n # <x_t^2>
    sq_mean_stack = (np.sum(stack, axis = 0) / n) ** 2 # <x_t>^2
    variance = mean_stack_sq - sq_mean_stack

    if plot_graphs:
        plt.plot([dt * i for i in range(steps + 1)], variance)
        plt.title(f'Variance <x_t\'^2> estimated across n={n} paths')
        plt.xlabel('time')
        plt.ylabel('<x_t\'^2>')
        plt.show()

        print("We see that variance is about %s as t -> infinity." % variance[-1])
    if return_timeseries:
        return variance[-1], stack
    return variance[-1]

def autocorrelation(n=10000, dt=0.1, steps=1000):
    variance, timeseries = variance_estimate(n=n, dt=dt, steps=steps, plot_graphs=False, return_timeseries=True)
    xinf = timeseries[:, -1].mean()
    rs = []
    for s in range(steps):
        est_term = np.array([((timeseries[:, t] - xinf) * (timeseries[:, t+s] - xinf)).mean() for t in range(steps - s)])
        rs_term = est_term.mean() / variance
        rs.append(rs_term)

    plt.plot([i * dt for i in range(steps)], rs)
    plt.title(f'R(s) estimate with n={n}')
    plt.xlabel('s')
    plt.ylabel('R(s)')
    plt.show()
    return rs

if __name__ == '__main__':
    plot_timeseries()
    variance_estimate()
    autocorrelation()
