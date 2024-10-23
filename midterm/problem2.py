import numpy as np

# Code recycled from hw1
def euler(dx, start_t, end_t, x0, steps=20):
    # dx(x, t) = x'

    l = [x0]
    time = [start_t]
    dt = (end_t - start_t) / steps
    for i in range(1, steps+1):
        t = start_t + i * dt
        x = l[-1]
        l.append(x + dt * dx(x, t))
        time.append(t)
    return time, l

def compute_forward_euler():
    # Computes the forward euler approximation to the following ODE:
    #   du_1/dt = -2u_1 + u_2 + 1
    #   du_2/dt = 2u_1 - 3u_2 + 1
    # with initial conditions u_1(0) = 0.5, u_2(0) = 1.5, and end time T=5.

    x_0 = np.array([0.5, 1.5])
    start_t = 0
    end_t = 5
    steps = 50
    A = np.array([[-2, 1], [2, -3]])
    b = np.array([1, 1])
    dx = lambda x, t: A.dot(x) + b

    time, us = euler(dx, start_t, end_t, x_0, steps=steps)

    return us[-1]

if __name__ == '__main__':
    print("estimated value of [u_1, u_2] at T=5 is %s" % compute_forward_euler())
