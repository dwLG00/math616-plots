import numpy as np
import matplotlib.pyplot as plt
import math

# Problem 1 plots and aux functions

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

def plot_1b1(steps=20):
    # Plot the solution of the differential equation using euler's method

    # constants
    u_0 = 5
    a = 0.5
    f = 2
    # differential equation
    du = lambda u, t: -a * u + f
    # time range constants
    start_t = 0
    end_t = 2
    time_increment = (end_t - start_t) / steps

    # compute using euler
    time, data = euler(du, start_t, end_t, u_0, steps=steps)

    # plot
    plt.plot(time, data)
    plt.title(f"Euler's method plot of df = -0.5u + 2 with u_0 = 5, using {steps} steps")
    plt.xlabel("t")
    plt.ylabel("u")
    plt.show()

def plot_1b2():
    # Computes difference between euler's method and analytical function
    stepsizes = [math.ceil(step) for step in (2 / 1e-5, 2 / 5e-5, 2 / 1e-4, 2 / 5e-5, 2 / 1e-3, 2 / 5e-3)]

    # The solution to du/dt = -0.5u + 2 with u(0) = 5 is u = e^{-0.5t} + 4
    anal_sol = lambda t: np.float64(math.e)**(-0.5 * t) + 4

    # constants
    u_0 = np.float64(5)
    a = 0.5
    f = 2
    # differential equation
    du = lambda u, t: -a * u + f
    # Euler differences
    euler_diff = []
    for steps in stepsizes:
        time, euler_solution = euler(du, 0, 2, u_0, steps=steps)
        final_val = euler_solution[-1]
        print((steps, final_val))
        difference = abs(anal_sol(2) - final_val)
        euler_diff.append((steps, difference))

    timescale, value = zip(*euler_diff)

    # Regression
    m, b = np.polyfit(np.log(timescale), np.log(value), deg=1)

    plt.scatter(timescale, value)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("time step")
    plt.ylabel("absolute error")
    plt.title(f"Euler method error for various time steps (slope: {m})")
    plt.show()

# Problem 2 plots and aux functions

def plot_curve(a, b, c, f):
    # Used in Problem 2 a-b (i)
    func = lambda u: a*u + b*u**2 - c*u**3 + f
    u = [-2 + i * 0.01 for i in range(401)]
    val = list(map(func, u))
    plt.plot(u, val)
    plt.plot(u, [0 for _ in u])
    plt.xlabel("u")
    plt.ylabel(f"{a}u - {abs(b)}u^2 - {c}u^3 + {f}")
    plt.title(f"u vs {a}u - {abs(b)}u^2 - {c}u^3 + {f}")
    plt.show()

def plot_solutions(a, b, c, f):
    # Used in Problem 2 a-b (ii)
    du = lambda u, t: a*u + b*u**2 - c*u**3 + f
    u_0s = [-8, -4, -2, -1, 0, 1, 2, 4, 8]
    start_t = 0
    end_t = 1
    steps = 1000

    for u_0 in u_0s:
        time, data = euler(du, start_t, end_t, u_0, steps=steps)
        plt.plot(time, data)

    plt.title(f"du = {a}u - {abs(b)}u^2 - {c}u^3 + {f} with various u_0")
    plt.xlabel("t")
    plt.ylabel("u")
    plt.show()

def plot_2a1():
    # Constants
    a = 4
    b = -4
    c = 4
    f = 10

    plot_curve(a, b, c, f)

def plot_2a2():
    # Constants
    a = 4
    b = -4
    c = 4
    f = 10

    plot_solutions(a, b, c, f)


def plot_2b1():
    # Constants
    a = 4
    b = -4
    c = 4
    f = 2

    plot_curve(a, b, c, f)

def plot_2b2():
    # Constants
    a = 4
    b = -4
    c = 4
    f = 2

    plot_solutions(a, b, c, f)

# Problem 3 plots and aux functions

def plot_3(use_log=False, end_t=1, start_u1val=1):
    # setting use_log=True will display the plot using a symmetric log scale
    # setting end_t will set the end time
    # setting start_u1val will set the initial u_1 value

    start_t = 0
    steps = 10000
    u_0 = np.array([start_u1val, 0])

    # Case: a = -1
    A = np.array([[1, 1], [-1, 1]])
    du = lambda u, t: np.dot(A, u)
    _, data = euler(du, start_t, end_t, u_0, steps=steps)
    x, y = np.array(data).T
    plt.plot(x, y, label="a = -1")

    # Case: a = 0
    B = np.array([[0, 1], [-1, 0]])
    du = lambda u, t: np.dot(B, u)
    _, data = euler(du, start_t, end_t, u_0, steps=steps)
    x, y = np.array(data).T
    plt.plot(x, y, label="a = 0")

    # Case: a = 1
    C = np.array([[-1, 1], [-1, -1]])
    du = lambda u, t: np.dot(C, u)
    _, data = euler(du, start_t, end_t, u_0, steps=steps)
    x, y = np.array(data).T
    plt.plot(x, y, label="a = 1")

    plt.title("Trajectories of linear model given different values of a")
    if use_log:
        plt.xscale("symlog")
        plt.yscale("symlog")
    plt.xlabel("u_1")
    plt.ylabel("u_2")
    plt.legend()

    plt.show()

def print_usage():
    print("Usage: python3 plots.py [plot_number]\n"
        "List of plot numbers & descriptions:\n"
        "\t1: Problem 1b i) Numerical solution of du/dt = -au + f\n"
        "\t2: Problem 1b ii) Error of approximate solutions of du/dt = -au + f\n"
        "\t3: Problem 2a i) Plot of au + bu^2 - cu^3 + f as a function of u\n"
        "\t4: Problem 2a ii) Plot of du/dt = au + bu^2 - cu^3 + f for various u_0\n"
        "\t5: Problem 2b i) Plot of au + bu^2 - cu^3 + f as a function of u\n"
        "\t6: Problem 2b ii) Plot of du/dt = au + bu^2 - cu^3 + f for various u_0\n"
        "\t7: Problem 3: (logarithmic) plot of system of equations u, w\n")

if __name__ == "__main__":
    import sys
    try:
        plot_number = int(sys.argv[1])
        if plot_number == 1: plot_1b1()
        elif plot_number == 2: plot_1b2()
        elif plot_number == 3: plot_2a1()
        elif plot_number == 4: plot_2a2()
        elif plot_number == 5: plot_2b1()
        elif plot_number == 6: plot_2b2()
        elif plot_number == 7: plot_3(use_log=True, end_t=10, start_u1val=1)
        else: raise Error()
    except:
        print("Incorrect argument (or # of arguments)!")
        print_usage()


