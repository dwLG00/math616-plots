import numpy as np
import random
import matplotlib.pyplot as plt
import math

def monte_carlo_e(n=100):
    # We take a uniformly random value in [0, 3], and sum the characteristic ln(x) <= 1.
    chi = lambda x: 1 if np.log(x) <= 1 else 0

    hits = sum(chi(random.random() * 3) for _ in range(n))
    return 3 * hits / n

def error_computations(n_trials=100):
    n_values = [100, 1000, 10000, 100000]
    variances = []

    for n in n_values:
        errors = np.array([monte_carlo_e(n=n) - math.e for _ in range(n_trials)])
        square_errors = errors ** 2
        variance = square_errors.mean() - errors.mean() ** 2
        variances.append(variance)

    plt.loglog(n_values, variances)
    plt.title('Variance of Monte Carlo simulation vs sample count')
    plt.xlabel('Number of samples')
    plt.ylabel('Variance')
    plt.show()

if __name__ == '__main__':
    error_computations()
