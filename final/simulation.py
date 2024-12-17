import numpy as np
import random
import matplotlib.pyplot as plt
import math
from tqdm import tqdm

def euler_maruyama(sigma=1, f=lambda x, t: -x, T=500, dt=0.01, x0=None):
    if x0 == None:
        xs = [np.random.normal(0, sigma)]
    else:
        xs = [x0]
    for i in range(int(T / dt)):
        x = xs[-1]
        t = i * dt
        diffx = f(x, t) * dt + np.random.normal(0, sigma * dt)
        xs.append(x + diffx)
    return np.array(xs)

def ts(T=500, dt=0.01, shift=0):
    n_samples = int(T / dt) + 1
    timescale = np.array([i * dt for i in range(n_samples)])
    return timescale + shift

def problem3_em():
    # Part a
    # dx_t = -0.5 x_t dt + 0.4 dW_t
    dt = 0.05
    T = 8
    N = 20
    true_timescale = ts(T=50, dt=dt)
    timescale = ts(T=T, dt=dt)
    generate_sample = lambda : euler_maruyama(sigma=0.4, f=lambda x, t: -0.5*x, T=8, dt=dt, x0=2) # nullary function for generating a single sample
    true_truth = euler_maruyama(sigma=0.4, f=lambda x, t: -0.5*x, T=50, dt=dt, x0=2)
    truth = true_truth[:len(timescale)] # ends at T = 8

    # Part b
    # Generate ensemble data and plot
    ensemble = [generate_sample() for _ in range(20)]
    ensemble_mean = sum(ensemble) / N
    # find stdev
    ensemble_stack = np.stack(ensemble)
    ensemble_std = np.std(ensemble_stack, axis=0)

    for ensemble_member in ensemble:
        plt.plot(timescale, ensemble_member, color="lightgreen")
    plt.plot(timescale, truth, label="truth", color="blue")
    plt.plot(timescale, ensemble_mean, label="ensemble mean", color="green")
    plt.fill_between(timescale, ensemble_mean - ensemble_std, ensemble_mean + ensemble_std, alpha=0.5, label="ensemble mean +- std")
    plt.legend()
    plt.title("Truth and 20 ensemble members")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.show()

    # Part c
    # only the first forecast used for the first plot
    index = lambda t: int(t / dt) # convert between time units and index
    forecasts = np.array([2] + [euler_maruyama(sigma=0.4, f = lambda x, t: -0.5*x, T=1, dt=dt, x0=truth[index(i)])[-1] for i in range(8)])
    forecast_timescale = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8]) # manual timescale
    # ^ essentially, for t \in [0, 1, \dots, 8-n] do euler-maruyama for n time units, starting at truth[t], and nab the last value (t+n)

    plt.plot(timescale, truth, label="truth", color="blue")
    plt.plot(forecast_timescale, forecasts, label="forecast", color="#ed1717")
    plt.legend()
    plt.title("Truth and forecast (lead 1)")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.show()

    forecasts2 = np.array([2] + [euler_maruyama(sigma=0.4, f = lambda x, t: -0.5*x, T=2, dt=dt, x0=truth[index(i)])[-1] for i in range(7)])
    forecast_timescale2 = np.array([0, 2, 3, 4, 5, 6, 7, 8]) # manual timescale
    forecasts3 = np.array([2] + [euler_maruyama(sigma=0.4, f = lambda x, t: -0.5*x, T=3, dt=dt, x0=truth[index(i)])[-1] for i in range(6)])
    forecast_timescale3 = np.array([0, 3, 4, 5, 6, 7, 8]) # manual timescale
    forecasts4 = np.array([2] + [euler_maruyama(sigma=0.4, f = lambda x, t: -0.5*x, T=4, dt=dt, x0=truth[index(i)])[-1] for i in range(5)])
    forecast_timescale4 = np.array([0, 4, 5, 6, 7, 8]) # manual timescale

    plt.plot(timescale, truth, label="truth", color="blue")
    plt.plot(forecast_timescale, forecasts, label="forecast", color="red")
    plt.plot(forecast_timescale2, forecasts2, label="forecast", color="orange")
    plt.plot(forecast_timescale3, forecasts3, label="forecast", color="yellow")
    plt.plot(forecast_timescale4, forecasts4, label="forecast", color="green")
    plt.legend()
    plt.title("Truth and forecasts (leads 1-4)")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.show()

    def rmse(x, y):
        return np.sqrt(np.mean((x - y)**2))

    corrs = []
    rmses = []
    # lead times
    for t in range(1, 48):
        forecasts_full = np.array([euler_maruyama(sigma=0.4, f=lambda x, t: -0.5*x, T=t, dt=dt, x0=true_truth[index(i)])[-1] for i in range(50 - t)])
        truth_values = np.array([true_truth[index(i + t)] for i in range(50 - t)])
        corr = np.correlate(forecasts_full, truth_values)[0]
        rmse_ = rmse(forecasts_full, truth_values)
        corrs.append(corr)
        rmses.append(rmse_)
    #forecasts_full_1 = np.array([euler_maruyama(sigma=0.4, f=lambda x, t: -0.5*x, T=1, dt=dt, x0=truth[index(i)])[-1] for i in range(49)])
    #truth_values_1 = np.array([truth[index(i + 1)] for i in range(49)])
    #forecasts_full_2 = np.array([euler_maruyama(sigma=0.4, f=lambda x, t: -0.5*x, T=2, dt=dt, x0=truth[index(i)])[-1] for i in range(48)])
    #truth_values_2 = np.array([truth[index(i + 2)] for i in range(48)])
    #forecasts_full_3 = np.array([euler_maruyama(sigma=0.4, f=lambda x, t: -0.5*x, T=3, dt=dt, x0=truth[index(i)])[-1] for i in range(47)])
    #truth_values_3 = np.array([truth[index(i + 3)] for i in range(47)])
    #forecasts_full_4 = np.array([euler_maruyama(sigma=0.4, f=lambda x, t: -0.5*x, T=4, dt=dt, x0=truth[index(i)])[-1] for i in range(46)])
    #truth_values_4 = np.array([truth[index(i + 4)] for i in range(46)])


    #corr_1 = np.correlate(forecasts_full_1, truth_values_1)[0]
    #rmse_1 = rmse(forecasts_full_1, truth_values_1)
    #corr_2 = np.correlate(forecasts_full_2, truth_values_2)[0]
    #rmse_2 = rmse(forecasts_full_2, truth_values_2)
    #corr_3 = np.correlate(forecasts_full_3, truth_values_3)[0]
    #rmse_3 = rmse(forecasts_full_3, truth_values_3)
    #corr_4 = np.correlate(forecasts_full_4, truth_values_4)[0]
    #rmse_4 = rmse(forecasts_full_4, truth_values_4)

    times = np.array([i for i in range(1, 48)])
    corrs = np.array(corrs)
    rmses = np.array(rmses)
    plt.plot(times, corrs, label="correlation")
    plt.plot(times, rmses, label="rmse")
    plt.legend()
    plt.title("correlation and rmse vs lead times")
    plt.xlabel("lead time")
    plt.ylabel("value")
    plt.show()

def problem4_kalman():
    A12 = 0.2
    A21 = 0.2
    obs_var = 0.1
    n = 50
    base_data, observations, (pre_means, pre_vars), (post_means, post_vars) = kalman(A12, A21, obs_var, n=n)

    base_u1 = base_data[:, 0]
    base_u2 = base_data[:, 1]
    pre_means_u1 = pre_means[:, 0]
    pre_means_u2 = pre_means[:, 1]
    post_means_u1 = post_means[:, 0]
    post_means_u2 = post_means[:, 1]
    pre_vars_u1 = pre_vars[:, 0, 0]
    pre_vars_u2 = pre_vars[:, 1, 1]
    post_vars_u1 = post_vars[:, 0, 0]
    post_vars_u2 = post_vars[:, 1, 1]

    timeseries = np.array(range(n))

    # plot u1
    plt.plot(timeseries, base_u1, label="truth")
    plt.scatter(timeseries, observations, label="observations", marker=".", color="lightgreen")
    plt.plot(timeseries, pre_means_u1, label="forecast")
    plt.plot(timeseries, post_means_u1, label="posterior mean")
    plt.legend()
    plt.title("truth, observations, and forecast u1")
    plt.xlabel("time")
    plt.ylabel("value")
    plt.show()

    plt.plot(timeseries, base_u2, label="truth")
    plt.scatter(timeseries, observations, label="observations", marker=".", color="lightgreen")
    plt.plot(timeseries, pre_means_u2, label="forecast")
    plt.plot(timeseries, post_means_u2, label="posterior mean")
    plt.legend()
    plt.title("truth, observations, and forecast u2")
    plt.xlabel("time")
    plt.ylabel("value")
    plt.show()

    plt.plot(timeseries, pre_vars_u1, label="prior variance")
    plt.plot(timeseries, post_vars_u1, label="posterior variance")
    plt.legend()
    plt.title("prior vs posterior variance u1")
    plt.xlabel("time")
    plt.ylabel("variance")
    plt.show()

    plt.plot(timeseries, pre_vars_u2, label="prior variance")
    plt.plot(timeseries, post_vars_u2, label="posterior variance")
    plt.legend()
    plt.title("prior vs posterior variance u2")
    plt.xlabel("time")
    plt.ylabel("variance")
    plt.show()


def kalman(A12, A21, obs_var, n=500):
    # Generate the objective truth data
    A = np.array([[-0.8, A12], [A12, -0.8]])
    base_data = generate_kalman_data(A, r=0.01, n=n)
    #base_data = generate_data(dt=0.4)
    G = np.array([[1], [0]]).T # observation only occurs on u1
    meano = 0
    Ro = obs_var

    # arbitrarily-chosen pre_mean and pre_var
    pre_mean = np.array([[0], [0]])
    pre_var = np.array([[0, 0], [0, 0]])
    post_mean = np.array([[0.5], [0.5]])
    post_var = np.array([[1, 0], [0, 1]])

    observations = []
    forecasts = []
    posteriors = []
    pre_vars = []
    post_vars = []

    def gain(pre_var, obs_var):
        return (pre_var @ G.T) * (G @ pre_var @ G.T + Ro)**(-1)
        #return pre_var / (obs_var + pre_var)

    for i in tqdm(range(n)):
        # Forecast the next using the posterior mean
        # Eqn taken from 10.1.2, where omega = 0 (due to lack of complex term)
        pre_mean = A.dot(post_mean)
        pre_var = A @ post_var @ (A.T)
        #pre_mean = post_mean * math.e**(-A * dt) + (F / A) * (1 - math.e**(-A * dt))
        #pre_var = post_var * math.e**(-2*A*dt) + sigma**2 / (2*A) * (1 - math.e**(-2 * A * dt))

        forecasts.append(pre_mean)
        pre_vars.append(pre_var)

        # Observe, compute kalman gain, and update the posterior mean and variance
        observation = G @ base_data[i] + np.random.normal(meano, Ro)
        observations.append(observation)
        kalman_gain = gain(pre_var, obs_var)
        #print('[%s] Forecast: %s; Observed: %s; delta: %s; Prior Var: %s; Observation Var: %s; Kalman Gain: %s' % (i, pre_mean, observation, forecast - observation, pre_var, obs_var, kalman_gain))
        post_mean = pre_mean + kalman_gain @ (observation - (G @ pre_mean))
        post_var = (np.identity(2) - (kalman_gain @ G)) @ pre_var
        #post_mean = (1 - kalman_gain) * pre_mean + kalman_gain * observation
        #post_var = (1 - kalman_gain) * pre_var
        posteriors.append(post_mean)
        post_vars.append(post_var)

    return base_data, np.array(observations), (np.array(forecasts), np.array(pre_vars)), (np.array(posteriors), np.array(post_vars))

def generate_kalman_data(A, r, n):
    l = []
    u = np.array([[1], [1]]) #initial condition

    for _ in range(n):
        l.append(u)
        u_next = A @ u
        noise = np.array([[np.random.normal(0, r)], [np.random.normal(0, r)]])
        u = u_next + noise

    return np.array(l)

def problem5_param():
    # dx/dt = ax + bsin(2pi t + phi)x^2 - e^c x^3 + sigma W
    # bsin(2pi t + phi) = bsin(2pi t)cos(phi) + bsin(phi)cos(2pi t) = b1 sin(2pi t) + b2 cos(2pi t)
    #   where b1 = b cos(phi) and b2 = b sin(phi) <- use to recover b, phi
    # c' = e^c <- use to recover c
    dt = 0.1
    T = 500
    timeseries, truth_raw = generate_p5_data(T=T, dt=dt)
    truth = truth_raw[:-1] #cut off the last value bc we don't have its slope

    # array of derivatives (true value)
    dx = np.diff(truth_raw) / dt

    # estimate sin(2pi t) and cos(2pi t)
    sin = np.vectorize(lambda t: math.sin(2*math.pi*t))
    cos = np.vectorize(lambda t: math.cos(2*math.pi*t))
    sin_row = sin(timeseries)[:-1]
    cos_row = cos(timeseries)[:-1]

    print(sin_row)
    print(cos_row)

    # parameter estimation via linear regression
    x_row = truth.copy()
    x2_row_sin = truth**2 * sin_row
    x2_row_cos = truth**2 * cos_row
    x3_row = truth**3

    # downsample to prevent overfitting

    x_row = np.array([x_row[10*i] for i in range(500)])
    x2_row_sin = np.array([x2_row_sin[10*i] for i in range(500)])
    x2_row_cos = np.array([x2_row_cos[10*i] for i in range(500)])
    x3_row = np.array([x3_row[10*i] for i in range(500)])

    stack = np.stack([x_row, x2_row_sin, x2_row_cos, x3_row]) # should be a matrix of column vectors [x_i, x^2_i, x^3_i]^T

    print(x2_row_sin)
    print(x2_row_cos)

    print(stack.shape)
    truth = np.array([truth[10*i] for i in range(500)])
    print(truth.shape)

    # Linear regression: minimize |stack @ params - truth|
    coeffs, residuals, rank, s = np.linalg.lstsq(stack.T, truth, rcond = None)
    print(coeffs)
    a = coeffs[0]
    _b1 = coeffs[1]
    _b2 = coeffs[2]
    _c = -coeffs[3] # prevent log
    phi = math.atan(_b2 / _b1)
    b = _b1 / math.cos(phi)
    c = math.log(_c)
    print('a: %s, b: %s, c: %s, phi: %s' % (a, b, c, phi))
    print('residuals: %s' % residuals)
    print('rank: %s' % rank)
    print('singular values: %s' % s)

    _, regenerated = generate_p5_data(T=T, dt=dt, a=a, b=b, phi=phi, c=c)
    plt.plot(timeseries, truth_raw, label="truth")
    plt.plot(timeseries, regenerated, label="recreated")
    print(truth_raw)
    print(regenerated)
    plt.legend()
    plt.title("Truth vs Estimated parameter recreated data")
    plt.xlabel("time")
    plt.ylabel("value")
    plt.show()

def generate_p5_data(T=500, dt=0.01, a=-1, b=1, phi=math.pi, c=0.5, sigma=0.5):
    x0 = 0
    timeseries = ts(T=T, dt=dt)
    def f(x, t):
        return a*x + b*math.sin(2*math.pi*t + phi) * x**2 - math.e**c * x**3
    data = euler_maruyama(sigma=sigma, f=f, T=T, dt=dt, x0=x0)
    return timeseries, data

if __name__ == '__main__':
    #problem3_em()
    #problem4_kalman()
    problem5_param()
