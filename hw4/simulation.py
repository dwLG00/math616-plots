import numpy as np
import random
import matplotlib.pyplot as plt
import math

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

def generate_data(a=1, f=1, sigma=1, T=500, dt=0.01, x0=None):
    # Euler-Maruyama on the differential equation
    return euler_maruyama(sigma=sigma, f=lambda x, t: -a*x + f, T=T, dt=dt, x0=x0)

def sample_into(data, in_dt=0.01, out_dt=0.4):
    xs = [data[0]]
    spacing = int(out_dt / in_dt)
    for i in range(int((len(data) - 1) / spacing)):
        xs.append(data[i * spacing])

    return np.array(xs)

def kalman(obs_var=0.5, return_posteriors=False):
    # Generate the objective truth data
    base_data = sample_into(generate_data())
    #base_data = generate_data(dt=0.4)
    dt = 0.4
    A = 1
    F = 1
    sigma = 1

    pre_mean = 0
    pre_var = 0
    post_mean = 0.5
    post_var = 1

    forecasts = []
    posteriors = []

    def gain(pre_var, obs_var):
        return pre_var / (obs_var + pre_var)

    for i in range(len(base_data)):
        # Forecast the next using the posterior mean
        # Eqn taken from 10.1.2, where omega = 0 (due to lack of complex term)
        pre_mean = post_mean * math.e**(-A * dt) + (F / A) * (1 - math.e**(-A * dt))
        pre_var = post_var * math.e**(-2*A*dt) + sigma**2 / (2*A) * (1 - math.e**(-2 * A * dt))

        forecasts.append(pre_mean)

        # Observe, compute kalman gain, and update the posterior mean and variance
        observation = base_data[i] + np.random.normal(0, obs_var)
        kalman_gain = gain(pre_var, obs_var)
        #print('[%s] Forecast: %s; Observed: %s; delta: %s; Prior Var: %s; Observation Var: %s; Kalman Gain: %s' % (i, pre_mean, observation, forecast - observation, pre_var, obs_var, kalman_gain))
        post_mean = (1 - kalman_gain) * pre_mean + kalman_gain * observation
        post_var = (1 - kalman_gain) * pre_var
        posteriors.append(post_mean)

    if return_posteriors:
        return base_data, np.array(forecasts), np.array(posteriors)
    return base_data, np.array(forecasts)

def plot_param_estimate(T=5, dt=0.01):
    data = generate_data(0.5, 1, 0.4, dt=dt, x0=2)
    mean_delta = np.diff(data) / dt # (x(t + dt) - x(t))/dt, whose expected value should be f(x(t), t)
    xvals = data[:-1] #trim the last data point, as we can't predict off of it

    print('Original data: %s' % data)
    print('normalized deltas: %s' % mean_delta)

    n_samples = int(T / dt) + 1
    mean_delta = mean_delta[:n_samples]
    xvals = xvals[:n_samples]

    # Construct basis of our regression
    xs = np.stack([xvals**0, xvals**1, xvals**2, xvals**3, xvals**4, xvals**5])
    #xs = np.stack([xvals**0, xvals**1])
    xs = xs.T
    print(xs)

    print('xs: %s; mean_delta: %s' % (xs.shape, mean_delta.shape))

    # We want to find a vector beta = [f, -a, b, c, d, e] such that np.linalg.norm(xs @ x - mean_delta)^2 is minimized
    beta = np.linalg.inv(xs.T @ xs) @ xs.T @ mean_delta
    #beta, residuals, rank, singulars = np.linalg.lstsq(xs, mean_delta)
    #print(beta, residuals, rank, singulars)
    print(beta)

    estimated_f = lambda x, t: beta[1] * x + beta[2] * x**2 + beta[3] * x**3 + beta[4] * x**4 + beta[5] * x**5 + beta[0]
    #estimated_f = lambda x, t: beta[1] * x + beta[0]
    estimated_data = np.vectorize(lambda x: x + estimated_f(x, 0)*dt)(data[:n_samples])
    reconstruction = euler_maruyama(sigma=0.4, f=estimated_f, T=T, dt=0.01, x0=2)

    timescale = np.array([i * dt for i in range(n_samples)])
    plt.plot(timescale, data[:n_samples], 'o', label="ground truth")
    #plt.plot(timescale, mean_delta[:n_samples], label="dx data")
    plt.plot(timescale, estimated_data, label="regression estimate")
    plt.plot(timescale, reconstruction, label="euler-maruyama reconstruction")
    plt.legend()
    plt.title('Ground Truth v.s. Regression Estimate v.s. reconstructed E-M path, T=%s' % T)
    plt.xlabel("time")
    plt.ylabel("value")
    plt.show()


def plot_kalman1(obs_var=0.5):
    base_data, forecast, posteriors = kalman(obs_var=obs_var, return_posteriors=True)
    dt = 0.4
    T = 500
    time_axis = np.array([i * dt for i in range(int(T /dt) + 1)])

    print(len(time_axis), len(base_data), len(forecast))

    plt.plot(time_axis, base_data, label="ground truth")
    plt.plot(time_axis, forecast, label="kalman forecast")
    #plt.plot(time_axis, posteriors, label="posterior mean")
    plt.legend(loc='best')
    plt.title('Kalman forecast with observation variance 0.5')
    plt.xlabel('time')
    plt.ylabel('x')
    plt.show()

def plot_kalman2():
    data = {}
    for obs_var in map(lambda x: 0.01 * x, range(1, 100)):
        base_data, forecast = kalman(obs_var)
        base_data = base_data[int(50 / 0.4):]
        forecast = forecast[int(50 / 0.4):]
        rmse = np.sqrt(np.mean((base_data - forecast)**2))
        data[obs_var] = rmse

    plt.scatter(data.keys(), data.values())
    plt.title('Observation variance vs RMSE')
    plt.xlabel('observation variance')
    plt.ylabel('rmse')
    plt.show()

if __name__ == '__main__':
    #plot_kalman1(obs_var=0.01)
    plot_kalman1()
    plot_kalman2()
    plot_param_estimate()
    plot_param_estimate(T=50)

