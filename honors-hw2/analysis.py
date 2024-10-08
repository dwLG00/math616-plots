import pandas as pd
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import numpy.linalg as LA
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import scipy.stats as stats

def load(dir):
    #amplitude_path = os.path.join(dir, "amplitude.tsv")
    #phase_path = os.path.join(dir, "phase.tsv")

    # RMM data file paths
    rmm1_path = os.path.join(dir, "rmm1.tsv")
    rmm2_path = os.path.join(dir, "rmm2.tsv")

    # Load RMM data files, and merge into a single pandoc df
    rmm1 = pd.read_csv(rmm1_path, sep="\t", header=None).transpose()
    rmm2 = pd.read_csv(rmm2_path, sep="\t", header=None).transpose()
    rmm_df = pd.concat([rmm1[0], rmm2[0]], axis=1, keys=['RMM1', 'RMM2'])

    # Convert pandoc df into numpy array
    rmm_np = rmm_df.to_numpy()

    # Use date range of 1 Jan 1999 to 31 Dec 2013 (same as in paper)
    rmm_np = rmm_np[8980:(8980+5479)]

    # "Normalize" by subtracting mean from dataset
    mean1 = np.mean(rmm_np[:, 0])
    mean2 = np.mean(rmm_np[:, 1])
    rmm_np[:, 0] = rmm_np[:, 0] - mean1
    rmm_np[:, 1]= rmm_np[:, 1] - mean2

    return rmm_np

def generate_figure_1(rmm_np):
    '''Generate a figure similar to figure 1'''

    start = 365
    end=730

    data = rmm_np[start:end, :]
    rmm1 = data[:, 0]
    rmm2 = data[:, 1]

    plt.plot(rmm1)
    plt.xlabel("Days since Jan 1, 2000")
    plt.ylabel("Centered RMM1")
    plt.title("RMM1 between Jan 1, 2000 - Dec 31, 2000")
    plt.show()

    plt.plot(rmm2)
    plt.xlabel("Days since Jan 1, 2000")
    plt.ylabel("Centered RMM2")
    plt.title("RMM2 between Jan 1, 2000 - Dec 31, 2000")
    plt.show()

    # Generate histogram pdfs
    rmm1 = rmm_np[:, 0]
    rmm2 = rmm_np[:, 1]
    density1 = stats.gaussian_kde(rmm1)
    density2 = stats.gaussian_kde(rmm2)

    rmm1_min = min(rmm1)
    rmm1_max = max(rmm1)
    rmm2_min = min(rmm2)
    rmm2_max = max(rmm2)

    _, bins1 = np.histogram(rmm1, bins=50, range=(rmm1_min, rmm1_max))
    plt.plot(bins1, density1(bins1))
    plt.xlabel("RMM1 value")
    plt.ylabel("Probability density")
    plt.title("RMM1 PDF")
    plt.show()

    _, bins2 = np.histogram(rmm2, bins=50, range=(rmm2_min, rmm2_max))
    plt.plot(bins2, density1(bins2))
    plt.xlabel("RMM2 value")
    plt.ylabel("Probability density")
    plt.title("RMM2 PDF")
    plt.show()

def trad_ssa(rmm_np, M=51):
    # Computes traditional SSA algorithm on rmm with window size M

    N = rmm_np.shape[0]

    print("Means: %s, %s" % (mean1, mean2))
    print(rmm_np, rmm_np.shape)

    # Compute window size, and construct time-lagged datapoint array
    window_size = N - M + 1
    X = sliding_window_view(rmm_np, window_size, axis=0)
    X = np.concatenate(X, axis=0)
    XT = X.transpose()

    # Create covariance matrix, and find eigenvalues/eigenvectors
    C = X.dot(XT) / window_size
    eigenvalues, eigenvectors = LA.eig(C)

    def MLU(t):
        # Sum count and lower/upper bounds for sum (used for generating z(t))
        if t < M - 1:
            return (t + 1, 0, t+1)
        elif M - 1 <= t < N - M + 1:
            return (M, 0, M)
        elif N - M + 1 <= t < N:
            return (N - t, t - N + M, M)

    def generate_t(phi, w):
        # Construct z(t) from phi and w
        l = []
        for t in range(N):
            M_t, L_t, U_t = MLU(t)
            x = 1 / M_t * sum(phi[t - i] * w[i] for i in range(L_t, U_t))
            l.append(x)
        return np.array(l)

    # Construct RCs
    zs = []
    for i in tqdm(range(eigenvalues.size)):
        v = eigenvectors[i]
        phi = XT.dot(v) # Projection of v onto the timelagged data space
        w = np.reshape(v, (M, 2)) # Group v into a sequence of pair vectors (RMM1, RMM2)
        zs.append(generate_t(phi, w))

    print(zs)
