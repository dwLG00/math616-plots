import pandas as pd
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import numpy.linalg as LA
import os
from tqdm import tqdm

def load(dir):
    #amplitude_path = os.path.join(dir, "amplitude.tsv")
    #phase_path = os.path.join(dir, "phase.tsv")
    rmm1_path = os.path.join(dir, "rmm1.tsv")
    rmm2_path = os.path.join(dir, "rmm2.tsv")

    rmm1 = pd.read_csv(rmm1_path, sep="\t", header=None).transpose()
    rmm2 = pd.read_csv(rmm2_path, sep="\t", header=None).transpose()
    return pd.concat([rmm1[0], rmm2[0]], axis=1, keys=['RMM1', 'RMM2'])

def trad_ssa(rmm, M=51):
    # Computes traditional SSA algorithm on rmm with window size M

    # Convert to numpy. It's easier
    rmm_np = rmm.to_numpy()
    print(rmm_np, rmm_np.shape)

    # Use 1 Jan 1999 to 31 Dec 2013
    rmm_np = rmm_np[8980:(8980+5479)]
    N = rmm_np.shape[0]

    # Normalize
    mean1 = np.sum(rmm_np[:, 0]) / N
    mean2 = np.sum(rmm_np[:, 1]) / N
    print(rmm_np[:, 0])
    print(rmm_np[:, 1])
    rmm_np[:, 0] = rmm_np[:, 0] - mean1
    rmm_np[:, 1]= rmm_np[:, 1] - mean2
    print("Means: %s, %s" % (mean1, mean2))
    print(rmm_np, rmm_np.shape)

    window_size = N - M + 1
    X = sliding_window_view(rmm_np, window_size, axis=0)
    X = np.concatenate(X, axis=0)
    XT = X.transpose()

    C = X.dot(XT) / window_size
    eigenvalues, eigenvectors = LA.eig(C)

    def MLU(t):
        if t < M - 1:
            return (t + 1, 0, t+1)
        elif M - 1 <= t < N - M + 1:
            return (M, 0, M)
        elif N - M + 1 <= t < N:
            return (N - t, t - N + M, M)

    def generate_t(phi, w):
        l = []
        for t in range(N):
            M_t, L_t, U_t = MLU(t)
            x = 1 / M_t * sum(phi[t - i] * w[i] for i in range(L_t, U_t))
            l.append(x)
        return np.array(l)

    zs = []
    for i in tqdm(range(eigenvalues.size)):
        v = eigenvectors[i]
        phi = XT.dot(v)
        w = np.reshape(v, (M, 2))
        zs.append(generate_t(phi, w))

    print(zs)
