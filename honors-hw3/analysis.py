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

def generate_figure_1(rmm_np, mode_no=None):
    '''Generate a figure similar to figure 1'''

    start = 365
    end=730

    modecode = "z%s " % mode_no if mode_no else ""

    data = rmm_np[start:end, :]
    rmm1 = data[:, 0]
    rmm2 = data[:, 1]

    plt.plot(rmm1)
    plt.xlabel("Days since Jan 1, 2000")
    plt.ylabel("Centered RMM1")
    plt.title(modecode + "RMM1 between Jan 1, 2000 - Dec 31, 2000")
    plt.show()

    plt.plot(rmm2)
    plt.xlabel("Days since Jan 1, 2000")
    plt.ylabel("Centered RMM2")
    plt.title(modecode + "RMM2 between Jan 1, 2000 - Dec 31, 2000")
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
    plt.title(modecode + "RMM1 PDF")
    plt.show()

    _, bins2 = np.histogram(rmm2, bins=50, range=(rmm2_min, rmm2_max))
    plt.plot(bins2, density1(bins2))
    plt.xlabel("RMM2 value")
    plt.ylabel("Probability density")
    plt.title(modecode + "RMM2 PDF")
    plt.show()

def generate_figure_12(rmm_np, mode_no=None):
    start = 365
    end = 365 + 60

    modecode = "z%s " % mode_no if mode_no else ""

    # Transpose rmm1, rmm2 data into polar
    rmm1 = rmm_np[start:end, 0]
    rmm2 = rmm_np[start:end, 1]
    r = np.sqrt(rmm1**2 + rmm2**2)
    phi = np.arctan2(rmm2, rmm1)

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.plot(phi, r)
    ax.set_rmax(max(r)*1.5)
    ax.grid(True)
    ax.set_title(modecode + "Phase diagrams of RMM1 and RMM2 raw data")
    plt.show()

# Same code from Honors HW2
def trad_ssa(rmm_np, M=51, gen_z=4, until=-1):
    # Computes traditional SSA algorithm on rmm with window size M

    rmm_np = rmm_np[:until]
    N = rmm_np.shape[0]

    print(rmm_np, rmm_np.shape)


    # Compute window size, and construct time-lagged datapoint array
    window_size = N - M + 1
    X = sliding_window_view(rmm_np, window_size, axis=0)
    X = np.concatenate(X, axis=0)
    XT = X.transpose()

    # Create covariance matrix, and find eigenvalues/eigenvectors
    C = X.dot(XT) / window_size
    eigenvalues, eigenvectors = LA.eig(C)

    # Sort eigenvectors in order of eigenvalue magnitude
    eigenorder = eigenvalues.argsort()
    #eigenvectors = np.sqrt(eigenvalues) * eigenvectors
    eigenvectors = eigenvectors[eigenorder[::-1]]

    def MLU(t):
        # Sum count and lower/upper bounds for sum (used for generating z(t))
        if t < M - 1:
            return (t + 1, 0, t + 1)
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
    for i in tqdm(range(min(gen_z, eigenvalues.size))):
        v = eigenvectors[i]
        v = v / LA.norm(v) # Normalize each eigenvector
        phi = XT.dot(v) # Projection of v onto the timelagged data space
        w = np.reshape(v, (M, 2)) # Group v into a sequence of pair vectors (RMM1, RMM2)
        zs.append(generate_t(phi, w))

    return zs

def ssa_cp(rmm_np, M=51, gen_z=4, until=-1):
    #Computes SSA-CP, an improved method for estimating the last M-1 entries of each RC

    rmm_np = rmm_np[:until]
    N = rmm_np.shape[0]

    print(rmm_np, rmm_np.shape)

    # Compute window size, and construct time-lagged datapoint array
    window_size = N - M + 1
    X = sliding_window_view(rmm_np, window_size, axis=0)
    X = np.concatenate(X, axis=0)
    XT = X.transpose()

    # Create covariance matrix, and find eigenvalues/eigenvectors
    C = X.dot(XT) / window_size
    eigenvalues, eigenvectors = LA.eig(C)

    # Sort eigenvectors in order of eigenvalue magnitude
    eigenorder = eigenvalues.argsort()
    #eigenvectors = np.sqrt(eigenvalues) * eigenvectors
    eigenvectors = eigenvectors[eigenorder[::-1]]

    # Calculate the covariance matrices for N + 1 <= k <= N + M - 1
    last_m_1 = rmm_np[N-M+1:]
    tail = []
    for i in range(M - 1):
        xs = X[:2*(i+1), :] # take the first i+1 vectors of each data point
        ys = X[2*(i+1):, :] # last M - (i + 1) vectors of each data point
        c = np.cov(xs, ys) # Covariance matrix
        c11 = c[:2*(i+1), :2*(i+1)] # Variance of xs
        c21 = c[2*(i+1):, :2*(i+1)] # Covariance matrix of ys wrt xs
        y1 = np.concatenate(rmm_np[N-i-1:], axis=0) # Known data
        y2 = c21.dot(np.linalg.inv(c11).dot(y1)) #Unknown data (mu_{2|1} in the paper)
        x = np.concatenate((y1, y2), axis=0) #Combine data to create new data point
        tail.append(x)
    tail.reverse()
    Xaug = np.stack(tail, axis=1) # Additional augmented data
    print(X.shape, Xaug.shape)
    X = np.concatenate((X, Xaug), axis=1) #Combine
    XT = X.transpose()

    def MLU(t):
        # Sum count and lower/upper bounds for sum (used for generating z(t))
        if t < M - 1:
            return (t + 1, 0, t + 1)
        elif M - 1 <= t < N:
            return (M, 0, M)
        elif N <= t < N + M:
            return (N - t + M - 1, t - N + 1, M)

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
    for i in tqdm(range(min(gen_z, eigenvalues.size))):
        v = eigenvectors[i]
        v = v / LA.norm(v) # Normalize each eigenvector
        phi = XT.dot(v) # Projection of v onto the timelagged data space
        w = np.reshape(v, (M, 2)) # Group v into a sequence of pair vectors (RMM1, RMM2)
        zs.append(generate_t(phi, w))

    return zs

def cumul_plot(zs, n=10):
    start = 365
    end = 730

    cumul = 0
    for i in range(min(n, len(zs))):
        cumul = zs[i] + cumul
        rmm1 = cumul[start:end, 0]
        plt.plot(rmm1)

    plt.xlabel("Days since Jan 1, 2000")
    plt.ylabel("Centered RMM1")
    plt.title("Cumulative RC RMM1 between Jan 1, 2000 - Dec 31, 2000")
    plt.show()

def generate_composite_plots(rmm_np, trad_ssa_zs, ssa_cp_zs):
    start = -100
    M = 51
    trad_component_1 = trad_ssa_zs[0] + trad_ssa_zs[1]
    cp_component_1 = ssa_cp_zs[0] + ssa_cp_zs[1]
    trad_component_2 = sum((trad_ssa_zs[i] for i in range(4)))
    cp_component_2 = sum((ssa_cp_zs[i] for i in range(4)))

    rmm_np = rmm_np[start:]
    trad_component_1 = trad_component_1[start:]
    cp_component_1 = cp_component_1[start:]
    trad_component_2 = trad_component_2[start:]
    cp_component_2 = cp_component_2[start:]

    # Plot (a) and (d)
    plt.plot(rmm_np[:, 0], label="Truth")
    plt.plot(trad_component_1[:, 0], label="Trad RC")
    plt.plot(cp_component_1[:, 0], label="SSA-CP")
    plt.legend()
    plt.title("RMM1 (Modes 1-2)")
    plt.show()

    plt.plot(rmm_np[:, 0], label="Truth")
    plt.plot(trad_component_2[:, 0], label="Trad RC")
    plt.plot(cp_component_2[:, 0], label="SSA-CP")
    plt.legend()
    plt.title("RMM1 (Modes 1-4)")
    plt.show()

    # Correlation and RMSE
    N = trad_component_1.shape[0]
    trad_corr_1 = np.array([np.corrcoef(rmm_np[i:i+5, 0], trad_component_1[i:i+5, 0])[0, 1] for i in range(N - 5)])
    cp_corr_1 = np.array([np.corrcoef(rmm_np[i:i+5, 0], cp_component_1[i:i+5, 0])[0, 1] for i in range(N - 5)])
    trad_corr_2 = np.array([np.corrcoef(rmm_np[i:i+5, 0], trad_component_2[i:i+5, 0])[0, 1] for i in range(N - 5)])
    cp_corr_2 = np.array([np.corrcoef(rmm_np[i:i+5, 0], cp_component_2[i:i+5, 0])[0, 1] for i in range(N - 5)])

    def rmse(a, b):
        return np.linalg.norm(a - b) / np.sqrt(len(a))

    trad_rmse_1 = np.array([rmse(rmm_np[i:i+5, 0], trad_component_1[i:i+5, 0]) for i in range(N - 5)])
    cp_rmse_1 = np.array([rmse(rmm_np[i:i+5, 0], cp_component_1[i:i+5, 0]) for i in range(N - 5)])
    trad_rmse_2 = np.array([rmse(rmm_np[i:i+5, 0], trad_component_2[i:i+5, 0]) for i in range(N - 5)])
    cp_rmse_2 = np.array([rmse(rmm_np[i:i+5, 0], cp_component_2[i:i+5, 0]) for i in range(N - 5)])

    # Plot (b), (c), (e), (f)
    plt.plot(trad_corr_1, label="Trad SSA")
    plt.plot(cp_corr_1, label="SSA-CP")
    plt.legend()
    plt.title("Correlation (Modes 1-2)")
    plt.show()

    plt.plot(trad_rmse_1, label="Trad SSA")
    plt.plot(cp_rmse_1, label="SSA-CP")
    plt.legend()
    plt.title("RMSE (Modes 1-2)")
    plt.show()

    plt.plot(trad_corr_2, label="Trad SSA")
    plt.plot(cp_corr_2, label="SSA-CP")
    plt.legend()
    plt.title("Correlation (Modes 1-4)")
    plt.show()

    plt.plot(trad_rmse_2, label="Trad SSA")
    plt.plot(cp_corr_2, label="SSA-CP")
    plt.legend()
    plt.title("RMSE (Modes 1-4)")
    plt.show()

if __name__ == '__main__':
    rmm_np = load("data")
    trad_zs = trad_ssa(rmm_np, gen_z=1000)
    cp_zs = ssa_cp(rmm_np, gen_z=1000)

    generate_composite_plots(rmm_np, trad_zs, cp_zs)

    # Generate plots using raw rmm data
    #generate_figure_1(rmm_np)
    #generate_figure_12(rmm_np)
    '''
    for i, z in enumerate(zs[:4]):
        generate_figure_1(z, mode_no=i+1)
        generate_figure_12(z, mode_no=i+1)
    '''
    '''
    for i, z in enumerate(zs):
        generate_figure_1(z, mode_no=i+1)
        #generate_figure_12(z, mode_no=i+1)
    '''
    #cumul_plot(zs, n=len(zs))
    #cumul_plot(zs, n=5)
