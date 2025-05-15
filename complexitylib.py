from file_processing import edf_to_arr
import numpy as np
from scipy.spatial.distance import pdist, squareform
import os


# Patient Class
class Patient:
    def __init__(self, id, name, DMDbefore = None, DMDafter = None, cluster = None, response = None):
        
        self.id = id
        self.name = name
        self.DMDbefore = DMDbefore
        self.DMDafter = DMDafter
        self.cluster = cluster
        self.response = response
    
    def get_eeg_before(self):
        filepath = f'C:/Users/aless/dev/complex-systems-project/datasets/antidepressant-study/'
        filename = f'{filepath}{self.name}S1EC-edf.edf'
        return (edf_to_arr(filename))
    
    def get_eeg_after(self):
        filepath = f'C:/Users/aless/dev/complex-systems-project/datasets/antidepressant-study/'
        filename = f'{filepath}{self.name}S2EC-edf.edf'
        return (edf_to_arr(filename))


# Embedding Function
def embed_signal(signal, m, tau):
    N = len(signal)
    L = N - (m - 1) * tau
    if L <= 0:
        raise ValueError(f"Signal too short: N={N}, m={m}, tau={tau}")
    X = np.empty((L, m))
    for k in range(m):
        X[:, k] = signal[k * tau : k * tau + L]
    return X

# Correlation Sum
def correlation_sum(X, r):
    D = squareform(pdist(X))
    np.fill_diagonal(D, np.inf)
    return np.mean(D < r)

# Generate Radius Values
def make_r_vals(X, n_r=20, r_min=0.1, r_max=2.0):
    sigma = np.std(X)
    return np.linspace(r_min * sigma, r_max * sigma, n_r)

# Correlation Dimension (Single Channel)
def correlation_dimension(signal, m=10, tau=1, n_r=20):
    X = embed_signal(signal, m, tau)
    r_vals = make_r_vals(X, n_r=n_r)
    C = np.array([correlation_sum(X, r) for r in r_vals])
    log_r = np.log(r_vals)
    log_C = np.log(C + 1e-12)
    slope, _ = np.polyfit(log_r[2:-2], log_C[2:-2], 1)
    return slope

# Per Subject (All Channels)
def compute_subject_dims(eeg_subject, m=10, tau=1, n_r=20):
    return np.array([
        correlation_dimension(eeg_subject[ch], m=m, tau=tau, n_r=n_r)
        for ch in range(eeg_subject.shape[0])
    ])  # shape: (n_channels,)