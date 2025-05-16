# EEG Complexity Analysis Utilities for Antidepressant Study
'''
This module provides functions and classes for EEG complexity analysis,
including correlation dimension computation and patient data handling.
'''

from file_processing import edf_to_arr
import numpy as np
from scipy.spatial.distance import pdist, squareform

# --- Patient Data Class ---
class Patient:
    """
    Patient class for handling EEG file access and metadata.
    Attributes:
        id: Unique patient identifier
        name: Patient code (matches file prefix)
        DMDbefore, DMDafter: Optional DMD results
        cluster: Cluster assignment
        response: Treatment response label
    """
    def __init__(self, id, name, DMDbefore=None, DMDafter=None, cluster=None, response=None):
        self.id = id
        self.name = name
        self.DMDbefore = DMDbefore
        self.DMDafter = DMDafter
        self.cluster = cluster
        self.response = response

    def get_eeg_before(self):
        """
        Load pre-treatment EEG for this patient (first session).
        Returns:
            signal: np.ndarray (channels x timepoints)
            labels: list of channel names
        """
        filepath = 'C:/Users/aless/dev/complex-systems-project/datasets/antidepressant-study/'
        filename = f'{filepath}{self.name}S1EC-edf.edf'
        return edf_to_arr(filename)

    def get_eeg_after(self):
        """
        Load post-treatment EEG for this patient (second session).
        Returns:
            signal: np.ndarray (channels x timepoints)
            labels: list of channel names
        """
        filepath = 'C:/Users/aless/dev/complex-systems-project/datasets/antidepressant-study/'
        filename = f'{filepath}{self.name}S2EC-edf.edf'
        return edf_to_arr(filename)

# --- Time-Delay Embedding ---
def embed_signal(signal, m, tau):
    """
    Construct a time-delay embedding matrix from a 1D signal.
    Args:
        signal: 1D np.ndarray
        m: embedding dimension
        tau: time delay
    Returns:
        X: np.ndarray (L x m), where L = N - (m-1)*tau
    """
    N = len(signal)
    L = N - (m - 1) * tau
    if L <= 0:
        raise ValueError(f"Signal too short: N={N}, m={m}, tau={tau}")
    X = np.empty((L, m))
    for k in range(m):
        X[:, k] = signal[k * tau : k * tau + L]
    return X

# --- Correlation Sum Calculation ---
def correlation_sum(X, r):
    """
    Compute the correlation sum for a set of embedded vectors X at radius r.
    Args:
        X: np.ndarray (L x m)
        r: radius threshold
    Returns:
        float: fraction of pairs with distance < r
    """
    D = squareform(pdist(X))
    np.fill_diagonal(D, np.inf)
    return np.mean(D < r)

# --- Generate Radius Values for Correlation Sum ---
def make_r_vals(X, n_r=20, r_min=0.1, r_max=2.0):
    """
    Generate a range of radius values for correlation sum calculation.
    Args:
        X: embedded matrix (for std calculation)
        n_r: number of radius values
        r_min, r_max: min/max as multiples of std(X)
    Returns:
        np.ndarray of radius values
    """
    sigma = np.std(X)
    return np.linspace(r_min * sigma, r_max * sigma, n_r)

# --- Correlation Dimension (Single Channel) ---
def correlation_dimension(signal, m=10, tau=1, n_r=20):
    """
    Estimate the correlation dimension of a 1D signal using the Grassberger-Procaccia method.
    Args:
        signal: 1D np.ndarray
        m: embedding dimension
        tau: time delay
        n_r: number of radius values
    Returns:
        float: estimated correlation dimension (slope of log-log plot)
    """
    X = embed_signal(signal, m, tau)
    r_vals = make_r_vals(X, n_r=n_r)
    C = np.array([correlation_sum(X, r) for r in r_vals])
    log_r = np.log(r_vals)
    log_C = np.log(C + 1e-12)
    slope, _ = np.polyfit(log_r[2:-2], log_C[2:-2], 1)
    return slope

# --- Correlation Dimension for All Channels (Per Subject) ---
def compute_subject_dims(eeg_subject, m=10, tau=1, n_r=20):
    """
    Compute the correlation dimension for each channel of a subject's EEG data.
    Args:
        eeg_subject: np.ndarray (channels x timepoints)
        m, tau, n_r: see above
    Returns:
        np.ndarray of correlation dimensions (length = n_channels)
    """
    return np.array([
        correlation_dimension(eeg_subject[ch], m=m, tau=tau, n_r=n_r)
        for ch in range(eeg_subject.shape[0])
    ])  # shape: (n_channels,)