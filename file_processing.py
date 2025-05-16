# EEG Data Processing Utilities for Antidepressant Study
'''
This module provides functions for reading, preprocessing, and decomposing EEG data
for scientific analysis, including DMD-based feature extraction and heatmap generation.
'''

import pyedflib
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.signal import detrend, butter, filtfilt
from sklearn.preprocessing import StandardScaler
from pydmd import DMD

# --- Paths ---
data_path = 'datasets/antidepressant-study'  # Path to raw EEG data
output_arr = 'output/raw'                    # Output directory for raw arrays
output_heats = 'output/heatmaps'             # Output directory for heatmaps

# --- Utility: Check if EDF file is readable ---
def check_file(file_path):
    """
    Check if an EDF file can be opened by pyedflib.
    Prints an error message if the file cannot be read.
    """
    try:
        pyedflib.EdfReader(file_path)
    except Exception as e:
        print(f"Error reading EDF file: {file_path}, {e}")

# --- EDF to Array ---
def edf_to_arr(edf_path):
    """
    Load EEG data from an EDF file and return the first 32 channels as a NumPy array.
    Returns:
        sigbufs: np.ndarray (channels x timepoints)
        signal_labels: list of channel names
    """
    f = pyedflib.EdfReader(edf_path)
    n = f.signals_in_file
    signal_labels = f.getSignalLabels()
    sigbufs = np.zeros((n, f.getNSamples()[0]))
    for i in np.arange(n):
        sigbufs[i, :] = f.readSignal(i)
    sigbufs = sigbufs[:32, :]
    signal_labels = signal_labels[:32]
    f.close()
    return sigbufs, signal_labels

# --- EEG Preprocessing ---
def preprocessing(eeg_data, lowcut=4, highcut=45, samplingrate=250, order=4):
    """
    Preprocess EEG data: detrend, bandpass filter, and z-score normalization.
    Args:
        eeg_data: np.ndarray (channels x timepoints)
        lowcut, highcut: bandpass filter frequencies (Hz)
        samplingrate: sampling rate (Hz)
        order: filter order
    Returns:
        eeg_normalized: np.ndarray (channels x timepoints)
    """
    eeg_detrended = detrend(eeg_data, axis=-1, type='linear')
    b, a = butter(order, [lowcut / (0.5 * samplingrate), highcut / (0.5 * samplingrate)], btype='band')
    eeg_bandfiltered = filtfilt(b, a, eeg_detrended, axis=-1)
    eeg_normalized = StandardScaler().fit_transform(eeg_bandfiltered.T).T
    return eeg_normalized

# --- Dynamic Mode Decomposition (DMD) ---
def dmd_decomposition(signal, n_modes=10):
    """
    Perform Dynamic Mode Decomposition (DMD) on EEG data.
    Args:
        signal: np.ndarray (channels x timepoints)
        n_modes: number of DMD modes to keep
    Returns:
        dmd_modes: spatial modes (channels x modes)
        frequencies: mode frequencies (cycles/sample)
        damping_ratios: normalized damping ratios
    """
    dmd = DMD(svd_rank=n_modes)
    dmd.fit(signal)
    eigenvalues = dmd.eigs
    dmd_modes = dmd.modes
    dmd_amplitudes = dmd.amplitudes
    dmd_dynamics = dmd.dynamics
    frequencies = np.abs(np.angle(eigenvalues)) / (2 * np.pi)
    damping_ratios = -np.real(eigenvalues) / np.sqrt(np.real(eigenvalues)**2 + np.imag(eigenvalues)**2)
    return (dmd_modes, frequencies, damping_ratios)

# --- Heatmap Generation for DMD Features ---
def create_heatmap_decomposition(decomposition, filename):
    """
    Create and save a heatmap (PNG, CSV) of DMD features: damping ratios, frequencies, real/imaginary parts of modes.
    Args:
        decomposition: tuple (dmd_modes, frequencies, damping_ratios)
        filename: output file base name (no extension)
    """
    dmd_modes, frequencies, damping_ratios = decomposition

    def normalize(x):
        x_norm = (x - np.min(x)) / (np.max(x) - np.min(x))
        return x_norm

    Phi_phys_unique_real = np.real(dmd_modes)
    Phi_phys_unique_imag = np.imag(dmd_modes)
    Phi_phys_unique_norm_real = normalize(Phi_phys_unique_real)
    Phi_phys_unique_norm_imag = normalize(Phi_phys_unique_imag)
    zeta_map = normalize(damping_ratios)
    fn_map = normalize(frequencies)

    # Stack all features for heatmap
    heatmap_data = np.vstack([
        zeta_map,
        fn_map,
        Phi_phys_unique_norm_real,
        Phi_phys_unique_norm_imag
    ])

    # Plot and save heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(heatmap_data, cmap='seismic', cbar=False, xticklabels=False, yticklabels=False)
    output_filename = f'output/heatmaps/{filename}.png'
    plt.savefig(output_filename, bbox_inches='tight')
    plt.close()

    # Save heatmap data to CSV
    csv_output_filename = f'output/raw/{filename}.csv'
    np.savetxt(csv_output_filename, heatmap_data, delimiter=",")

# --- Batch Processing ---
if __name__ == "__main__":
    # Ensure output directories exist
    os.makedirs(output_arr, exist_ok=True)
    os.makedirs(output_heats, exist_ok=True)
    for file in os.listdir(data_path):
        if file.endswith('.edf'):
            edf_path = os.path.join(data_path, file)
            try:
                pyedflib.EdfReader(edf_path)
            except Exception as e:
                print(f"Error reading EDF file: {edf_path}, {e}")
            else:
                signal, labels = edf_to_arr(edf_path)
                signal = preprocessing(signal)
                decomposition = dmd_decomposition(signal)
                create_heatmap_decomposition(decomposition, os.path.splitext(file)[0])