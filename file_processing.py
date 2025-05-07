import pyedflib
import numpy as np
import pandas as pd
import mne
from pydmd import DMD
import os
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.signal import detrend
from sklearn.preprocessing import StandardScaler
from scipy.signal import butter, filtfilt

data_path = 'datasets/antidepressant-study'
output_arr = 'output/raw'
output_heats = 'output/heatmaps'

def check_file(file_path):
    edf_path = file_path
    try: pyedflib.EdfReader(edf_path)
    except: print(f"Error reading EDF file: {file_path}")

    
def edf_to_arr(edf_path):
    f = pyedflib.EdfReader(edf_path)
    n = f.signals_in_file
    signal_labels = f.getSignalLabels()
    sigbufs = np.zeros((n, f.getNSamples()[0]))
    for i in np.arange(n):
        sigbufs[i, :] = f.readSignal(i)
    sigbufs = sigbufs[:-3,:]
    signal_labels = signal_labels[:-3]

    return sigbufs, signal_labels

def preprocessing(eeg_data, lowcut=4, highcut=45, samplingrate= 250, order=4, ):
  #detrend first
  eeg_detrended = detrend(eeg_data, axis=-1, type='linear')
  #bandpass filter as griffith did
  b, a = butter(order, [lowcut / (0.5 * samplingrate), highcut / (0.5 * samplingrate)], btype='band')
  eeg_bandfiltered = filtfilt(b, a, eeg_detrended, axis=-1)
  #standardize
  eeg_normalized = StandardScaler().fit_transform(eeg_bandfiltered.T).T
  return eeg_normalized

def dmd_decomposition(signal, n_modes=10):

    eeg_data = signal

    # Step 2: Apply DMD (you can transpose depending on dimensionality preference)
    dmd = DMD(svd_rank=n_modes)  # svd_rank = number of modes to keep
    dmd.fit(eeg_data)

    # Step 3: Output details
    eigenvalues = dmd.eigs         # Complex frequencies of modes
    dmd_modes = dmd.modes          # Each column = spatial mode (shape: channels x modes)
    dmd_amplitudes = dmd.amplitudes #Initial weights (importance) of each mode
    dmd_dynamics = dmd.dynamics    # Temporal evolution of each mode

    # Step 4: Visualize frequency spectrum (from eigenvalues)
    frequencies = np.abs(np.angle(eigenvalues)) / (2 * np.pi)  # In cycles/sample
    damping_ratios = -np.real(dmd.eigs) / np.sqrt(np.real(dmd.eigs)**2 + np.imag(dmd.eigs)**2)

    return (dmd_modes, frequencies, damping_ratios)

def create_heatmap_decomposition(decomposition, filename):
    dmd_modes, frequencies, damping_ratios = decomposition

    Phi_phys_unique_norm_real = np.real(dmd_modes)  # Real part of normalized DMD modes
    Phi_phys_unique_norm_imag = np.imag(dmd_modes)  # Imaginary part of normalized DMD modes
    zeta_map = damping_ratios  # Normalized damping ratios
    fn_map = frequencies    # Normalized frequencies

    # Combine data for heatmap
    heatmap_data = np.vstack([
        zeta_map,
        fn_map,
        Phi_phys_unique_norm_real,
        Phi_phys_unique_norm_imag
    ])

    # Create heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(heatmap_data, cmap='seismic', cbar=False, xticklabels=False, yticklabels=False)

    # Save the figure
    output_filename = f'output/heatmaps/{filename}.png'
    plt.savefig(output_filename, bbox_inches='tight')
    plt.close()

    # Save heatmap data to a CSV file
    csv_output_filename = f'output/raw/{filename}.csv'
    np.savetxt(csv_output_filename, heatmap_data, delimiter=",")

if __name__ == "__main__":

    os.makedirs(output_arr, exist_ok=True)
    os.makedirs(output_heats, exist_ok=True)
    for file in os.listdir(data_path):
        if file.endswith('.edf'):
            edf_path = os.path.join(data_path, file) 
            try: pyedflib.EdfReader(edf_path)
            except: print(f"Error reading EDF file: {edf_path}")
            else:
                signal, labels = edf_to_arr(edf_path)   # Read the EDF file and convert to array
                signal = preprocessing(signal)          # Preprocess the signal
                # Decompose the signal using DMD and save the resulting heatmap to .png and .csv files
                decomposition = dmd_decomposition(signal)
                create_heatmap_decomposition(decomposition, os.path.splitext(file)[0])
                print(f"Processed {file} and saved heatmap and CSV files.")
            