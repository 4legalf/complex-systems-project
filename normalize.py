import os
import numpy as np
import json
import seaborn as sns
import matplotlib.pyplot as plt

real_min = 100
real_max = -100
imag_min = 100
imag_max = -100
freq_min = 100
freq_max = -100
damp_min = 100
damp_max = -100

for file in os.listdir('output/raw/'):
    # read the CSV file with numpy
    data = np.loadtxt(os.path.join('output/raw/', file), delimiter=',')
    
    # get the min and max of the first row
    damp_min = min(damp_min, np.min(data[0]))
    damp_max = max(damp_max, np.max(data[0]))
    
    # get the min and max of the second row
    freq_min = min(freq_min, np.min(data[1]))
    freq_max = max(freq_max, np.max(data[1]))
    
    # get the min and max of the real part of the modes
    real_min = min(real_min, np.min(data[2:34]))
    real_max = max(real_max, np.max(data[2:34]))
    
    # get the min and max of the imaginary part of the modes
    imag_min = min(imag_min, np.min(data[34:]))
    imag_max = max(imag_max, np.max(data[34:]))

# save the min and max values to a json file
with open('min_max.json', 'w') as f:
    json.dump({
        'real_min': real_min,
        'real_max': real_max,
        'imag_min': imag_min,
        'imag_max': imag_max,
        'freq_min': freq_min,
        'freq_max': freq_max,
        'damp_min': damp_min,
        'damp_max': damp_max
    }, f, indent=4)

# Ensure the normalized directory exists
os.makedirs('output/raw_normalized', exist_ok=True)

for file in os.listdir('output/raw/'):
    # read the CSV file with numpy
    data = np.loadtxt(os.path.join('output/raw/', file), delimiter=',')
    
    # normalize the data
    data[0] = (data[0] - damp_min) / (damp_max - damp_min)
    data[1] = (data[1] - freq_min) / (freq_max - freq_min)
    data[2:34] = (data[2:34] - real_min) / (real_max - real_min)
    data[34:] = (data[34:] - imag_min) / (imag_max - imag_min)

    # Create heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(data, cmap='seismic', cbar=False, xticklabels=False, yticklabels=False)

    
    # Save the normalized heatmap figure
    output_filename = f'output/heatmaps_normalized/{os.path.splitext(os.path.basename(file))[0]}.png'
    plt.savefig(output_filename, bbox_inches='tight')
    plt.close()

    # save the normalized data to a new CSV file
    np.savetxt(os.path.join('output/raw_normalized/', file), data, delimiter=',')
