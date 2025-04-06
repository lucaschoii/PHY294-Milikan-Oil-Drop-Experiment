import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import glob

velocities = '/Users/lucaschoi/Documents/GitHub/PHY294-Milikan-Oil-Drop-Experiment/velocities.tsv'
velocities_df = pd.read_csv(velocities, sep='\t')
annotations = '/Users/lucaschoi/Documents/GitHub/PHY294-Milikan-Oil-Drop-Experiment/annotations.tsv'
annotations_df = pd.read_csv(annotations, sep='\t')

data_path = '/Users/lucaschoi/Documents/GitHub/PHY294-Milikan-Oil-Drop-Experiment/data'
files = glob.glob(os.path.join(data_path, '*.tsv'))

# Conversions
PX_PER_MM = 520 
PX_PER_MM_UNC = 1
MM_PER_M = 1000
M_PER_PIXEL = 1 / (PX_PER_MM * MM_PER_M) 
M_PER_PIXEL_UNC = PX_PER_MM_UNC / ((PX_PER_MM * MM_PER_M) ** 2)
PX_UNC = 0.5

FRAME_RATE = 10
FRAME_RATE_UNC = 0.1


def linear(x, m, b):
    return m * x + b

for file in files:
    filename = os.path.basename(file)
    print(f"Showing: {filename}")
    data = pd.read_csv(file, sep='\t', na_values='#NV')

    x = [i / FRAME_RATE for i in range(len(data))]
    velocity_data = velocities_df[velocities_df['filename'] == filename]

    # Extract velocities and intercepts
    v_rise = velocity_data['v_rise'].values[0]
    rise_intercept = velocity_data['rise_intercept'].values[0]
    v_fall = velocity_data['v_fall'].values[0]
    fall_intercept = velocity_data['fall_intercept'].values[0]

    # Extract start and end frames for rise and fall
    fall_start = annotations_df[annotations_df['filename'] == filename]['fall_start'].values[0]
    fall_end = annotations_df[annotations_df['filename'] == filename]['fall_end'].values[0]
    rise_start = annotations_df[annotations_df['filename'] == filename]['rise_start'].values[0]
    rise_end = annotations_df[annotations_df['filename'] == filename]['rise_end'].values[0]

    # Calculate uncertainty in y-position (M_PER_PIXEL_UNC is the uncertainty factor in meters per pixel)
    y_position_unc = data['y-position'].apply(lambda y: y * M_PER_PIXEL * np.sqrt((PX_UNC / PX_PER_MM) ** 2 + (M_PER_PIXEL_UNC / M_PER_PIXEL) ** 2))

    # Plot the data with uncertainties and the velocity lines
    rise_x = np.linspace(rise_start / FRAME_RATE, stop=rise_end / FRAME_RATE)
    rise_y = linear(rise_x, v_rise, rise_intercept)

    fall_x = np.linspace(fall_start / FRAME_RATE, stop=fall_end / FRAME_RATE)
    fall_y = linear(fall_x, v_fall, fall_intercept)

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.scatter(x, data['y-position'] * M_PER_PIXEL, label=os.path.basename(file), s=0.2)
    ax.plot(rise_x, rise_y, label='Rising Velocity', color='green', linewidth=4)
    ax.plot(fall_x, fall_y, label='Falling Velocity', color='red', linewidth=4)

    # Plotting uncertainties
    ax.errorbar(x, data['y-position'] * M_PER_PIXEL, yerr=y_position_unc, fmt='o', color='blue', label='Position Uncertainty', markersize=3)

    ax.set_xlabel('Time (s)', fontsize=16)
    ax.set_ylabel('Y-Position (m)', fontsize=16)
    ax.set_title(f'Fits for {filename} Position Vs Time', fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.legend()

    plt.show()

    





