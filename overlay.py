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
MM_PER_M = 1000
M_PER_PIXEL = 1 / (PX_PER_MM * MM_PER_M) 

FRAME_RATE = 10

def linear(x, m, b):
    return m * x + b

for file in files:
    filename = os.path.basename(file)
    print(f"Showing: {filename}")
    data = pd.read_csv(file, sep='\t', na_values='#NV')
    x = [i / FRAME_RATE for i in range(len(data))]
    velocity_data = velocities_df[velocities_df['filename'] == filename]



    v_rise = velocity_data['v_rise'].values[0]
    rise_intercept = velocity_data['rise_intercept'].values[0]
    v_fall = velocity_data['v_fall'].values[0]
    fall_intercept = velocity_data['fall_intercept'].values[0]

    fall_start = annotations_df[annotations_df['filename'] == filename]['fall_start'].values[0]
    fall_end = annotations_df[annotations_df['filename'] == filename]['fall_end'].values[0]
    rise_start = annotations_df[annotations_df['filename'] == filename]['rise_start'].values[0]
    rise_end = annotations_df[annotations_df['filename'] == filename]['rise_end'].values[0]

    # Plot the data with the velocity lines
    rise_x = np.linspace(rise_start / FRAME_RATE, stop=rise_end / FRAME_RATE)
    rise_y = linear(rise_x, v_rise, rise_intercept)
    fall_x = np.linspace(fall_start / FRAME_RATE, stop=fall_end / FRAME_RATE)
    fall_y = linear(fall_x, v_fall, fall_intercept)
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(x, data['y-position'] * M_PER_PIXEL, label=os.path.basename(file))
    ax.plot(rise_x, rise_y, label='Rising Velocity', color='green', linewidth=2)
    ax.plot(fall_x, fall_y, label='Falling Velocity', color='red', linewidth=2)
    ax.set_title(f'{filename}')
    plt.show()

    





