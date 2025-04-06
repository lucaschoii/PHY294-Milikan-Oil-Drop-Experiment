# output velocities in a table format for latex

import pandas as pd
import os
import glob
import numpy as np

# Configuration
velocoties_file = '/Users/lucaschoi/Documents/GitHub/PHY294-Milikan-Oil-Drop-Experiment/velocities.tsv'
velocities_df = pd.read_csv(velocoties_file, sep='\t')

# sort by stopping voltage
velocities_df = velocities_df.sort_values(by='stopping_voltage')

for i, row in velocities_df.iterrows():
    filename = row['filename']
    stopping_voltage = row['stopping_voltage']

    v_fall = row['v_fall']
    v_fall_unc = row['v_fall_unc']
    r_squared = row['r_squared_rise']
    reduced_chi_squared = row['reduced_chi_squared_rise']

    print(f"{filename.split('_')[0]}\_{filename.split('_')[1]} & ${stopping_voltage} \pm 0.05$  & ${v_fall:.7f} \pm {v_fall_unc:.7f}$ & {r_squared:.4f} & {reduced_chi_squared:.4f} \\\\")