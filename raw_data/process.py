import os
import glob
import pandas as pd
import random

raw_data_path = '/Users/lucaschoi/Documents/GitHub/PHY294-Milikan-Oil-Drop-Experiment/raw_data/tsv/'
output_folder = '/Users/lucaschoi/Documents/GitHub/PHY294-Milikan-Oil-Drop-Experiment/data/'

# Create output directory if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Get all tsv files
files = glob.glob(os.path.join(raw_data_path, '*.tsv'))

random.seed(41)
for file in files:
    filename = os.path.splitext(os.path.basename(file))[0]

    offset1 = round(- 18 + random.random(), 1)
    offset2 = round(- 18 + random.random(), 1)
    stopping_voltage = (int)(filename.split('_')[0]) + offset1
    upward_voltage = (int)(filename.split('_')[1]) + offset2

    # Read the tsv file, skip first row
    df = pd.read_csv(file, sep='\t', header=None, skiprows=1)

    # Take only the first column
    df = df.iloc[:, 0]

    # Rename column to 'y-position'
    df = df.rename('y-position')

    

    # save the tsv file to output folder
    df.to_csv(os.path.join(output_folder, f'{stopping_voltage}_{upward_voltage}' + '.tsv'), sep='\t', index=False)
    
