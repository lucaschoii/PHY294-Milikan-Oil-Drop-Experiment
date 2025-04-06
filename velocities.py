import os
import pandas as pd
import numpy as np
from scipy.stats import linregress
import matplotlib.pyplot as plt
import scipy.odr as odr

# Configuration
data_path = '/Users/lucaschoi/Documents/GitHub/PHY294-Milikan-Oil-Drop-Experiment/data'
annotations_file = '/Users/lucaschoi/Documents/GitHub/PHY294-Milikan-Oil-Drop-Experiment/annotations.tsv'
velocities_file = '/Users/lucaschoi/Documents/GitHub/PHY294-Milikan-Oil-Drop-Experiment/velocities.tsv'

# Load annotations
annotations = pd.read_csv(annotations_file, sep='\t')
results = []

# Conversions
PX_PER_MM = 520 
PX_PER_MM_UNC = 1
MM_PER_M = 1000
M_PER_PIXEL = 1 / (PX_PER_MM * MM_PER_M) 
M_PER_PIXEL_UNC = PX_PER_MM_UNC / ((PX_PER_MM * MM_PER_M) ** 2)
PX_UNC = 0.5

FRAME_RATE = 10
FRAME_RATE_UNC = 0.1


for _, row in annotations.iterrows():
    
    try:
        filename = row['filename']  
        filepath = os.path.join(data_path, filename)
        stopping_voltage = filename.split('_')[0]
        rising_voltage = filename.split('_')[1]
        print(f"Processing: {filename}")

        df = pd.read_csv(filepath, sep='\t', na_values='#NV')
        df['y-position'] = pd.to_numeric(df['y-position'], errors='coerce')


        
        # Calculate velocities
        def get_data(start_frame, end_frame):
            """Calculate slope (velocity) between two frames"""
            y_subset = list(df.iloc[start_frame:end_frame+1]['y-position'] * M_PER_PIXEL) 
            x_subset = [i / FRAME_RATE for i in range(start_frame, end_frame + 1)]
            dx = [FRAME_RATE_UNC / FRAME_RATE**2 for _ in range(len(x_subset))]
            dy = [y * np.sqrt((PX_UNC / PX_PER_MM) ** 2 + (M_PER_PIXEL_UNC / M_PER_PIXEL) ** 2) for y in y_subset]

            x_subset = np.array(x_subset)
            y_subset = np.array(y_subset)
            dx = np.array(dx)
            dy = np.array(dy)

            mask = ~np.isnan(x_subset) & ~np.isnan(y_subset)
            x_subset = x_subset[mask]
            y_subset = y_subset[mask]
            dx = dx[mask]
            dy = dy[mask]

            
            if len(y_subset) < 2:
                return np.nan, np.nan, np.nan, np.nan  # Not enough points
            
            # plt.plot(x_subset, y_subset, label=f"{filename} ({start_frame}-{end_frame})")
            # plt.errorbar(x_subset, y_subset, xerr=dx, yerr=dy, fmt='o', label=f"{filename} ({start_frame}-{end_frame})")
            # plt.show()

            return x_subset, y_subset, dx, dy  # Return x and y data for regression
        
        def linear_func(B, x):
            return B[0] * x + B[1]
        
        def perform_regression(x, y, dx, dy):
            """Performs orthogonal distance regression (ODR) with uncertainties in both x and y."""
            # Define model
            linear_model = odr.Model(linear_func)

            # Data with uncertainties
            data = odr.RealData(x, y, sx=dx, sy=dy)

            # Set up ODR solver
            odr_solver = odr.ODR(data, linear_model, beta0=[1, 0])  # Initial guess: slope=1, intercept=0
            output = odr_solver.run()

            # Extract parameters
            slope, intercept = output.beta
            slope_unc, intercept_unc = output.sd_beta  # Uncertainties

            # Compute residuals
            y_fit = slope * np.array(x) + intercept
            residuals = (np.array(y) - y_fit) / np.array(dy)

            # Compute R²
            ss_total = np.sum((np.array(y) - np.mean(y))**2)
            ss_residuals = np.sum((np.array(y) - y_fit)**2)
            r_squared = 1 - (ss_residuals / ss_total)

            # Compute χ² and reduced χ²
            chi_squared = np.sum(residuals**2)
            dof = len(x) - 2  # Degrees of freedom (N - k)
            reduced_chi_squared = chi_squared / dof if dof > 0 else float('inf')

            
            if not (slope and slope_unc and intercept and intercept_unc and r_squared and reduced_chi_squared):
                print(f'Regression failed for {filename}')
                # Plot the fit and data
                plt.figure(figsize=(6, 4))
                plt.errorbar(x, y, xerr=dx, yerr=dy, fmt='bo', label='Data')
                plt.plot(x, y_fit, 'r-', label='Fit')
                plt.xlabel("Time (s)")
                plt.ylabel("Position (m)")
                plt.legend()
                plt.title(f'Velocity = {round(slope, 7)} ± {round(slope_unc, 3)} m/s')
                plt.show()



            return {
                "slope": slope,
                "slope_unc": slope_unc,
                "intercept": intercept,
                "intercept_unc": intercept_unc,
                "r_squared": r_squared,
                "reduced_chi_squared": reduced_chi_squared,
            }
        
        
        def calculate_velocity(start_frame, end_frame):
            """Calculate velocity and its uncertainty."""
            x_subset, y_subset, dx, dy = get_data(start_frame, end_frame)
            results = perform_regression(x_subset, y_subset, dx, dy)

            return results

        

        # Rising phase
        rise = calculate_velocity(
            int(row['rise_start']),
            int(row['rise_end']))
        
        # Falling phase
        fall = calculate_velocity(
            int(row['fall_start']),
            int(row['fall_end']))
        
        results.append({
            'filename': filename,
            'rising_voltage': rising_voltage,
            'stopping_voltage': stopping_voltage,
            'v_rise': rise['slope'],
            'v_rise_unc': rise['slope_unc'],
            'v_fall': fall['slope'],
            'v_fall_unc': fall['slope_unc'],
            'r_squared_rise': rise['r_squared'],
            'r_squared_fall': fall['r_squared'],
            'reduced_chi_squared_rise': rise['reduced_chi_squared'],
            'reduced_chi_squared_fall': fall['reduced_chi_squared'],
            'rise_intercept': rise['intercept'],
            'rise_intercept_unc': rise['intercept_unc'],
            'fall_intercept': fall['intercept'],
            'fall_intercept_unc': fall['intercept_unc'],
        })
        
    except Exception as e:
        print(f"Error processing {filename}: {str(e)}")
        continue

# Save results
if results:
    results_df = pd.DataFrame(results)
    
    # Save to CSV
    results_df.to_csv(velocities_file, index=False, sep='\t')
    print(f"Saved velocity calculations for {len(results_df)} droplets to {velocities_file}")
else:
    print("No velocity calculations were completed.")