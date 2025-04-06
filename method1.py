import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt


rho_oil = 875.3
rho_air = 1.204
g = 9.80
nu = 1.827e-5
d = 6.0e-3
precision = 19

#  equation 5
# q = C1 * (v_d) ** (3/2) / V_stop

velocities_file = '/Users/lucaschoi/Documents/GitHub/PHY294-Milikan-Oil-Drop-Experiment/velocities.tsv'
method1_results_file = '/Users/lucaschoi/Documents/GitHub/PHY294-Milikan-Oil-Drop-Experiment/method1.py'

charges = []
charge_uncs = []
V_unc = 0.3

# read velocities file
velocities_df = pd.read_csv(velocities_file, sep='\t')

for _, row in velocities_df.iterrows():
    try:
        V_stop = row['stopping_voltage']
        v_d = row['v_fall']
        v_d_unc = row['v_fall_unc']

        q = v_d ** (3/2) / V_stop

        # Calculate the uncertainty in charge
        partial_q_vd = (3 / 2) * (v_d ** (1/2)) / V_stop
        partial_q_V_stop = -(v_d ** (3/2)) / (V_stop ** 2)
        q_unc = np.sqrt((partial_q_vd * v_d_unc) ** 2 + (partial_q_V_stop * V_unc) ** 2)
        charges.append(q)
        charge_uncs.append(q_unc)
        # print(f'q = {q:.10} \pm {q_unc:.10} C')

    except Exception as e:
        print(f"{e}")



def estimate_elementary_charge(q_list, e_min=1e-21, e_max=2e-8, num_steps=1000000):
    """
    Estimate the elementary charge by trying possible divisors of q_list.
    
    Args:
        q_list (array-like): Measured charges in Coulombs
        e_min, e_max: Range to scan for candidate e
        num_steps: Number of candidate e values to test
    
    Returns:
        best_e: Estimated elementary charge
        scores: List of scores for all candidate e values
    """
    q_array = np.array(q_list)
    candidate_es = np.linspace(e_min, e_max, num_steps)
    best_score = float('inf')
    best_e = None
    scores = []

    for e in candidate_es:

        # Divide all q values by e and check how close to nearest integer
        multiples = q_array / e
        residuals = np.abs(multiples - np.round(multiples))
        score = np.max(residuals) 
        scores.append(score)
        
        if score < best_score:
            best_score = score
            best_e = e

    return best_e, scores, candidate_es

# best_e, scores, candidate_es = estimate_elementary_charge(charges)
# print(f"Estimated elementary charge: {best_e:.10} C")
# print(f"Best score: {np.min(scores)}")

# plt.plot(candidate_es, scores)
# plt.xlabel("Candidate e (C)")
# plt.show()

# charges = sorted(charges)
# print(charges)

# plot a histogram of the charges
trimmed_charges = np.array([charge for charge in charges if charge < 2.0e-9])
q_values = np.array(trimmed_charges)  

bin_size = 5e-12  # 0.5 nC
bins = np.arange(trimmed_charges.min(), trimmed_charges.max() + bin_size, bin_size)

# Plot histogram
plt.hist(q_values, bins=bins, edgecolor="black", alpha=0.7)
plt.xlabel("Charge (C)")
plt.ylabel("Frequency")
plt.title("Histogram of Measured Charge Values")
plt.grid(True)
plt.show()


# # compute differences between charges
# charges = sorted(charges)
# charge_diffs = []
# for i in range(len(charges) - 1):
#     charge_diffs.append(charges[i + 1] - charges[1])

# print(f"charge_diffs = {charge_diffs}")


