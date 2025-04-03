import pandas as pd
import numpy as np
import math


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

    except Exception as e:
        print(f"{e}")

scaled = [int(round(num * 10**precision)) for num in charges]
gcd = np.gcd.reduce(scaled)

print(f"gcd: {gcd / 10**precision} \pm {min(charge_uncs)}")
