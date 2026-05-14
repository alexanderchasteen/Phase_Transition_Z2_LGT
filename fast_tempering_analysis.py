import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv

# --- Configuration ---
Spatial_Size = 5
Temporal_Size = Spatial_Size
Nplaq = Spatial_Size**3 * Temporal_Size * 6
folder = './Multi_Canon_Results/'
blocks = 50

# Load Data
df = pd.read_csv('/home/alexa/Z2_Lattice_Gauge_Theory/Z2_CPP_NEW/Raw_MC_Data_5^3_5_GOAT.csv')
beta_array = df.columns.to_numpy(dtype=float)
CONFIGS = len(beta_array)
beta_nbhd_radius = (beta_array[1] - beta_array[0]) / 2

# -------------------
# Optimized Weights Lookup
# -------------------
class FastWeights:
    def __init__(self, filename):
        data = np.loadtxt(filename)
        self.plaq_bins = data[:, 0]
        self.weight_vals = data[:, 1]

        self.min_bin = self.plaq_bins[0]
        self.bin_size = self.plaq_bins[1] - self.plaq_bins[0]
        self.n_bins = len(self.weight_vals)

    def find_bin_vectorized(self, values):
        bins = np.floor((values - self.min_bin) / self.bin_size).astype(int)
        bins = np.clip(bins, 0, self.n_bins - 1)
        return bins

    def get_weights_vectorized(self, values):
        indices = self.find_bin_vectorized(values)
        return self.weight_vals[indices]

weights = FastWeights("/home/alexa/Z2_Lattice_Gauge_Theory/Z2_CPP_NEW/weight_function_normalized.txt")
# weights = FastWeights("/home/alexa/Z2_Lattice_Gauge_Theory/Z2_CPP_NEW/No_weight.txt")


print("\n=== PYTHON BIN CHECK ===")
for i in range(5):
    print(f"edge[{i}] = {weights.plaq_bins[i]} | weight[{i}] = {weights.weight_vals[i]}")
print("last edge =", weights.plaq_bins[-1])
print("sizes:", len(weights.plaq_bins), len(weights.weight_vals))
# -------------------
# Storage Lists (Same as before)
# -------------------
avg_plaq = []
SE_avg_plaq = []
plaq_sus = []
SE_plaq_sus = []

svendsen_beta_array = []
svendsen_avg_plaq_array = []
svendsen_avg_plaq_SE_array = []
svendsen_sus_array = []
svendsen_sus_SE_array = []
svendsen_binder = []
svendsen_binder_SE_array = []

sus_jackknife_samples = []
binder_jackknife_samples = []

# -------------------
# Main Loop
# -------------------
for i in range(CONFIGS):
    print(f"Processing Beta index {i}: {beta_array[i]}")
    array = df.iloc[:, i].to_numpy()
    
    # Pre-calculate powers once per beta to save time
    p2_array = array**2
    p4_array = array**4
    
    # Pre-fetch unbiasing weights for the whole config ensemble
    bias_unundo = weights.get_weights_vectorized(array)
    
    k_array = np.linspace(-beta_nbhd_radius, beta_nbhd_radius, 1001)
    
    for k in k_array:
        # 1. Vectorized Reweighting
        exponent = (k * Nplaq * array) - bias_unundo
        w_array = np.exp(np.clip(exponent - np.max(exponent), -700, 700))

        # 2. Fast Vectorized Blocking
        # Reshape for blocking: (blocks, configs_per_block)
        w_reshaped = w_array.reshape(blocks, -1)
        weight_sums = np.sum(w_reshaped, axis=1)
        
        sv_block_P = np.sum(array.reshape(blocks, -1) * w_reshaped, axis=1) / weight_sums
        sv_block_P2 = np.sum(p2_array.reshape(blocks, -1) * w_reshaped, axis=1) / weight_sums
        sv_block_P4 = np.sum(p4_array.reshape(blocks, -1) * w_reshaped, axis=1) / weight_sums

        # 3. Vectorized Jackknife (Leave-One-Out sums)
        sum_P, sum_P2, sum_P4 = np.sum(sv_block_P), np.sum(sv_block_P2), np.sum(sv_block_P4)
        
        P_loo = (sum_P - sv_block_P) / (blocks - 1)
        P2_loo = (sum_P2 - sv_block_P2) / (blocks - 1)
        P4_loo = (sum_P4 - sv_block_P4) / (blocks - 1)

        # Observables
        sus_samples = Nplaq * (P2_loo - P_loo**2)
        binder_samples = 1 - P4_loo / (3 * P2_loo**2)
        
        # Full estimators
        full_P = np.mean(sv_block_P)
        full_sus = Nplaq * (np.mean(sv_block_P2) - full_P**2)
        full_binder = 1 - np.mean(sv_block_P4) / (3 * np.mean(sv_block_P2)**2)

        # Bias Correction & Error Calc
        def get_jack_stats(full, loo_samples):
            mean_loo = np.mean(loo_samples)
            bias_corr = blocks * full - (blocks - 1) * mean_loo
            err = np.sqrt((blocks - 1) / blocks * np.sum((loo_samples - mean_loo)**2))
            return bias_corr, err

        val_P, err_P = get_jack_stats(full_P, P_loo)
        val_S, err_S = get_jack_stats(full_sus, sus_samples)
        val_B, err_B = get_jack_stats(full_binder, binder_samples)

        # 4. Append to Svendsen Lists
        svendsen_beta_array.append(beta_array[i] + k)
        svendsen_avg_plaq_array.append(val_P)
        svendsen_avg_plaq_SE_array.append(err_P)
        svendsen_sus_array.append(val_S)
        svendsen_sus_SE_array.append(err_S)
        svendsen_binder.append(val_B)
        svendsen_binder_SE_array.append(err_B)
        
        sus_jackknife_samples.append(sus_samples)
        binder_jackknife_samples.append(binder_samples)

        # 5. Capture original simulation points (k=0)
        if np.abs(k) < 1e-12:
            avg_plaq.append(val_P)
            SE_avg_plaq.append(err_P)
            plaq_sus.append(val_S)
            SE_plaq_sus.append(err_S)

# -------------------
# Plotting
# -------------------

plt.errorbar(beta_array,avg_plaq,yerr=SE_avg_plaq,fmt='o',label='Parallel Tempering Data',color='blue',ecolor='red',elinewidth=3)
plt.errorbar(svendsen_beta_array,svendsen_avg_plaq_array,yerr=svendsen_avg_plaq_SE_array,fmt='o',label='Svendsen Reweighted Data',color='blue',ecolor='yellow',elinewidth=3,capsize=0,markersize=1)
plt.show()
plt.errorbar(beta_array,plaq_sus,yerr=SE_plaq_sus,fmt='o',label='Parallel Tempering Data',color='blue',ecolor='red',elinewidth=3)
plt.errorbar(svendsen_beta_array,svendsen_sus_array,yerr=svendsen_sus_SE_array,fmt='o',label='Svendsen Reweighted Data',color='blue',ecolor='yellow',elinewidth=3,capsize=0,markersize=1)
plt.show()
plt.errorbar(svendsen_beta_array,svendsen_binder,yerr=svendsen_binder_SE_array,fmt='o',label='Svendsen Reweighted Data',color='blue',ecolor='yellow',elinewidth=3,capsize=0,markersize=1)
plt.show()



montecarlo_data=np.transpose(np.array([beta_array,avg_plaq,plaq_sus,SE_avg_plaq,SE_plaq_sus]))
filename=folder + 'PT'+str(Spatial_Size)+'^3*'+str(Temporal_Size)+'_analysis.csv'

header = ["Beta", "Avg Plaq", "Plaq Sus", "Jackknife Action SE", "Jackknife Sus SE"]
with open(filename, 'w', newline='') as csvfile:
    # Create a CSV writer object
    csv_writer = csv.writer(csvfile)

    # Write all rows at once
    csv_writer.writerow(header)
    csv_writer.writerows(montecarlo_data)


print(f"CSV file '{filename}' created successfully.")


svendsen_data=np.transpose(np.array([svendsen_beta_array,svendsen_avg_plaq_array,svendsen_sus_array,svendsen_binder,svendsen_avg_plaq_SE_array,svendsen_sus_SE_array,svendsen_binder_SE_array]))
filename2=folder + 'PT'+str(Spatial_Size)+'^3*'+str(Temporal_Size)+'_svendsen_analysis.csv'

header = ["Beta", "Avg Plaq", "Plaq Sus","Binder Cum", "Jackknife Action SE", "Jackknife Sus SE","Binder Cum SE"]
with open(filename2, 'w', newline='') as csvfile:
    # Create a CSV writer object
    csv_writer = csv.writer(csvfile)

    # Write all rows at once
    csv_writer.writerow(header)
    csv_writer.writerows(svendsen_data)


print(f"CSV file '{filename2}' created successfully.")



jackknife_filename =folder + 'PT'+str(Spatial_Size)+'^3*'+str(Temporal_Size)+'sus_jackknife_samples.csv'


jackknife_matrix = np.transpose(np.array(sus_jackknife_samples))

# Create header row (beta values)
header = [f"{beta:.6f}" for beta in svendsen_beta_array]

with open(jackknife_filename, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(header)
    writer.writerows(jackknife_matrix)

print(f"CSV file '{jackknife_filename}' created successfully with betas as columns.")


jackknife_filename =folder + 'PT'+str(Spatial_Size)+'^3*'+str(Temporal_Size)+'binder_jackknife_samples.csv'


jackknife_matrix = np.transpose(np.array(binder_jackknife_samples))

# Create header row (beta values)
header = [f"{beta:.6f}" for beta in svendsen_beta_array]

with open(jackknife_filename, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(header)
    writer.writerows(jackknife_matrix)

print(f"CSV file '{jackknife_filename}' created successfully with betas as columns.")


