import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv

# FIXED: Corrected variable name to prevent overwriting and using IAT as block count
block_vector_raw = [4, 9, 6, 8, 18, 35, 46, 530]
block_vector = [500000 // (20 * x) for x in block_vector_raw]

# --- Configuration ---
for x in range(3, 11):
    Spatial_Size = x
    Temporal_Size = Spatial_Size
    Nplaq = Spatial_Size**3 * Temporal_Size * 6
    folder = './Multi_Canon_Results/'
    blocks = block_vector[x-3]
    
    # Load Data
    df = pd.read_csv(f'Raw_MC_Data_{x}^3_{x}_GOAT.csv')
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

    weights = FastWeights(f"/home/alexa/Phase_Transition_Z2_LGT/weights/weight_function{Spatial_Size}.txt")

    print("\n=== PYTHON BIN CHECK ===")
    for i in range(5):
        print(f"edge[{i}] = {weights.plaq_bins[i]} | weight[{i}] = {weights.weight_vals[i]}")
    print("last edge =", weights.plaq_bins[-1])
    print("sizes:", len(weights.plaq_bins), len(weights.weight_vals))
    
    # -------------------
    # Storage Lists
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
        new_length = len(array) // blocks * blocks
        array = array[:new_length]
        
        # Pre-calculate powers once per beta to save time
        p2_array = array**2
        p4_array = array**4
        
        # Pre-fetch unbiasing weights for the whole config ensemble
        bias_unundo = weights.get_weights_vectorized(array)
        
        k_array = np.linspace(-beta_nbhd_radius, beta_nbhd_radius, 101)
        
        for k in k_array:
            # 1. Vectorized Reweighting
            exponent = (k * Nplaq * array) - bias_unundo
            w_array = np.exp(np.clip(exponent - np.max(exponent), -700, 700))

            # 2. Fast Vectorized Blocking (Corrected for Jackknife)
            w_reshaped = w_array.reshape(blocks, -1)
            
            # Sum of weights in each block
            W_block = np.sum(w_reshaped, axis=1)
            
            # Unnormalized weighted sums per block
            O_P_block = np.sum(array.reshape(blocks, -1) * w_reshaped, axis=1)
            O_P2_block = np.sum(p2_array.reshape(blocks, -1) * w_reshaped, axis=1)
            O_P4_block = np.sum(p4_array.reshape(blocks, -1) * w_reshaped, axis=1)

            # Total sums over all blocks
            W_tot = np.sum(W_block)
            O_P_tot = np.sum(O_P_block)
            O_P2_tot = np.sum(O_P2_block)
            O_P4_tot = np.sum(O_P4_block)

            # 3. Vectorized Jackknife (Correct Leave-One-Out estimators)
            W_loo = W_tot - W_block
            P_loo = (O_P_tot - O_P_block) / W_loo
            P2_loo = (O_P2_tot - O_P2_block) / W_loo
            P4_loo = (O_P4_tot - O_P4_block) / W_loo

            # Observables on the leave-one-out samples
            sus_samples = Nplaq * (P2_loo - P_loo**2)
            binder_samples = 1 - P4_loo / (3 * P2_loo**2)
            
            # Full estimators
            full_P = O_P_tot / W_tot
            full_sus = Nplaq * ((O_P2_tot / W_tot) - full_P**2)
            full_binder = 1 - (O_P4_tot / W_tot) / (3 * (O_P2_tot / W_tot)**2)

            # Bias Correction & Error Calc
            def get_jack_stats(full, loo_samples):
                err = np.sqrt((blocks - 1) / blocks * np.sum((loo_samples - np.mean(loo_samples))**2))
                return full, err

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
    plt.errorbar(beta_array, avg_plaq, yerr=SE_avg_plaq, fmt='o', label='Parallel Tempering Data', color='blue', ecolor='red', elinewidth=3)
    plt.errorbar(svendsen_beta_array, svendsen_avg_plaq_array, yerr=svendsen_avg_plaq_SE_array, fmt='o', label='Svendsen Reweighted Data', color='blue', ecolor='yellow', elinewidth=3, capsize=0, markersize=1)
    # plt.show()
    
    plt.errorbar(beta_array, plaq_sus, yerr=SE_plaq_sus, fmt='o', label='Parallel Tempering Data', color='blue', ecolor='red', elinewidth=3)
    plt.errorbar(svendsen_beta_array, svendsen_sus_array, yerr=svendsen_sus_SE_array, fmt='o', label='Svendsen Reweighted Data', color='blue', ecolor='yellow', elinewidth=3, capsize=0, markersize=1)
    # plt.show()
    
    plt.errorbar(svendsen_beta_array, svendsen_binder, yerr=svendsen_binder_SE_array, fmt='o', label='Svendsen Reweighted Data', color='blue', ecolor='yellow', elinewidth=3, capsize=0, markersize=1)
    # plt.show()

    # -------------------
    # CSV Exports
    # -------------------
    montecarlo_data = np.transpose(np.array([beta_array, avg_plaq, plaq_sus, SE_avg_plaq, SE_plaq_sus]))
    filename = folder + 'PT' + str(Spatial_Size) + '^3*' + str(Temporal_Size) + '_analysis.csv'

    header = ["Beta", "Avg Plaq", "Plaq Sus", "Jackknife Action SE", "Jackknife Sus SE"]
    with open(filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(header)
        csv_writer.writerows(montecarlo_data)

    print(f"CSV file '{filename}' created successfully.")

    svendsen_data = np.transpose(np.array([svendsen_beta_array, svendsen_avg_plaq_array, svendsen_sus_array, svendsen_binder, svendsen_avg_plaq_SE_array, svendsen_sus_SE_array, svendsen_binder_SE_array]))
    filename2 = folder + 'PT' + str(Spatial_Size) + '^3*' + str(Temporal_Size) + '_svendsen_analysis.csv'

    header = ["Beta", "Avg Plaq", "Plaq Sus", "Binder Cum", "Jackknife Action SE", "Jackknife Sus SE", "Binder Cum SE"]
    with open(filename2, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(header)
        csv_writer.writerows(svendsen_data)

    print(f"CSV file '{filename2}' created successfully.")

    # Jackknife CSVs
    jackknife_filename = folder + 'PT' + str(Spatial_Size) + '^3*' + str(Temporal_Size) + 'sus_jackknife_samples.csv'
    jackknife_matrix = np.transpose(np.array(sus_jackknife_samples))
    header = [f"{beta}" for beta in svendsen_beta_array]

    with open(jackknife_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        writer.writerows(jackknife_matrix)

    print(f"CSV file '{jackknife_filename}' created successfully with betas as columns.")

    jackknife_filename = folder + 'PT' + str(Spatial_Size) + '^3*' + str(Temporal_Size) + 'binder_jackknife_samples.csv'
    jackknife_matrix = np.transpose(np.array(binder_jackknife_samples))

    with open(jackknife_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        writer.writerows(jackknife_matrix)

    print(f"CSV file '{jackknife_filename}' created successfully with betas as columns.")