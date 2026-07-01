import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv

# --- Helper Function for Fast Overlapping Blocks ---
def rolling_sum(a, window):
    cs = np.cumsum(a, dtype=float)
    res = np.empty(len(a) - window + 1)
    res[0] = cs[window - 1]
    res[1:] = cs[window:] - cs[:-window]
    return res

# FIXED: Store tau_int directly. We will calculate block sizes dynamically below.
# tau_int_vector = [9, 6, 8, 18, 35, 46, 530]
# multiplier_vector = [550, 200, 150, 175, 350, 200, 410]
block_size_vector = [448,186,367,538,419,368,409]
thinning_vector = [20, 10, 10, 10, 20, 30,20]
block_size_vector = [a * b for a, b in zip(block_size_vector, thinning_vector)]
# --- Configuration ---
N_BOOT = 4000  # Number of bootstrap resamples

for x in range(4, 11):
    Spatial_Size = x
    Temporal_Size = Spatial_Size
    Nplaq = Spatial_Size**3 * Temporal_Size * 6
    folder = './Multi_Canon_Results/'
    
    # 1. Grab the specific tau_int for this volume
    # tau_int = tau_int_vector[x-3]
    # multiplier = multiplier_vector[x-3]
    # 2. Define block size directly as a multiple of tau_int
    block_size = block_size_vector[x-4]
    # block_size = 5000
    
    # Load Data
    df = pd.read_csv(f'Raw_MC_Data_{x}^3_{x}_GOAT.csv')

    # THIN_STEP = thinning_vector[x-4]

    # df = df.iloc[::THIN_STEP, :]

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

    sus_bootstrap_samples = []
    binder_bootstrap_samples = []

    # ==========================================================
    # CRITICAL FIX 2: Generate Bootstrap Indices ONCE GLOBALLY
    # ==========================================================
    # All columns in the dataframe have the identical length
    total_rows = len(df)
    blocks = total_rows // block_size
    new_length = block_size * blocks
    num_overlapping_blocks = new_length - block_size + 1

    # One master set of bootstrap indices for all beta streams and all k-shifts
    boot_idx = np.random.randint(0, num_overlapping_blocks, size=(N_BOOT, blocks))

    # -------------------
    # Main Loop
    # -------------------
    for i in range(CONFIGS):
        print(f"Processing Beta index {i}: {beta_array[i]}")
        array = df.iloc[:, i].to_numpy()
        
        # Truncate array to perfectly fit the block sizes we defined above
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

            # 2. Fast Vectorized Overlapping Blocking
            W_block = rolling_sum(w_array, block_size)
            O_P_block = rolling_sum(array * w_array, block_size)
            O_P2_block = rolling_sum(p2_array * w_array, block_size)
            O_P4_block = rolling_sum(p4_array * w_array, block_size)

            # Total sums over all points
            W_tot = np.sum(w_array)
            O_P_tot = np.sum(array * w_array)
            O_P2_tot = np.sum(p2_array * w_array)
            O_P4_tot = np.sum(p4_array * w_array)

            # 3. Vectorized Overlapping Block Bootstrap (MBB)
            # This now universally relies on the globally generated boot_idx
            W_boot = np.sum(W_block[boot_idx], axis=1)
            P_boot = np.sum(O_P_block[boot_idx], axis=1) / W_boot
            P2_boot = np.sum(O_P2_block[boot_idx], axis=1) / W_boot
            P4_boot = np.sum(O_P4_block[boot_idx], axis=1) / W_boot

            # ... [Rest of code continues exactly as before] ...

            # Observables evaluated on bootstrap resamples
            sus_samples = Nplaq * (P2_boot - P_boot**2)
            binder_samples = 1 - P4_boot / (3 * P2_boot**2)
            
            # Full sample estimators
            full_P = O_P_tot / W_tot
            full_sus = Nplaq * ((O_P2_tot / W_tot) - full_P**2)
            full_binder = 1 - (O_P4_tot / W_tot) / (3 * (O_P2_tot / W_tot)**2)

            # Error Calculation via Bootstrap Standard Deviation
            def get_boot_stats(full, boot_samples):
                err = np.std(boot_samples, ddof=1)
                return full, err

            val_P, err_P = get_boot_stats(full_P, P_boot)
            val_S, err_S = get_boot_stats(full_sus, sus_samples)
            val_B, err_B = get_boot_stats(full_binder, binder_samples)

            # 4. Append to Svendsen Lists
            svendsen_beta_array.append(beta_array[i] + k)
            svendsen_avg_plaq_array.append(val_P)
            svendsen_avg_plaq_SE_array.append(err_P)
            svendsen_sus_array.append(val_S)
            svendsen_sus_SE_array.append(err_S)
            svendsen_binder.append(val_B)
            svendsen_binder_SE_array.append(err_B)
            
            sus_bootstrap_samples.append(sus_samples)
            binder_bootstrap_samples.append(binder_samples)

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

    header = ["Beta", "Avg Plaq", "Plaq Sus", "Bootstrap Action SE", "Bootstrap Sus SE"]
    with open(filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(header)
        csv_writer.writerows(montecarlo_data)

    print(f"CSV file '{filename}' created successfully.")

    svendsen_data = np.transpose(np.array([svendsen_beta_array, svendsen_avg_plaq_array, svendsen_sus_array, svendsen_binder, svendsen_avg_plaq_SE_array, svendsen_sus_SE_array, svendsen_binder_SE_array]))
    filename2 = folder + 'PT' + str(Spatial_Size) + '^3*' + str(Temporal_Size) + '_svendsen_analysis.csv'

    header = ["Beta", "Avg Plaq", "Plaq Sus", "Binder Cum", "Bootstrap Action SE", "Bootstrap Sus SE", "Binder Cum SE"]
    with open(filename2, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(header)
        csv_writer.writerows(svendsen_data)

    print(f"CSV file '{filename2}' created successfully.")

    # Bootstrap Sample CSVs
    boot_filename = folder + 'PT' + str(Spatial_Size) + '^3*' + str(Temporal_Size) + 'sus_bootstrap_samples.csv'
    boot_matrix = np.transpose(np.array(sus_bootstrap_samples))
    header = [f"{beta}" for beta in svendsen_beta_array]

    with open(boot_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        writer.writerows(boot_matrix)

    print(f"CSV file '{boot_filename}' created successfully with betas as columns.")

    boot_filename = folder + 'PT' + str(Spatial_Size) + '^3*' + str(Temporal_Size) + 'binder_bootstrap_samples.csv'
    boot_matrix = np.transpose(np.array(binder_bootstrap_samples))

    with open(boot_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        writer.writerows(boot_matrix)

    print(f"CSV file '{boot_filename}' created successfully with betas as columns.")