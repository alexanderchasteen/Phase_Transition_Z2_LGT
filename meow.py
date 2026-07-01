import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
import os
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
import warnings
# 3: 125 



# Suppress runtime warnings from exploratory curve_fit steps
warnings.filterwarnings('ignore', category=RuntimeWarning)

# --- Helper Function for Fast Overlapping Blocks ---
def rolling_sum(a, window):
    cs = np.cumsum(a, dtype=float)
    res = np.empty(len(a) - window + 1)
    res[0] = cs[window - 1]
    res[1:] = cs[window:] - cs[:-window]
    return res

# --- Model ---
def lorentzian(x_val, x0_val, gamma_val, a_val):
    g = np.abs(gamma_val)
    return a_val * g**2 / ((x_val - x0_val)**2 + g**2)

# --- Configuration ---
N_BOOT = 1000  # Number of bootstrap resamples
top_percent = 0.1  # Isolate the top 10% of the susceptibility peak
folder = './Multi_Canon_Results/'
os.makedirs(folder, exist_ok=True) # Ensure folder exists
tau_int_vector = [4, 9, 6, 8, 18, 35, 46]

# --- Optimized Weights Lookup ---
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


# =========================
# Main Execution Loop
# =========================
for x in range(4, 11):
    Spatial_Size = int(x)
    Temporal_Size = Spatial_Size
    Nplaq = Spatial_Size**3 * Temporal_Size * 6
    
    tau_int = tau_int_vector[x-3]
    weights = FastWeights(f"/home/alexa/Phase_Transition_Z2_LGT/weights/weight_function{Spatial_Size}.txt")
    
    # Load raw data once per spatial size
    df = pd.read_csv(f'Raw_MC_Data_{x}^3_{x}_GOAT.csv')
    beta_array = df.columns.to_numpy(dtype=float)
    CONFIGS = len(beta_array)
    beta_nbhd_radius = (beta_array[1] - beta_array[0]) / 2
    total_rows = len(df)
    
    print(f"\n=============================================")
    print(f"Processing Spatial Size: {Spatial_Size}^3 * {Temporal_Size}")
    print(f"=============================================")

    # Containers to hold data for graphing this specific volume size
    plot_multipliers = []
    plot_se_beta = []
    plot_se_beta_err = []

    for multiplier in range(250, 1000, 25):
        block_size = int(multiplier * tau_int)
        blocks = total_rows // block_size
        new_length = block_size * blocks
        num_overlapping_blocks = new_length - block_size + 1
        
        # Stop tracking if we run out of statistical entities (block starvation)
        if blocks < 8:
            print(f"Multiplier {multiplier} results in too few blocks ({blocks}). Stopping sweep.")
            break

        # Generate Bootstrap Indices
        boot_idx = np.random.randint(0, num_overlapping_blocks, size=(N_BOOT, blocks))
        
        # Storage for this multiplier
        svendsen_beta_array = []
        sus_bootstrap_samples = []
        svendsen_sus_array = []
        svendsen_sus_SE_array = []
        
        # --- 1. Reweighting and Bootstrapping ---
        for i in range(CONFIGS):
            array = df.iloc[:, i].to_numpy()[:new_length]
            p2_array = array**2
            bias_unundo = weights.get_weights_vectorized(array)
            k_array = np.linspace(-beta_nbhd_radius, beta_nbhd_radius, 101)
            
            for k in k_array:
                exponent = (k * Nplaq * array) - bias_unundo
                w_array = np.exp(np.clip(exponent - np.max(exponent), -700, 700))

                W_block = rolling_sum(w_array, block_size)
                O_P_block = rolling_sum(array * w_array, block_size)
                O_P2_block = rolling_sum(p2_array * w_array, block_size)

                W_tot = np.sum(w_array)
                O_P_tot = np.sum(array * w_array)
                O_P2_tot = np.sum(p2_array * w_array)

                W_boot = np.sum(W_block[boot_idx], axis=1)
                P_boot = np.sum(O_P_block[boot_idx], axis=1) / W_boot
                P2_boot = np.sum(O_P2_block[boot_idx], axis=1) / W_boot

                sus_samples = Nplaq * (P2_boot - P_boot**2)
                
                full_P = O_P_tot / W_tot
                full_sus = Nplaq * ((O_P2_tot / W_tot) - full_P**2)
                err_S = np.std(sus_samples, ddof=1)

                svendsen_beta_array.append(beta_array[i] + k)
                svendsen_sus_array.append(full_sus)
                svendsen_sus_SE_array.append(err_S)
                sus_bootstrap_samples.append(sus_samples)

        svendsen_beta = np.array(svendsen_beta_array)
        sus_sv = np.array(svendsen_sus_array)
        boot_matrix = np.transpose(np.array(sus_bootstrap_samples))

        # --- 2. Master Fit and Masking ---
        emp_max_idx = np.argmax(sus_sv)
        emp_beta = svendsen_beta[emp_max_idx]
        emp_max = sus_sv[emp_max_idx]
        emp_min = np.min(sus_sv)

        threshold = emp_max - top_percent * (emp_max - emp_min)
        master_mask = sus_sv > threshold

        x_fit = svendsen_beta[master_mask]
        y_master = sus_sv[master_mask]
        initial_guess = [emp_beta, 0.0005, emp_max]

        try:
            param, _ = curve_fit(lorentzian, x_fit, y_master, p0=initial_guess, maxfev=10000)
            x0, gamma_raw, a = param
            gamma = np.abs(gamma_raw)
        except RuntimeError:
            print(f"Master fit failed for multiplier {multiplier}. Skipping.")
            continue

        # --- 3. Bootstrap Fits ---
        crit_beta_samples = []
        for b_idx in range(boot_matrix.shape[0]): 
            y_fit = boot_matrix[b_idx, master_mask]
            if len(x_fit) < 5:
                continue
            
            try:
                param_i, _ = curve_fit(lorentzian, x_fit, y_fit, p0=[x0, gamma, a], maxfev=10000)
                crit_beta_samples.append(param_i[0])
            except RuntimeError:
                continue

        crit_beta_samples = np.array(crit_beta_samples)
        B_beta = len(crit_beta_samples)
        
        if B_beta == 0:
            print(f"All bootstrap fits failed for multiplier {multiplier}. Skipping.")
            continue

        # --- 4. Evaluate and Store Errors ---
        SE_beta = np.std(crit_beta_samples, ddof=1)
        SE_beta_err = SE_beta / np.sqrt(2 * (B_beta - 1))

        print(f"M={multiplier:2d} (Block={block_size}) | SE_beta: {SE_beta:.3e} ± {SE_beta_err:.3e}")
        
        # Save values for plotting
        plot_multipliers.append(multiplier)
        plot_se_beta.append(SE_beta)
        plot_se_beta_err.append(SE_beta_err)

    # ==========================================
    # Generate and Save Plateau Graph
    # ==========================================
    if len(plot_multipliers) > 0:
        plt.figure(figsize=(10, 6))
        plt.errorbar(
            plot_multipliers, 
            plot_se_beta, 
            yerr=plot_se_beta_err, 
            fmt='-o', 
            color='blue', 
            ecolor='red', 
            capsize=4, 
            linewidth=1.5, 
            markersize=5,
            label=f'Data Error Profile'
        )
        
        # Make the plot clean and readable
        plt.xlabel(r'Block Length Multiplier ($M = \text{Block Size} / \tau_{int}$)', fontsize=12)
        plt.ylabel(r'Standard Error of Critical Beta ($\sigma_{\beta_c}$)', fontsize=12)
        plt.title(f'Block Bootstrap Error Analysis: $Spatial Size={Spatial_Size}^3$', fontsize=14, fontweight='bold')
        plt.grid(True, linestyle='--', alpha=0.5)
        
        # Annotate features to watch out for
        plt.axvspan(5, 12, color='gray', alpha=0.1, label='Potential Bias Zone (Small blocks)')
        
        plt.legend(loc='best')
        plt.tight_layout()
        
        plot_filename = os.path.join(folder, f'plateau_analysis_L{Spatial_Size}.png')
        plt.savefig(plot_filename, dpi=300)
        plt.close()
        print(f"\n---> Successfully saved plateau diagnostic plot to: {plot_filename}\n")
    else:
        print(f"!!! Error: No successful data points gathered to graph for L={Spatial_Size}\n")