import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore', category=RuntimeWarning)

# --- Configuration ---
# You can wrap this back in the 'for x in range(3, 11):' loop if needed, 
# but testing on a single volume first is recommended to tune your K array.
Spatial_Size = 9
Temporal_Size = Spatial_Size
Nplaq = Spatial_Size**3 * Temporal_Size * 6
folder = './Multi_Canon_Results_25/'

# Load Data
filename = f'Raw_MC_Data_{Spatial_Size}^3_{Temporal_Size}_GOAT.csv'
df = pd.read_csv(filename)
beta_array = df.columns.to_numpy(dtype=float)
N_total = len(df)

# Define the block sizes K to test. 
# Using a geometric progression to sample small and large K efficiently.
# Max block size is N_total // 10 so we still have at least 10 blocks to compute a variance.
# Instead of N_total // 10, cap the max block size to ensure N_B >= 50
max_K = N_total // 1 
K_values = np.unique(np.geomspace(2, max_K, num=50).astype(int))


# Pick a subset of beta indices to plot so the graph doesn't become an unreadable mess
# (e.g., picking 5 evenly spaced betas across your phase transition range)
plot_indices = np.linspace(0, len(beta_array) - 1, 5, dtype=int)

plt.figure(figsize=(14, 6))

# Loop over the selected betas
for idx in plot_indices:
    beta = beta_array[idx]
    array = df.iloc[:, idx].to_numpy()
    
    variances = []
    
    for K in K_values:
        N_B = N_total // K
        
        # 1. Truncate data to fit perfectly into N_B blocks
        truncated_array = array[:N_B * K]
        
        # 2. Reshape to (N_B, K) for instant vectorized block calculations
        blocks = truncated_array.reshape(N_B, K)
        
        # 3. Compute block means for <P> and <P^2>
        block_P_mean = np.mean(blocks, axis=1)
        block_P2_mean = np.mean(blocks**2, axis=1)
        
        # 4. Compute the new variable Xi (Block Susceptibility)
        block_sus = Nplaq * (block_P2_mean - block_P_mean**2)
        
        # 5. Compute variance of these block variables
        # Using ddof=1 for an unbiased sample variance estimator
        var_X = np.var(block_sus, ddof=1)
        variances.append(var_X)
        
    variances = np.array(variances)
    
    # --- Plotting Subplot 1: Raw Variance ---
    plt.subplot(1, 2, 1)
    plt.plot(K_values, variances, marker='o', markersize=4, label=f'Beta = {beta:.4f}')
    
    # --- Plotting Subplot 2: Plateau Diagnostic ---
    plt.subplot(1, 2, 2)
    plt.plot(K_values, variances * K_values, marker='o', markersize=4, label=f'Beta = {beta:.4f}')

# Formatting Subplot 1
plt.subplot(1, 2, 1)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Block Size ($K$)')
plt.ylabel('Variance of Blocked Susceptibility')
plt.title('Decay of Block Variance (Expected $1/K$)')
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.legend()

# Formatting Subplot 2
plt.subplot(1, 2, 2)
plt.xscale('log')
plt.xlabel('Block Size ($K$)')
plt.ylabel('$K \\times$ Variance (Plateau Estimator)')
plt.title('Independence Plateau Diagnostic')
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.legend()

plt.tight_layout()
plt.show()