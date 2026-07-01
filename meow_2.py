# import pandas as pd
# from arch.bootstrap import optimal_block_length

# for x in range(4, 11):
#     filename = f'Raw_MC_Data_{x}^3_{x}_GOAT.csv'
#     try:
#         df = pd.read_csv(filename)
        
#         # Pass the entire DataFrame to arch. It handles all columns simultaneously.
#         # Output is a DataFrame with columns: 'stationary' and 'circular'
#         # and an index corresponding to your df's columns (the betas).
#         block_lengths = optimal_block_length(df)
        
#         # Select the bootstrap style you are using (e.g., 'stationary')
#         # and convert it to a dictionary mapping {beta_float: block_size_int}
#         beta_block_sizes = {
#             float(beta): int(size) 
#             for beta, size in block_lengths['stationary'].items()
#         }
        
#         # Identify the worst-performing beta (largest required block size)
#         worst_beta = max(beta_block_sizes, key=beta_block_sizes.get)
        
#         print(f"--- Lattice Size {x}^3 x {x} ---")
#         print(f"Worst beta = {worst_beta}")
#         print(f"Optimal block size = {beta_block_sizes[worst_beta]}\n")
        
#     except FileNotFoundError:
#         print(f"File {filename} not found, skipping.")
import pandas as pd
import numpy as np
from arch.bootstrap import optimal_block_length

# --- PARAMETERS ---
# If you are hitting the ceiling, start with 10. 
# This takes every 10th Monte Carlo sweep to break extreme autocorrelation.
THIN_STEP = 30

for x in [9]:
    filename = f'Raw_MC_Data_{x}^3_{x}_GOAT.csv'
    try:
        # Load data
        df = pd.read_csv(filename)
        
        # 1. Thin the data
        df_thinned = df.iloc[::THIN_STEP, :]
        n_thinned = len(df_thinned)
        
        # 2. Calculate the arch library's internal safety cap for the thinned data
        arch_ceiling = int(np.ceil(3 * np.sqrt(n_thinned)))
        
        # 3. Calculate block lengths
        block_lengths = optimal_block_length(df_thinned)
        
        print(f"\n================ LATTICE SIZE {x}^3 x {x} ================")
        print(f"Raw Sweeps           = {len(df)}")
        print(f"Thinned Sweeps (n)   = {n_thinned} (Taking every {THIN_STEP}th sweep)")
        print(f"Algorithm Safety Cap = {arch_ceiling}")
        print("-" * 80)
        print(f"{'Beta Column':<12} | {'MBB Size':<10} | {'Equivalent Raw Sweeps':<22} | {'Status'}")
        print("-" * 80)
        
        # 4. Use 'circular' for the Moving Block Bootstrap (MBB)
        for beta, size in block_lengths['circular'].items():
            size_int = int(size)
            
            # To get the actual number of sweeps your block represents, 
            # multiply the thinned block size by your thin step.
            raw_equivalent = size_int * THIN_STEP
            
            status = "Optimal"
            if size_int == arch_ceiling:
                status = "⚠️ STILL HITTING CEILING (Increase THIN_STEP)"
            elif size_int == 1:
                status = "ℹ️ Uncorrelated"
                
            print(f"{str(beta):<12} | {size_int:<10} | {raw_equivalent:<22} | {status}")
            
    except FileNotFoundError:
        print(f"\nFile {filename} not found, skipping.")