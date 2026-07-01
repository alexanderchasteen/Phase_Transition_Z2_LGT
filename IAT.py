import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import numpy as np

def autocorr(data, maxlag=5000):
    """
    Calculates the autocorrelation of a dataset up to a maximum lag.
    """
    data = np.asarray(data)
    N = len(data)
    m = np.mean(data)
    var = np.var(data)  # Population variance (denominator N)
    
    # Initialize array of zeros
    rho = np.zeros(maxlag)
    
    if var == 0.0:
        rho[0] = 1.0
        return rho
        
    for t in range(maxlag):
        # Vectorized overlap calculation: (N - t) elements
        c = np.mean((data[:N-t] - m) * (data[t:] - m))
        rho[t] = c / var
        
    return rho

def tau_int(rho):
    """
    Calculates the integrated autocorrelation time using a Madras-Sokal window
    combined with a C++ noise-floor cutoff.
    """
    # Start at 0.5 so that 2.0 * (0.5 + sum(rho)) equals the standard formula: 1 + 2*sum(rho)
    tau = 0.5 
    
    for t in range(1, len(rho)):
        # 1. C++ Guardrail: Stop if the signal drops into pure noise
        if rho[t] < 0.05:
            break
            
        tau += rho[t]
        
        # 2. Madras-Sokal Window: Stop when the window size t is safely 
        # larger than the autocorrelation time scale (typically 4 to 6 times tau)
        if t > 6 * tau:
            break
            
    return 2.0 * tau, rho

# --- Main Analysis ---
# Load your dataset
for N in range(3,11):

    df = pd.read_csv(f'Raw_MC_Data_{N}^3_{N}_GOAT.csv')
    header_length = len(df.columns)
    max = 0
    for x in range(0,header_length):
        beta_idx = x # Analyze the first beta column
        data = df.iloc[:, beta_idx].to_numpy()
        tau_val, rho_cut = tau_int(autocorr(data))
        if tau_val > max: 
            max = tau_val

    print(f"Size {N}: beta {x}")
    print(f"Integrated Autocorrelation Time (tau_int): {tau_val:.2f}")
    

        # Visualization of the autocorrelation decay
        # plt.figure(figsize=(10, 5))
        # plt.semilogy(rho_cut, label=f'$\\tau_{{int}} = {tau_int:.1f}$')
        # plt.axhline(0, color='black', linestyle='--')
        # plt.xlabel('Lag ($t$)')
        # plt.ylabel('$\\rho(t)$ (log scale)')
        # plt.title('Autocorrelation Function Decay')
        # plt.legend()
        # plt.grid(True)
        # plt.show()