import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
import warnings
# Chose multiplier: make block size larger and larger until it plateus. we see from 40-30 the error increases and then 30 to 20 it starts to plateu. If make too small then the math breaks down.
#Want to minimize the block size while at the error plateu (because then less noise because Error on the Error is minimized)

# Suppress runtime warnings from exploratory curve_fit steps
warnings.filterwarnings('ignore', category=RuntimeWarning)

# =========================
# Configuration Parameters
# =========================
top_percent = 0.1  # Isolate the top 10% of the susceptibility p    eak

# =========================
# Parameters Loop
# =========================
for x in range(4, 11):
    Spatial_Size = int(x)
    Temporal_Size = Spatial_Size
    folder = './Multi_Canon_Results/'

    # =========================
    # Load data
    # =========================
    df = pd.read_csv(folder + f'PT{Spatial_Size}^3*{Temporal_Size}_analysis.csv')
    df2 = pd.read_csv(folder + f'PT{Spatial_Size}^3*{Temporal_Size}_svendsen_analysis.csv')

    beta_pt = df['Beta'].to_numpy()
    sus_pt = df['Plaq Sus'].to_numpy()
    sus_pt_SE = df['Bootstrap Sus SE'].to_numpy()  # UPDATED: Updated to Bootstrap SE key

    svendsen_beta = df2['Beta'].to_numpy()
    sus_sv = df2['Plaq Sus'].to_numpy()
    sus_sv_SE = df2['Bootstrap Sus SE'].to_numpy() # UPDATED: Updated to Bootstrap SE key

    # =========================
    # Model (Enforced Absolute Bounds)
    # =========================
    def lorentzian(x_val, x0_val, gamma_val, a_val):
        g = np.abs(gamma_val)  # Keeps sign handling uniform across fits
        return a_val * g**2 / ((x_val - x0_val)**2 + g**2)

    # =========================
    # Dynamic Fit Window Selection (Master Fit)
    # =========================
    emp_max_idx = np.argmax(sus_sv)
    emp_beta = svendsen_beta[emp_max_idx]
    emp_max = sus_sv[emp_max_idx]
    emp_min = np.min(sus_sv)

    # Define the master mask ONCE based on the global dataset
    threshold = emp_max - top_percent * (emp_max - emp_min)
    master_mask = sus_sv > threshold

    # Apply master mask to the x and y coordinates
    x_fit = svendsen_beta[master_mask]
    y_master = sus_sv[master_mask]
    yerr = sus_sv_SE[master_mask]

    # Dynamically generate a reliable initial guess based on the data peak
    initial_guess = [emp_beta, 0.0005, emp_max]

    # Master Fit
    param, _ = curve_fit(lorentzian, x_fit, y_master, p0=initial_guess, maxfev=10000)
    x0, gamma_raw, a = param
    gamma = np.abs(gamma_raw)
    R2 = r2_score(y_master, lorentzian(x_fit, x0, gamma, a))

    # =========================
    # Corrected Bootstrap Parsing
    # =========================
    boot_filepath = folder + f'PT{Spatial_Size}^3*{Temporal_Size}sus_bootstrap_samples.csv'

    # 1. Safely extract exact matrix beta coordinates from the header row
    with open(boot_filepath, 'r') as csvfile:
        header_line = csvfile.readline().strip().split(',')
    beta_boot = np.array([float(beta_val) for beta_val in header_line])
    
    # Pre-slice the Bootstrap x-coordinates using the master mask
    x_fit_boot = beta_boot[master_mask]

    # 2. Load numeric rows skipping the column header labels
    boot = np.loadtxt(boot_filepath, delimiter=',', skiprows=1)

    crit_beta_samples = []
    FWHM_samples = []
    height_samples = []

    # Loop over true data rows (each row represents a full bootstrap sample dataset curve)
    for i in range(boot.shape[0]): 
        y_sample = boot[i, :]

        # 3. Apply the EXACT SAME mask from the master data
        y_fit = y_sample[master_mask]

        if len(x_fit_boot) < 5:
            continue
        
        try:
            param_i, _ = curve_fit(lorentzian, x_fit_boot, y_fit, p0=[x0, gamma, a], maxfev=10000)
            x0_i, gamma_i_raw, a_i = param_i
            gamma_i = np.abs(gamma_i_raw)
            
            # Interactive visualization diagnostics window (Triggers on sample 8)
            if i == 8:
                plt.figure(figsize=(8, 5))
                # Plot full background context
                plt.plot(beta_boot, y_sample, '.', color='lightgray', markersize=2, label='Full Bootstrap Sample')
                # Highlight strictly masked points
                plt.plot(x_fit_boot, y_fit, 'o', color='royalblue', markersize=4, label=f'Top {int(top_percent*100)}% Data')
                
                cts_i = np.linspace(np.min(x_fit_boot), np.max(x_fit_boot), 1000)
                plt.plot(cts_i, lorentzian(cts_i, x0_i, gamma_i, a_i), 'r-', label=f'x0={x0_i:.8f}, FWHM={2*gamma_i:.3e}, h={a_i:.2f}')
                
                plt.xlabel('Beta')
                plt.ylabel('Susceptibility')
                plt.title(f'Z2 LGT L={Spatial_Size}, Bootstrap sample {i} (Lorentzian)')
                plt.legend()
                
                # Snug axis limits tailored directly to the peak
                padding_x = (np.max(x_fit_boot) - np.min(x_fit_boot)) * 0.5
                plt.xlim(np.min(x_fit_boot) - padding_x, np.max(x_fit_boot) + padding_x)
                plt.ylim(threshold * 0.95, np.max(y_fit) * 1.02)
                plt.show()

            crit_beta_samples.append(x0_i)
            FWHM_samples.append(2 * gamma_i)
            height_samples.append(a_i)
        except RuntimeError:
            print(f"Fit failed to converge for Bootstrap sample {i}")
            continue

    # Convert lists to arrays for calculations
    crit_beta_samples = np.array(crit_beta_samples)
    FWHM_samples = np.array(FWHM_samples)
    height_samples = np.array(height_samples)

    # Final error estimation via Bootstrap Standard Deviation
    SE_beta = np.std(crit_beta_samples, ddof=1)
    SE_FWHM = np.std(FWHM_samples, ddof=1)
    SE_height = np.std(height_samples, ddof=1)

    B = len(crit_beta_samples)
    if B > 0:
        beta_bar = np.mean(crit_beta_samples)
        FWHM_bar = np.mean(FWHM_samples)
        height_bar = np.mean(height_samples)
        
        # Standard Bootstrap Bias Correction Formula: 2*theta_master - theta_boot_mean
        beta_bc = 2 * x0 - beta_bar
        FWHM_bc = 2 * (2 * gamma) - FWHM_bar
        height_bc = 2 * a - height_bar

        # Print Bias Corrected Results alongside the bootstrap standard errors
        # print(f"{Spatial_Size}, {x0}, {2*gamma}, {a}, {SE_beta}, {SE_FWHM}, {SE_height}")
        print(f"{Spatial_Size}, {beta_bc}, {FWHM_bc}, {height_bc}, {SE_beta}, {SE_FWHM}, {SE_height}")

        # Number of successful bootstrap fits
        B_beta = len(crit_beta_samples)
        B_FWHM = len(FWHM_samples)
        B_height = len(height_samples)

        # Error on the error
        SE_beta_err = SE_beta / np.sqrt(2 * (B_beta - 1))
        SE_FWHM_err = SE_FWHM / np.sqrt(2 * (B_FWHM - 1))
        SE_height_err = SE_height / np.sqrt(2 * (B_height - 1))

        # print(f"{SE_beta_err}, {SE_FWHM_err},{SE_height_err}")