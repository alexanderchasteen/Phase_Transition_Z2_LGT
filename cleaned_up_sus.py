import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
import warnings

# Suppress runtime warnings from exploratory curve_fit steps
warnings.filterwarnings('ignore', category=RuntimeWarning)

# =========================
# Configuration Parameters
# =========================
top_percent = 0.1  # Isolate the top 20% of the susceptibility peak

# =========================
# Parameters Loop
# =========================
for x in range(3, 11):
    Spatial_Size = int(x)
    Temporal_Size = Spatial_Size
    folder = './Multi_Canon_Results_moving/'

    # =========================
    # Load data
    # =========================
    df = pd.read_csv(folder + f'PT{Spatial_Size}^3*{Temporal_Size}_analysis.csv')
    df2 = pd.read_csv(folder + f'PT{Spatial_Size}^3*{Temporal_Size}_svendsen_analysis.csv')

    beta_pt = df['Beta'].to_numpy()
    sus_pt = df['Plaq Sus'].to_numpy()
    sus_pt_SE = df['Jackknife Sus SE'].to_numpy()

    svendsen_beta = df2['Beta'].to_numpy()
    sus_sv = df2['Plaq Sus'].to_numpy()
    sus_sv_SE = df2['Jackknife Sus SE'].to_numpy()

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
    # Corrected Jackknife Parsing
    # =========================
    jack_filepath = folder + f'PT{Spatial_Size}^3*{Temporal_Size}sus_jackknife_samples.csv'

    # 1. Safely extract exact matrix beta coordinates from the header row
    with open(jack_filepath, 'r') as csvfile:
        header_line = csvfile.readline().strip().split(',')
    beta_jack = np.array([float(beta_val) for beta_val in header_line])
    
    # Pre-slice the Jackknife x-coordinates using the master mask
    x_fit_jack = beta_jack[master_mask]

    # 2. Load numeric rows skipping the column header labels
    jack = np.loadtxt(jack_filepath, delimiter=',', skiprows=1)

    crit_beta_samples = []
    FWHM_samples = []
    height_samples = []
    
    def jackknife_var(data_array, reference):
        N = len(data_array)
        return (N - 1) * np.mean((data_array - np.mean(data_array))**2)

    # Loop over true data rows
    for i in range(jack.shape[0]): 
        y_sample = jack[i, :]

        # 3. Apply the EXACT SAME mask from the master data
        y_fit = y_sample[master_mask]

        if len(x_fit_jack) < 5:
            continue
        
        try:
            param_i, _ = curve_fit(lorentzian, x_fit_jack, y_fit, p0=[x0, gamma, a], maxfev=10000)
            x0_i, gamma_i_raw, a_i = param_i
            gamma_i = np.abs(gamma_i_raw)
            
            # Interactive visualization diagnostics window (Triggers on block 8)
            if i == 8:
                plt.figure(figsize=(8, 5))
                # Plot full background context
                plt.plot(beta_jack, y_sample, '.', color='lightgray', markersize=2, label='Full Data')
                # Highlight strictly masked points
                plt.plot(x_fit_jack, y_fit, 'o', color='royalblue', markersize=4, label=f'Top {int(top_percent*100)}% Data')
                
                cts_i = np.linspace(np.min(x_fit_jack), np.max(x_fit_jack), 1000)
                plt.plot(cts_i, lorentzian(cts_i, x0_i, gamma_i, a_i), 'r-', label=f'x0={x0_i:.8f}, FWHM={2*gamma_i:.3e}, h={a_i:.2f}')
                
                plt.xlabel('Beta')
                plt.ylabel('Susceptibility')
                plt.title(f'Z2 LGT L={Spatial_Size}, Jackknife block {i} (Lorentzian)')
                plt.legend()
                
                # Snug axis limits tailored directly to the peak (Updated variables)
                padding_x = (np.max(x_fit_jack) - np.min(x_fit_jack)) * 0.5
                plt.xlim(np.min(x_fit_jack) - padding_x, np.max(x_fit_jack) + padding_x)
                plt.ylim(threshold * 0.95, np.max(y_fit) * 1.02)
                plt.show()

            crit_beta_samples.append(x0_i)
            FWHM_samples.append(2 * gamma_i)
            height_samples.append(a_i)
        except RuntimeError:
            print(f"Fit failed to converge for Jackknife block {i}")
            continue

    # Final error estimation calculations
    SE_beta = np.sqrt(jackknife_var(np.array(crit_beta_samples), x0))
    SE_FWHM = np.sqrt(jackknife_var(np.array(FWHM_samples), 2*gamma))
    SE_height = np.sqrt(jackknife_var(np.array(height_samples), a))

    # Raw / Standard Fit metrics printout
    # print(f"{Spatial_Size}, {x0}, {2*gamma}, {a}, {SE_beta}, {SE_FWHM}, {SE_height}")

    B = len(crit_beta_samples)
    if B > 0:
        beta_bar = np.mean(crit_beta_samples)
        FWHM_bar = np.mean(FWHM_samples)
        height_bar = np.mean(height_samples)
        
        # Clean mathematical bias corrections
        beta_bc = B * x0 - (B - 1) * beta_bar
        FWHM_bc = B * (2 * gamma) - (B - 1) * FWHM_bar
        height_bc = B * a - (B - 1) * height_bar

        # Print Bias Corrected Results alongside the standard errors
        print(f"{Spatial_Size}, {beta_bc}, {FWHM_bc}, {height_bc}, {SE_beta}, {SE_FWHM}, {SE_height}")

