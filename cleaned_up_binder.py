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
top_percent = 0.05  # Isolate the lowest 15% of the Binder dip window

# =========================
# Parameters Loop
# =========================
for size in range(3, 11):
    Spatial_Size = size
    Temporal_Size = Spatial_Size
    folder = './Multi_Canon_Results/'

    # =========================
    # Load data
    # =========================
    df = pd.read_csv(folder + f'PT{Spatial_Size}^3*{Temporal_Size}_analysis.csv')
    df2 = pd.read_csv(folder + f'PT{Spatial_Size}^3*{Temporal_Size}_svendsen_analysis.csv')

    svendsen_beta = df2['Beta'].to_numpy()
    binder = df2['Binder Cum'].to_numpy()
    binder_SE = df2['Binder Cum SE'].to_numpy()

    # =========================
    # Model
    # =========================
    def quadratic(x, a, b, c):
        return b * (x - a)**2 + c

    # =========================
    # Dynamic Window Selection (Master Fit)
    # =========================
    # Locate the true empirical minimum of the Binder trough
    emp_min_idx = np.argmin(binder)
    emp_beta = svendsen_beta[emp_min_idx]
    emp_min = binder[emp_min_idx]
    emp_max = np.max(binder)

    # Dynamic mask: isolate the absolute bottom X% of the dip channel
    threshold = emp_min + top_percent * (emp_max - emp_min)
    mask = binder < threshold

    x = svendsen_beta[mask]
    y = binder[mask]
    yerr = binder_SE[mask]

    # Generate a rock-solid dynamic initial guess based on empirical findings
    initial_guess = [emp_beta, 100000, emp_min]

    # Master Fit
    param_master, _ = curve_fit(quadratic, x, y, p0=initial_guess, maxfev=100000)
    a, b, c = param_master

    R2 = r2_score(y, quadratic(x, a, b, c))

    # =========================
    # Plot Master Fit with Mask Zoom
    # =========================
    plt.rc('font', size=12)
    plt.figure(figsize=(8, 5))

    # 1. Background context: full unmasked sweep data
    plt.plot(svendsen_beta, binder, '.', color='lightgray', markersize=2, label='Full Data Context')

    # 2. Highlighted target region: strictly the masked points fed to curve_fit
    plt.errorbar(x, y, yerr=yerr, fmt='o', color='royalblue', ecolor='yellow', 
                 markersize=4, label=f'Bottom {int(top_percent*100)}% Dip Data')

    # 3. Fit line trace traced natively over the active window
    cts = np.linspace(np.min(x), np.max(x), 1000)
    plt.plot(cts, quadratic(cts, a, b, c), 'r-', linewidth=2, label=f'Fit R²={R2:.4f}\n(a={a:.6f})')

    plt.title(f'Z2 LGT L={Spatial_Size} Binder Cumulant Dynamic Fit')
    plt.xlabel('Beta')
    plt.ylabel('Binder Cumulant')
    plt.legend()
    
    # Target axes boundaries tightly around the extracted fit window
    padding_x = (np.max(x) - np.min(x)) * 0.5
    plt.xlim(np.min(x) - padding_x, np.max(x) + padding_x)
    plt.ylim(emp_min - np.abs(emp_min)*0.005, threshold + np.abs(threshold)*0.01)
    plt.show()

    # =========================
    # Corrected Jackknife Parsing
    # =========================
    jk_filepath = folder + f'PT{Spatial_Size}^3*{Temporal_Size}binder_jackknife_samples.csv'

    with open(jk_filepath, 'r') as csvfile:
        header_line = csvfile.readline().strip().split(',')
    beta_jack = np.array([float(beta_val) for beta_val in header_line])

    jack = np.loadtxt(jk_filepath, delimiter=',', skiprows=1)

    critical_beta = []

    def jackknife_var(data_array, reference):
        N = len(data_array)
        return (N - 1) * np.mean((data_array - reference)**2)
    
    # Loop through the Jackknife blocks
    for i in range(jack.shape[0]):
        y_sample = jack[i, :]

        # Calculate sample-specific minimum boundaries for this block
        jk_min = np.min(y_sample)
        jk_max = np.max(y_sample)
        jk_threshold = jk_min + top_percent * (jk_max - jk_min)
        mask_j = y_sample < jk_threshold

        x_fit = beta_jack[mask_j]
        y_fit = y_sample[mask_j]

        if len(x_fit) < 5:
            continue

        try:
            # Seed the fit using our high-accuracy master fit states
            param_j, _ = curve_fit(quadratic, x_fit, y_fit, p0=[a, b, c], maxfev=100000)
            critical_beta.append(param_j[0])
        except RuntimeError:
            print(f"Size {Spatial_Size}: Jackknife curve {i} failed to converge.")

    # =========================
    # Bias Correction & Output Statistics
    # =========================
    critical_beta = np.array(critical_beta)
    Njack = len(critical_beta)
    
    if Njack > 0:
        beta_bar = np.mean(critical_beta)
        beta_bc = Njack * a - (Njack - 1) * beta_bar
        SE_beta = np.sqrt(jackknife_var(critical_beta,a))
        
        # print(f"{Spatial_Size}, {a},{SE_beta}")
        print(f"{Spatial_Size}, {beta_bc},{SE_beta}")
    else:
        print(f"Size: {Spatial_Size:2d} | Critical block calculation failed.")
    