import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

# =========================
# Parameters
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
    # Plot raw Binder
    # =========================
    plt.rc('font', size=12)

    plt.errorbar(
        svendsen_beta,
        binder,
        yerr=binder_SE,
        fmt='o',
        color='blue',
        ecolor='yellow',
        markersize=1,
        label='Svendsen Data'
    )

    plt.title('Binder Cumulant vs Beta')
    plt.xlabel('Beta')
    plt.ylabel('Binder Cumulant')
    plt.legend()
    plt.show()

    # =========================
    # Model
    # =========================
    def quadratic(x, a, b, c):
        return b * (x - a)**2 + c

    # =========================
    # Fit window (tune per lattice size)
    # =========================
    if Spatial_Size == 3:
        beta_min = 0.423
        beta_max = 0.425
        initial_guess = [0.4355, 100000, 0.52]

    elif Spatial_Size == 4:
        beta_min = 0.435
        beta_max = 0.436
        initial_guess = [0.4355, 100000, 0.52]

    elif Spatial_Size == 5:   
        beta_min = 0.438
        beta_max = 0.439
        initial_guess = [0.4385, 100000, 0.52]

    elif Spatial_Size == 6:
        beta_min = 0.4395
        beta_max = 0.4398
        initial_guess = [0.4396, 100000, 0.52]

    elif Spatial_Size == 7:
        beta_min = 0.44005
        beta_max = 0.4402
        initial_guess = [0.4401, 100000, 0.52]

    elif Spatial_Size == 8:
        beta_min = 0.44025
        beta_max = 0.4405
        initial_guess = [0.44035, 100000, 0.52]

    elif Spatial_Size == 9:
        beta_min = 0.44044
        beta_max = 0.44055
        initial_guess = [0.4405, 100000, 0.52]

    elif Spatial_Size == 10:
        beta_min = 0.4405
        beta_max = 0.44062
        initial_guess = [0.44055, 100000, 0.52]

    # Mask for the central data fit
    mask = (svendsen_beta >= beta_min) & (svendsen_beta <= beta_max)

    x = svendsen_beta[mask]
    y = binder[mask]
    yerr = binder_SE[mask]

    # =========================
    # Master Fit
    # =========================
    param_master, _ = curve_fit(quadratic, x, y, p0=initial_guess, maxfev=100000)
    a, b, c = param_master

    R2 = r2_score(y, quadratic(x, a, b, c))

    # =========================
    # Plot fit
    # =========================
    cts = np.linspace(beta_min, beta_max, 1000)

    plt.rc('font', size=15)

    plt.errorbar(x, y, yerr=yerr,
                fmt='o', color='blue', ecolor='yellow',
                label='Data')

    plt.plot(cts, quadratic(cts, a, b, c),
            label=f'Fit R²={R2:.4f}', alpha=0.5)

    plt.title('Binder Cumulant Fit')
    plt.xlabel('Beta')
    plt.ylabel('Binder Cumulant')
    plt.legend()
    plt.show()

    # =========================
    # Corrected Jackknife Parsing
    # =========================
    jk_filepath = folder + f'PT{Spatial_Size}^3*{Temporal_Size}binder_jackknife_samples.csv'

    # 1. Read the correct beta array strings from the column headers
    with open(jk_filepath, 'r') as csvfile:
        header_line = csvfile.readline().strip().split(',')
    beta_jack = np.array([float(beta_val) for beta_val in header_line])

    # 2. Load the numerical sample array skipping the string header line
    jack = np.loadtxt(jk_filepath, delimiter=',', skiprows=1)

    critical_beta = []

    def jackknife_var(data):
        N = len(data)
        mean = np.mean(data)
        return (N - 1) * np.mean((data - mean)**2)

    # 3. Correct row iteration: run from index 0 to the end of the data rows
    for i in range(jack.shape[0]):
        y_sample = jack[i, :]

        # Enforce the master fit mask strictly to avoid window-drifting errors
        x_fit = beta_jack[mask]
        y_fit = y_sample[mask]

        if len(x_fit) < 5:
            continue

        try:
            # Use separate variable name 'param_j' to shield master parameter states
            param_j, _ = curve_fit(quadratic, x_fit, y_fit, p0=[a, b, c], maxfev=100000)
            critical_beta.append(param_j[0])
        except RuntimeError:
            print(f"Size {Spatial_Size}: Jackknife curve {i} failed to converge.")

    # Calculate errors and print metrics
    SE_beta = np.sqrt(jackknife_var(critical_beta))
    # print(f"{size},{a},{SE_beta}")

    # =========================
    # Bias Correction (Optional Active Code Block)
    # =========================
    critical_beta = np.array(critical_beta)
    Njack = len(critical_beta)
    beta_bar = np.mean(critical_beta)
    beta_bc = Njack * a - (Njack - 1) * beta_bar
    
    print(f"{size},{beta_bc}, {SE_beta}" )

