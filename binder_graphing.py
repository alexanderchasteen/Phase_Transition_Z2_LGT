import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

# =========================
# Parameters
# =========================
Spatial_Size = 3
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
beta_min = 0.423
beta_max = 0.425
initial_guess = [0.4355, 100000, 0.52]

beta_min = 0.435
beta_max = 0.436
initial_guess = [0.4355, 100000, 0.52]


beta_min = 0.438
beta_max = 0.439
initial_guess = [0.4385, 100000, 0.52]

beta_min = 0.4394
beta_max = 0.4398
initial_guess = [0.4396, 100000, 0.52]

beta_min = 0.44
beta_max = 0.4403
initial_guess = [0.4401, 100000, 0.52]

beta_min = 0.4402
beta_max = 0.4405
initial_guess = [0.4403, 100000, 0.52]


beta_min = 0.4404
beta_max = 0.4406
initial_guess = [0.4405, 100000, 0.52]

beta_min = 0.44045
beta_max = 0.44065
initial_guess = [0.4406, 100000, 0.52]


mask = (svendsen_beta >= beta_min) & (svendsen_beta <= beta_max)

x = svendsen_beta[mask]
y = binder[mask]
yerr = binder_SE[mask]

# =========================
# Fit
# =========================
param, _ = curve_fit(quadratic, x, y, sigma=yerr,absolute_sigma=True, p0=initial_guess)
a, b, c = param

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
# Jackknife
# =========================
jack = np.loadtxt(
    folder + f'PT{Spatial_Size}^3*{Temporal_Size}binder_jackknife_samples.csv',
    delimiter=','
)

beta_jack = jack[0, :]
delta = (beta_max - beta_min) / 2

critical_beta = []

def jackknife_var(data):
    N = len(data)
    mean = np.mean(data)
    return (N - 1) * np.mean((data - mean)**2)

for i in range(1, jack.shape[0]):

    y_sample = jack[i, :]

    min_idx = np.argmin(y_sample)
    beta_peak = beta_jack[min_idx]

    mask_j = (beta_jack >= beta_peak - delta) & (beta_jack <= beta_peak + delta)

    x_fit = beta_jack[mask_j]
    y_fit = y_sample[mask_j]

    if len(x_fit) < 5:
        continue

    param, _ = curve_fit(quadratic, x_fit, y_fit, p0=param, maxfev=100000)

    critical_beta.append(param[0])

SE_beta = np.sqrt(jackknife_var(critical_beta))

print("Binder critical beta:", a, ",", SE_beta)