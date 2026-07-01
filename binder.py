import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit 
from sklearn.metrics import r2_score

# 1. Load Data
df = pd.read_csv('binder.csv')
Lattice_size = df['lattice_size'].to_numpy()
beta_min_data = df['binder_min'].to_numpy()
beta_error = df['Error'].to_numpy()

# 2. Define Fitting Model
def linear_fit(x, a, b):
    return a * x + b

# 3. Data Transformation
volume = np.array([x**4 for x in Lattice_size], dtype=float)
inverse_volume = 1.0 / volume

# 4. Perform Weighted Curve Fit
params, pcov = curve_fit(
    linear_fit, 
    inverse_volume, 
    beta_min_data, 
    sigma=beta_error, 
    absolute_sigma=True
)

fit_slope, fit_intercept = params
perr_raw = np.sqrt(np.diag(pcov)) 

# --- NEW: CALCULATE CHI-SQUARE STATISTICS ---
y_fit = linear_fit(inverse_volume, *params)
residuals = beta_min_data - y_fit

# Chi-Square formula: sum( (data - fit)^2 / error^2 )
chi_square = np.sum((residuals / beta_error)**2)
dof = len(beta_min_data) - len(params)
reduced_chi_square = chi_square / dof

# Correction for small errors/bad fit:
# If reduced chi-square is high, we scale the parameter errors by sqrt(chi_red)
error_scaling = np.sqrt(reduced_chi_square) if reduced_chi_square > 1 else 1.0
perr_corrected = perr_raw * error_scaling

r2 = r2_score(beta_min_data, y_fit)

# 5. Output Results
print("-" * 40)
print(f"FIT RESULTS (Extrapolation to 1/V -> 0)")
print(f"Infinite Vol Beta_c: {fit_intercept} +/- {perr_corrected[1]:.6f}")
print(f"Scaling Slope (a):   {fit_slope:.6f} +/- {perr_corrected[0]:.6f}")
print("-" * 40)
print(f"STATISTICS")
print(f"Chi-Square:          {chi_square:.4f}")
print(f"Degrees of Freedom:  {dof}")
print(f"Reduced Chi-Square:  {reduced_chi_square:.4f}")
print(f"R-squared:           {r2:.4f}")

if reduced_chi_square > 2:
    print("\nWARNING: Reduced Chi-Square is high!")
    print(f"Errors have been scaled up by factor of {error_scaling:.2f}")
print("-" * 40)


plt.errorbar(
    inverse_volume, beta_min_data,
    yerr=beta_error,
    fmt='o',
     capsize=16,
    elinewidth=4,
    markersize=20,
    color='teal',
    ecolor='gray',
    label=r'$\beta_c(V)$ computed'
)

fitA_exp = int(np.floor(np.log10(abs(fit_slope)))) if fit_slope != 0 else 0
fitA_coef = fit_slope / 10**fitA_exp if fit_slope != 0 else 0

fit_label = (
    rf'Fit: $\beta_c = {fitA_coef:.3f} \times 10^{{{fitA_exp}}} (1/|\Lambda|) + {fit_intercept:.6f}$' '\n'
    rf'$R^2 = {r2:.3f},\ \chi^2_{{\mathrm{{red}}}} ={reduced_chi_square:.4f}$'
)

x_range = np.linspace(0, np.max(inverse_volume) * 1.1, 100)

plt.plot(
    x_range,
    linear_fit(x_range, *params),
    linestyle='-',
    label=fit_label
)

plt.xlabel('1/V', fontsize=32)
plt.ylabel(r'$\beta_c$', fontsize=32)
plt.title(r'Binder cummulant: $\beta_c$ vs 1/V', fontsize=36, pad=10)

plt.grid(True, linestyle='--', alpha=0.6)

plt.legend(frameon=True, loc='best', fontsize=32)
plt.xticks(fontsize = 32)
plt.yticks(fontsize = 32)
plt.tight_layout()
plt.show()