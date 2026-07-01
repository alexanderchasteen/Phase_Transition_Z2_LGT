import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit 
from scipy.special import erf
from scipy.stats import chi2  # Imported for rigorous p-value calculation
from sklearn.metrics import r2_score

# Load data
df = pd.read_csv('SUSS.csv')
Lattice_Size = df['Lattice_Size'].to_numpy()
Critical_Beta = df['Critical_Beta'].to_numpy()
FWHM = df['FWHM'].to_numpy()
Peak = df['Peak'].to_numpy()
Critical_Beta_SD = df['Critical_Beta_SD'].to_numpy()
FWHM_SD = df['FWHM_SD'].to_numpy()
Peak_SD = df['Peak_SD'].to_numpy()

def linear(x, a, b):
    return a * x + b

volume = np.array([x**4 for x in Lattice_Size], dtype=float)
inverse_volume = np.array([1/x for x in volume], dtype=float)

# ==========================================
# Linear Fits Section
# ==========================================

# 1. Critical Beta (Linear)
param, pcov = curve_fit(linear, inverse_volume, Critical_Beta, sigma=Critical_Beta_SD, absolute_sigma=True)
fitA, fitB = param
perr = np.sqrt(np.diag(pcov))
print('--- Linear Critical Beta ---')
print('optimal parameters: slope, intercept', fitA, fitB)
print("Standard errors of parameters:", perr)
y_fit = linear(inverse_volume, fitA, fitB)

DOF = len(Critical_Beta) - 2
chi2_total = np.sum(((Critical_Beta - y_fit) / Critical_Beta_SD)**2)
chi_squared_red = chi2_total / DOF
p_val = chi2.sf(chi2_total, DOF)  # Compute p-value

R2 = round(r2_score(Critical_Beta, y_fit), 4)

plt.errorbar(
    inverse_volume, Critical_Beta,
    yerr=Critical_Beta_SD,
    fmt='o', capsize=16, elinewidth=4, markersize=20,
    color='teal', ecolor='gray',
    label=r'$\beta_c(V)$ computed'
)

fitA_exp = int(np.floor(np.log10(abs(fitA))))  
fitA_coef = fitA / 10**fitA_exp                

fit_label = (
    rf'Fit: $\beta_c = {fitA_coef:.3f} \times 10^{{{fitA_exp}}} (1/V) + {fitB:.7f}$' '\n'
    rf'$R^2 = {R2:.3f},\ \chi^2_{{\mathrm{{red}}}} = {chi_squared_red:.2f},\ p = {p_val:.3f}$'
)

plt.plot(inverse_volume, y_fit, linestyle='-', label=fit_label)
plt.xlabel('1/V', fontsize=32)
plt.ylabel(r'$\beta_c$', fontsize=32)
plt.title(r'Susceptibility: $\beta_c$ vs 1/V', fontsize=36, pad=10)
plt.grid(True, linestyle='--', alpha=0.6)
plt.xticks(fontsize=32)
plt.yticks(fontsize=32)
plt.legend(frameon=True, loc='best', fontsize=26)
plt.show()


# 2. FWHM (Linear)
param, pcov = curve_fit(linear, inverse_volume, FWHM, sigma=FWHM_SD, absolute_sigma=True)
fitA, fitB = param
perr = np.sqrt(np.diag(pcov))
print('\n--- Linear FWHM ---')
print("Standard errors of parameters:", perr)
y_fit = linear(inverse_volume, fitA, fitB)

DOF = len(FWHM) - 2
chi2_total = np.sum(((FWHM - y_fit) / FWHM_SD)**2)
chi_squared_red = chi2_total / DOF
p_val = chi2.sf(chi2_total, DOF)  # Compute p-value

R2 = round(r2_score(FWHM, y_fit), 4)

plt.errorbar(
    inverse_volume, FWHM,
    yerr=FWHM_SD,
    fmt='o', capsize=16, elinewidth=4, markersize=20,
    color='teal', ecolor='gray',
    label=r'FWHM(V) Computed'
)

fitA_exp = int(np.floor(np.log10(abs(fitA))))
fitA_coef = fitA / 10**fitA_exp

fit_label_fwhm = (
    rf'Fit: $\mathrm{{FWHM}} = {fitA_coef:.3f} \times 10^{{{fitA_exp}}} (1/V) + {fitB:.6f}$' '\n'
    rf'$R^2 = {R2:.3f},\ \chi^2_{{\mathrm{{red}}}} = {chi_squared_red:.2f},\ p = {p_val:.3f}$'
)

plt.plot(inverse_volume, y_fit, linestyle='-', label=fit_label_fwhm)
plt.xlabel('1/V', fontsize=32)
plt.ylabel('FWHM', fontsize=32)
plt.title('Susceptibility FWHM vs 1/V', fontsize=36, pad=10)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(frameon=True, loc='best', fontsize=26)
plt.xticks(fontsize=32)
plt.yticks(fontsize=32)
plt.show()


# 3. Peak (Linear)
param, pcov = curve_fit(linear, volume, Peak, sigma=Peak_SD, absolute_sigma=True)
fitA, fitB = param
perr = np.sqrt(np.diag(pcov))
print('\n--- Linear Peak ---')
print("Standard errors of parameters:", perr)
y_fit = linear(volume, fitA, fitB)

# Using volume lengths to protect DOF mapping integrity
DOF = len(Peak) - 2
chi2_total = np.sum(((Peak - y_fit) / Peak_SD)**2)
chi_squared_red = chi2_total / DOF
p_val = chi2.sf(chi2_total, DOF)  # Compute p-value

R2 = round(r2_score(Peak, y_fit), 4)

plt.errorbar(
    volume, Peak,
    yerr=Peak_SD,
    fmt='o', capsize=16, elinewidth=4, markersize=20,
    color='teal', ecolor='gray',
    label=r'Peak(V) Computed'
)

fitA_exp = int(np.floor(np.log10(abs(fitA))))
fitA_coef = fitA / 10**fitA_exp

fit_label_peak = (
    rf'Fit: $\mathrm{{Peak}} = {fitA_coef:.3f} \times 10^{{{fitA_exp}}} V + {fitB:.6f}$' '\n'
    rf'$R^2 = {R2:.3f},\ \chi^2_{{\mathrm{{red}}}} = {chi_squared_red:.2f},\ p = {p_val:.3f}$'
)

plt.plot(volume, y_fit, linestyle='-', label=fit_label_peak)
plt.xlabel('V', fontsize=32)
plt.ylabel('Susceptibility Peak', fontsize=32)
plt.title('Susceptibility Peak vs V (Linear)', fontsize=36, pad=10)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(frameon=True, loc='best', fontsize=26)
plt.xticks(fontsize=32)
plt.yticks(fontsize=32)
plt.show()


# ==========================================================
# Higher-Order / Non-Linear Fit Section (Quadratic & Custom)
# ==========================================================

def quadratic(x, a, b, c):
    return a*x*x + b*x + c

def peak_model(V, a, b, c):
    return a*V + b + c/V

# Refactored statistical engine to rigorously forward p-values
def fit_stats(y, yfit, yerr, n_params):
    chi2_val = np.sum(((y - yfit) / yerr)**2)
    dof = len(y) - n_params
    chi2_red = chi2_val / dof
    r2 = r2_score(y, yfit)
    p_val = chi2.sf(chi2_val, dof)  # Survival function evaluation
    return chi2_val, chi2_red, r2, p_val


# 1. Critical Beta vs 1/V (Quadratic)
popt_c, pcov_c = curve_fit(quadratic, inverse_volume, Critical_Beta, sigma=Critical_Beta_SD, absolute_sigma=True)
a_c, b_c, c_c = popt_c
perr_c = np.sqrt(np.diag(pcov_c))
yfit_c = quadratic(inverse_volume, a_c, b_c, c_c)

chi2_c, chi2red_c, r2_c, pval_c = fit_stats(Critical_Beta, yfit_c, Critical_Beta_SD, 3)

print("\n===== Critical Beta Quadratic Fit =====")
print("a, b, c =", a_c, b_c, c_c)
print("errors =", perr_c)
print("chi^2 =", chi2_c)
print("chi^2_red =", chi2red_c)
print("R^2 =", r2_c)
print("p-value =", pval_c)

plt.errorbar(
    inverse_volume, Critical_Beta, yerr=Critical_Beta_SD,
    fmt='o', capsize=16, elinewidth=4, markersize=20,
    color='teal', ecolor='gray', label=r'$\beta_c(V)$ computed'
)

fit_label_c = (
    rf'Fit: $\beta_c = {a_c:.3e}(1/V)^2 + {b_c:.3e}(1/V) + {c_c:.6f}$' '\n'
    rf'$R^2 = {r2_c:.3f},\ \chi^2_{{\mathrm{{red}}}} = {chi2red_c:.2f},\ p = {pval_c:.3f}$'
)

xfit = np.linspace(inverse_volume.min(), inverse_volume.max(), 500)
plt.plot(xfit, quadratic(xfit, a_c, b_c, c_c), linestyle='-', label=fit_label_c)
plt.xlabel('1/V', fontsize=32)
plt.ylabel(r'$\beta_c$', fontsize=32)
plt.title(r'Susceptibility: $\beta_c$ vs 1/V (Quad)', fontsize=36, pad=10)
plt.grid(True, linestyle='--', alpha=0.6)
plt.xticks(fontsize=32)
plt.yticks(fontsize=32)
plt.legend(frameon=True, loc='best', fontsize=24)
plt.show()


# 2. FWHM vs 1/V (Quadratic)
popt_f, pcov_f = curve_fit(quadratic, inverse_volume, FWHM, sigma=FWHM_SD, absolute_sigma=True)
a_f, b_f, c_f = popt_f
perr_f = np.sqrt(np.diag(pcov_f))
yfit_f = quadratic(inverse_volume, a_f, b_f, c_f)

chi2_f, chi2red_f, r2_f, pval_f = fit_stats(FWHM, yfit_f, FWHM_SD, 3)

print("\n===== FWHM Quadratic Fit =====")
print("a, b, c =", a_f, b_f, c_f)
print("errors =", perr_f)
print("chi^2 =", chi2_f)
print("chi^2_red =", chi2red_f)
print("R^2 =", r2_f)
print("p-value =", pval_f)

plt.errorbar(
    inverse_volume, FWHM, yerr=FWHM_SD,
    fmt='o', capsize=16, elinewidth=4, markersize=20,
    color='teal', ecolor='gray', label=r'FWHM(V) Computed'
)

fit_label_f = (
    rf'Fit: $\mathrm{{FWHM}} = {a_f:.3e}(1/V)^2 + {b_f:.3e}(1/V) + {c_f:.6f}$' '\n'
    rf'$R^2={r2_f:.3f},\ \chi^2_{{\mathrm{{red}}}}={chi2red_f:.2f},\ p = {pval_f:.3f}$'
)

plt.plot(xfit, quadratic(xfit, a_f, b_f, c_f), linestyle='-', label=fit_label_f)
plt.xlabel('1/V', fontsize=32)
plt.ylabel('FWHM', fontsize=32)
plt.title('Susceptibility FWHM vs 1/V (Quad)', fontsize=36, pad=10)
plt.grid(True, linestyle='--', alpha=0.6)
plt.xticks(fontsize=32)
plt.yticks(fontsize=32)
plt.legend(frameon=True, loc='best', fontsize=22)
plt.show()


# 3. Peak vs Volume (Custom Peak Model)
popt_p, pcov_p = curve_fit(peak_model, volume, Peak, sigma=Peak_SD, absolute_sigma=True)
a_p, b_p, c_p = popt_p
perr_p = np.sqrt(np.diag(pcov_p))
yfit_p = peak_model(volume, a_p, b_p, c_p)

chi2_p, chi2red_p, r2_p, pval_p = fit_stats(Peak, yfit_p, Peak_SD, 3)

print("\n===== Peak Fit =====")
print("a, b, c =", a_p, b_p, c_p)
print("errors =", perr_p)
print("chi^2 =", chi2_p)
print("chi^2_red =", chi2red_p)
print("R^2 =", r2_p)
print("p-value =", pval_p)

plt.errorbar(
    volume, Peak, yerr=Peak_SD,
    fmt='o', capsize=16, elinewidth=4, markersize=20,
    color='teal', ecolor='gray', label=r'Peak(V) Computed'
)

xfit_vol = np.linspace(volume.min(), volume.max(), 500)

fit_label_p = (
    rf'Fit: $\mathrm{{Peak}} = {a_p:.3e}V + {b_p:.3f} + {c_p:.3e}/V$' '\n'
    rf'$R^2 = {r2_p:.3f},\ \chi^2_{{\mathrm{{red}}}} = {chi2red_p:.2f},\ p = {pval_p:.3f}$'
)

plt.plot(xfit_vol, peak_model(xfit_vol, a_p, b_p, c_p), linestyle='-', label=fit_label_p)
plt.xlabel('V', fontsize=32)
plt.ylabel('Susceptibility Peak', fontsize=32)
plt.title('Susceptibility Peak vs V ', fontsize=36, pad=10)
plt.grid(True, linestyle='--', alpha=0.6)
plt.xticks(fontsize=32)
plt.yticks(fontsize=32)
plt.legend(frameon=True, loc='best', fontsize=22)
plt.show()