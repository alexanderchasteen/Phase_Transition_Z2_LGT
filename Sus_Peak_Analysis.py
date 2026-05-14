import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit 
from scipy.special import erf
from sklearn.metrics import r2_score


df=pd.read_csv('Sus_peak_analysis copy.csv')
Lattice_Size = df['Lattice_Size'].to_numpy()
Critical_Beta = df['Critical_Beta'].to_numpy()
FWHM = df['FWHM'].to_numpy()
Peak = df['Peak'].to_numpy()
Critical_Beta_SD = df['Critical_Beta_SD'].to_numpy()
FWHM_SD = df['FWHM_SD'].to_numpy()
Peak_SD = df['Peak_SD'].to_numpy()



def linear(x,a,b):
    return a*x+b



volume = np.array([x**4 for x in Lattice_Size], dtype=float)
inverse_volume = np.array([1/x for x in volume], dtype=float)




param,pcov=curve_fit(linear,inverse_volume, Critical_Beta,sigma=Critical_Beta_SD, absolute_sigma=True )

fitA,fitB=param

perr = np.sqrt(np.diag(pcov))
print('optimal paramters: slope,intercept', fitA,fitB)
print("Standard errors of parameters:", perr)
y_fit=linear(inverse_volume,fitA,fitB)

chi_squared = np.sum(((Critical_Beta - y_fit) / Critical_Beta_SD)**2)

R2 = r2_score(Critical_Beta, linear(inverse_volume, fitA, fitB))
R2 = round(R2, 4)

plt.errorbar(
    inverse_volume, Critical_Beta,
    yerr=Critical_Beta_SD,
    fmt='o',
   capsize=16,
    elinewidth=4,
    markersize=20,
    color='teal',
    ecolor='gray',
    label=r'$\beta_c(V)$ computed'
)

fitA_exp = int(np.floor(np.log10(abs(fitA))))  # exponent
fitA_coef = fitA / 10**fitA_exp                # coefficient

fit_label = (
    rf'Fit: $\beta_c = {fitA_coef:.3f} \times 10^{{{fitA_exp}}} (1/V) + {fitB:.6f}$' '\n'
    rf'$R^2 = {R2:.3f},\ \chi^2 = {chi_squared:.2f}$'
)

plt.plot(
    inverse_volume,
    linear(inverse_volume, fitA, fitB),
    linestyle='-',
    label=fit_label
)

plt.xlabel('1/V', fontsize=32)
plt.ylabel(r'$\beta_c$', fontsize=32)
plt.title(r'Susceptibility: $\beta_c$ vs 1/V', fontsize=36, pad=10)
plt.grid(True, linestyle='--', alpha=0.6)
plt.xticks(fontsize=32)
plt.yticks(fontsize=32)
plt.legend(frameon=True, loc='best', fontsize=32)
plt.show()






param,pcov=curve_fit(linear,inverse_volume, FWHM,sigma=FWHM_SD, absolute_sigma=True)
fitA,fitB=param



perr = np.sqrt(np.diag(pcov))
print("Standard errors of parameters:", perr)
y_fit=linear(inverse_volume,fitA,fitB)

print(y_fit)


chi_squared = np.sum(((FWHM - y_fit) / FWHM_SD)**2)

R2 = r2_score(FWHM, linear(inverse_volume, fitA, fitB))
R2 = round(R2, 4)

plt.errorbar(
    inverse_volume, FWHM,
    yerr=FWHM_SD,
    fmt='o',
    capsize=16,
    elinewidth=4,
    markersize=20,
    color='teal',
    ecolor='gray',
    label=r'FWHM(V) Computed'
)

plt.plot(
    inverse_volume,
    linear(inverse_volume, fitA, fitB),
    linestyle='-',
    label=(
        rf'$R^2={R2}$, $\chi^2={chi_squared:.2f}$'
    )
)

plt.xlabel('1/V', fontsize=32)
plt.ylabel('FWHM', fontsize=32)
plt.title('Susceptibility FWHM vs 1/V', fontsize=36, pad=10)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(frameon=True, loc='best', fontsize=32)
plt.xticks(fontsize=32)
plt.yticks(fontsize=32)
plt.show()



param,pcov=curve_fit(linear,volume, Peak)
fitA,fitB=param
perr = np.sqrt(np.diag(pcov))
print("Standard errors of parameters:", perr)
y_fit=linear(volume,fitA,fitB)

print(y_fit)


chi_squared = np.sum(((Peak - y_fit) / Peak_SD)**2)

R2=r2_score(Peak,linear(volume,fitA,fitB))
R2=round(R2,4)

plt.errorbar(
    volume, Peak,
    yerr=Peak_SD,
    fmt='o',
   capsize=16,
    elinewidth=4,
    markersize=20,
    color='teal',
    ecolor='gray',
    label=r'Peak(V) Computed'
)

plt.plot(
    volume,
    linear(volume, fitA, fitB),
    linestyle='-',
    label=(
        rf'$R^2={R2}$, $\chi^2={chi_squared:.2f}$'
    )
)

plt.xlabel('V', fontsize=32)
plt.ylabel('Susceptibility Peak', fontsize = 32)
plt.title('Susceptibility Peak vs V', fontsize=36, pad=10)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(frameon=True, loc='best',fontsize=32)
plt.xticks(fontsize=32)
plt.yticks(fontsize=32)
plt.show()



#################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

# =========================
# Load data
# =========================
df = pd.read_csv('Sus_peak_analysis copy.csv')

Lattice_Size = df['Lattice_Size'].to_numpy()

Critical_Beta = df['Critical_Beta'].to_numpy()
FWHM = df['FWHM'].to_numpy()
Peak = df['Peak'].to_numpy()

Critical_Beta_SD = df['Critical_Beta_SD'].to_numpy()
FWHM_SD = df['FWHM_SD'].to_numpy()
Peak_SD = df['Peak_SD'].to_numpy()

# =========================
# Models
# =========================
def linear(x, a, b):
    return a*x + b

def quadratic(x, a, b, c):
    return a*x*x + b*x + c

def peak_model(V, a, b, c):
    return a*V + b + c/V

# =========================
# Derived variables
# =========================
volume = Lattice_Size.astype(float)**4
inverse_volume = 1.0 / volume


# ==========================================================
# Helper: fit + report stats
# ==========================================================
def fit_stats(y, yfit, yerr, n_params):
    chi2 = np.sum(((y - yfit) / yerr)**2)
    dof = len(y) - n_params
    chi2_red = chi2 / dof
    r2 = r2_score(y, yfit)
    return chi2, chi2_red, r2


# =========================
# 1. Critical Beta vs 1/V (Linear)
# =========================
popt_c, pcov_c = curve_fit(
    linear,
    inverse_volume,
    Critical_Beta,
    sigma=Critical_Beta_SD,
    absolute_sigma=True
)

a_c, b_c = popt_c
perr_c = np.sqrt(np.diag(pcov_c))
yfit_c = linear(inverse_volume, a_c, b_c)

chi2_c, chi2red_c, r2_c = fit_stats(
    Critical_Beta,
    yfit_c,
    Critical_Beta_SD,
    2
)

print("\n===== Critical Beta Fit =====")
print("a, b =", a_c, b_c)
print("errors =", perr_c)
print("chi^2 =", chi2_c)
print("chi^2_red =", chi2red_c)
print("R^2 =", r2_c)

plt.errorbar(
    inverse_volume,
    Critical_Beta,
    yerr=Critical_Beta_SD,
    fmt='o',
    capsize=16,
    elinewidth=4,
    markersize=20,
    color='teal',
    ecolor='gray',
    label=r'$\beta_c(V)$ computed'
)

fit_label = (
    rf'Fit: $\beta_c = {a_c:.3e}(1/V) + {b_c:.6f}$' '\n'
    rf'$R^2 = {r2_c:.3f},\ \chi^2 = {chi2_c:.2f}$'
)

xfit = np.linspace(
    inverse_volume.min(),
    inverse_volume.max(),
    500
)

plt.plot(
    xfit,
    linear(xfit, a_c, b_c),
    linestyle='-',
    label=fit_label
)

plt.xlabel('1/V', fontsize=32)
plt.ylabel(r'$\beta_c$', fontsize=32)
plt.title(r'Susceptibility: $\beta_c$ vs 1/V', fontsize=36, pad=10)
plt.grid(True, linestyle='--', alpha=0.6)
plt.xticks(fontsize=32)
plt.yticks(fontsize=32)
plt.legend(frameon=True, loc='best', fontsize=32)
plt.show()


# =========================
# 2. FWHM vs 1/V (Quadratic)
# =========================
popt_f, pcov_f = curve_fit(
    quadratic,
    inverse_volume,
    FWHM,
    sigma=FWHM_SD,
    absolute_sigma=True
)

a_f, b_f, c_f = popt_f
perr_f = np.sqrt(np.diag(pcov_f))

yfit_f = quadratic(inverse_volume, a_f, b_f, c_f)

chi2_f, chi2red_f, r2_f = fit_stats(
    FWHM,
    yfit_f,
    FWHM_SD,
    3
)

print("\n===== FWHM Quadratic Fit =====")
print("a, b, c =", a_f, b_f, c_f)
print("errors =", perr_f)
print("chi^2 =", chi2_f)
print("chi^2_red =", chi2red_f)
print("R^2 =", r2_f)

plt.errorbar(
    inverse_volume,
    FWHM,
    yerr=FWHM_SD,
    fmt='o',
    capsize=16,
    elinewidth=4,
    markersize=20,
    color='teal',
    ecolor='gray',
    label=r'FWHM(V) Computed'
)

xfit = np.linspace(
    inverse_volume.min(),
    inverse_volume.max(),
    500
)

fit_label = (
    rf'Fit: $a(1/V)^2 + b(1/V) + c$' '\n'
    rf'$a={a_f:.3e},\ b={b_f:.3e},\ c={c_f:.3e}$' '\n'
    rf'$R^2={r2_f:.3f},\ \chi^2={chi2_f:.2f}$'
)

plt.plot(
    xfit,
    quadratic(xfit, a_f, b_f, c_f),
    linestyle='-',
    label=fit_label
)

plt.xlabel('1/V', fontsize=32)
plt.ylabel('FWHM', fontsize=32)
plt.title('Susceptibility FWHM vs 1/V', fontsize=36, pad=10)
plt.grid(True, linestyle='--', alpha=0.6)
plt.xticks(fontsize=32)
plt.yticks(fontsize=32)
plt.legend(frameon=True, loc='best', fontsize=24)
plt.show()


# =========================
# 3. Peak vs Volume
# =========================
popt_p, pcov_p = curve_fit(
    peak_model,
    volume,
    Peak,
    sigma=Peak_SD,
    absolute_sigma=True
)

a_p, b_p, c_p = popt_p
perr_p = np.sqrt(np.diag(pcov_p))
yfit_p = peak_model(volume, a_p, b_p, c_p)

chi2_p, chi2red_p, r2_p = fit_stats(
    Peak,
    yfit_p,
    Peak_SD,
    3
)

print("\n===== Peak Fit =====")
print("a, b, c =", a_p, b_p, c_p)
print("errors =", perr_p)
print("chi^2 =", chi2_p)
print("chi^2_red =", chi2red_p)
print("R^2 =", r2_p)

plt.errorbar(
    volume,
    Peak,
    yerr=Peak_SD,
    fmt='o',
    capsize=16,
    elinewidth=4,
    markersize=20,
    color='teal',
    ecolor='gray',
    label=r'Peak(V) Computed'
)

xfit = np.linspace(
    volume.min(),
    volume.max(),
    500
)

plt.plot(
    xfit,
    peak_model(xfit, a_p, b_p, c_p),
    linestyle='-',
    label=rf'$R^2={r2_p:.3f},\ \chi^2={chi2_p:.2f}$'
)

plt.xlabel('V', fontsize=32)
plt.ylabel('Susceptibility Peak', fontsize=32)
plt.title('Susceptibility Peak vs V', fontsize=36, pad=10)
plt.grid(True, linestyle='--', alpha=0.6)
plt.xticks(fontsize=32)
plt.yticks(fontsize=32)
plt.legend(frameon=True, loc='best', fontsize=32)
plt.show()