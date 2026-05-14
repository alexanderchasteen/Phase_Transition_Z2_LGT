import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit 
from scipy.special import erf
from sklearn.metrics import r2_score  

Spatial_Size=3
Temporal_Size=Spatial_Size

folder = './Multi_Canon_Results/'
# folder = './Analyzed_Data/'

df=pd.read_csv(folder+'PT'+str(Spatial_Size)+'^3*'+str(Temporal_Size)+'_analysis.csv')

beta_array=df['Beta'].to_numpy()
avg_plaq=df['Avg Plaq'].to_numpy()
plaq_sus=df['Plaq Sus'].to_numpy()
SE_avg_plaq=df['Jackknife Action SE'].to_numpy()
SE_plaq_sus=df['Jackknife Sus SE'].to_numpy()

df2=pd.read_csv(folder+'PT'+str(Spatial_Size)+'^3*'+str(Temporal_Size)+'_svendsen_analysis.csv')

svendsen_beta_array=df2['Beta'].to_numpy()
svendsen_plaq_array=df2['Avg Plaq'].to_numpy()          
svendsen_sus_array=df2['Plaq Sus'].to_numpy()
svendsen_plaq_SE_array=df2['Jackknife Action SE'].to_numpy()
svendsen_sus_SE_array=df2['Jackknife Sus SE'].to_numpy()    

beta = df2['Beta'].to_numpy()
binder = df2['Binder Cum'].to_numpy()
binder_SE = df2['Binder Cum SE'].to_numpy()

plt.rc('font', size=12)

plt.errorbar(
    svendsen_beta_array,
    binder,
    yerr=binder_SE,
    fmt='o',
    label='Svendsen Reweighted Data',
    color='blue',
    ecolor='yellow',
    elinewidth=3,
    capsize=0,
    markersize=1
)

plt.title('Binder Cumulant vs Beta')
plt.xlabel('Beta')
plt.ylabel('Binder Cumulant')
plt.show()

min_idx = np.argmin(binder)
beta_min_discrete = beta[min_idx]
binder_min_discrete = binder[min_idx]
binder_err_at_min = binder_SE[min_idx]

print(f"Discrete Minimum: Binder = {binder_min_discrete:.6f} at Beta = {beta_min_discrete}")

jacknife_data_array = np.loadtxt(folder+'PT'+str(Spatial_Size)+'^3*'+str(Temporal_Size)+'binder_jackknife_samples.csv', delimiter=',')
beta_array_jacknife=jacknife_data_array[0,:]

svendsen_beta_array_store = svendsen_beta_array   

# Lattice Size 3 best params
# beta_max=0.426
# beta_min=0.42
# initial_guess=[0.424, 1, .52]


# Lattice Size 4 opt params

beta_max=0.436
beta_min=0.435
initial_guess=[0.4355, 100000, .52]

#Lattice Size 5 opt params
# beta_max=0.439
# beta_min=0.4386
# initial_guess=[0.4388, 60000, .52]

#Lattice Size 6 opt params
# beta_max=0.43975
# beta_min=0.4395
# initial_guess=[0.4396, 60000, .52]

mask = (svendsen_beta_array >= beta_min) & (svendsen_beta_array <= beta_max)

svendsen_beta_array     = svendsen_beta_array[mask]
binder = binder[mask]
binder_SE  = binder_SE[mask]
beta_array_jacknife = beta_array_jacknife[mask]
jacknife_data_array = jacknife_data_array[:, mask]  

def qaudratic(x, a, b, c):
    return b*(x-a)*(x-a)+c  

param,_=curve_fit(qaudratic,svendsen_beta_array, binder,p0 = initial_guess)
fitA,fitB,fitC=param

R2=r2_score(binder,qaudratic(svendsen_beta_array,fitA,fitB,fitC))
R2=round(R2,4)

a=fitA
b=fitB
c=fitC

cts_beta=np.arange(beta_min,beta_max,(beta_max-beta_min)/1000)
fitY=qaudratic(cts_beta,a,b,c)

plt.rc('font', size=15)
plt.errorbar(svendsen_beta_array,binder,yerr=binder_SE,fmt='o',label='Svendsen Reweighted Data',color='blue',ecolor='yellow',elinewidth=3,capsize=0,markersize=1)
plt.plot(cts_beta, fitY, '-', label='Lorentzian Fit R^2='+str(R2),color='blue',alpha=0.5)
plt.legend()
plt.title('Binder Cum vs Beta')
plt.xlabel('Beta')
plt.ylabel('Binder Cum')  
plt.show()

# --- NEW: adaptive jackknife setup ---
initial_guess_jack = [a, b, c]  # better initial guess
delta = (beta_max - beta_min) / 2  # adaptive window size

critical_beta_array = []

for i in range(1, jacknife_data_array.shape[0]):

    y = jacknife_data_array[i,:]

    # --- NEW: find minimum per sample ---
    min_idx = np.argmin(y)
    beta_peak = beta_array_jacknife[min_idx]

    # --- NEW: adaptive window ---
    mask_jack = (beta_array_jacknife >= beta_peak - delta) & \
                (beta_array_jacknife <= beta_peak + delta)

    x_fit = beta_array_jacknife[mask_jack]
    y_fit = y[mask_jack]

    # --- NEW: stability check ---
    if len(x_fit) < 5:
        continue

    param,_=curve_fit(qaudratic,x_fit,y_fit,p0=initial_guess_jack,maxfev=100000)

    fitA,fitB,fitC=param
    critical_beta_array.append(fitA)

def jackknife_variance(data):
    N = len(data)
    mean = np.mean(data)
    return (N - 1) * np.mean((data - mean)**2)

SE_critical_beta=np.sqrt(jackknife_variance(critical_beta_array))

print('Critical Beta:', a, '+/-', SE_critical_beta)

svendsen_beta_array = svendsen_beta_array_store         

jacknife_data_array = np.loadtxt(folder+'PT'+str(Spatial_Size)+'^3*'+str(Temporal_Size)+'sus_jackknife_samples.csv', delimiter=',')
beta_array_jacknife=jacknife_data_array[0,:]

plt.rc('font', size=12)
plt.errorbar(beta_array,avg_plaq,yerr=SE_avg_plaq,fmt='o',label='Parallel Tempering Data',color='blue',ecolor='red',elinewidth=3)
plt.errorbar(svendsen_beta_array,svendsen_plaq_array,yerr=svendsen_plaq_SE_array,fmt='o',label='Svendsen Reweighted Data',color='blue',ecolor='yellow',elinewidth=3,capsize=0,markersize=1)
plt.title('Average Plaquette vs Beta')
plt.xlabel('Beta')
plt.ylabel('Average Plaquette')
plt.show()

plt.errorbar(svendsen_beta_array,svendsen_sus_array,yerr=svendsen_sus_SE_array,fmt='o',label='Svendsen Reweighted Data',color='blue',ecolor='yellow',elinewidth=3,capsize=0,markersize=1)
plt.errorbar(beta_array,plaq_sus,yerr=SE_plaq_sus,fmt='o',label='Simulation Data',color='blue',ecolor='red',elinewidth=3)
plt.title('Plaquette Susceptibility vs Beta')
plt.xlabel('Beta')
plt.ylabel('Plaquette Susceptibility')
plt.legend()
plt.show()

# # Lattice size 3 best parms sus
beta_max=0.434
beta_min=0.428
initial_guess=[0.432, 0.02, 21]

# Lattice size 4 opt params
# beta_max=0.438
# beta_min=0.437
# initial_guess=[0.437, 0.005, 60]

# # # # Optimal paramters for 5
# beta_max=0.4397
# beta_min=0.439
# initial_guess=[0.4392, 0.005, 165]

# #opt params for 6
# beta_max=0.4401
# beta_min=0.43975
# initial_guess=[0.44, 0.005, 300]

mask = (svendsen_beta_array >= beta_min) & (svendsen_beta_array <= beta_max)

svendsen_beta_array     = svendsen_beta_array[mask]
svendsen_sus_array      = svendsen_sus_array[mask]
svendsen_sus_SE_array  = svendsen_sus_SE_array[mask]
beta_array_jacknife = beta_array_jacknife[mask]
jacknife_data_array = jacknife_data_array[:, mask]  

def lorentzian(x, x0, gamma, a):
    return a * gamma**2 / ((x - x0)**2 + gamma**2)  

param,_=curve_fit(lorentzian,svendsen_beta_array, svendsen_sus_array,p0 = initial_guess)
fitA,fitB,fitC=param

R2=r2_score(svendsen_sus_array,lorentzian(svendsen_beta_array,fitA,fitB,fitC))
R2=round(R2,4)

x0=fitA
gamma=fitB
a=fitC  

initial_guess=[x0, gamma, a]

cts_beta=np.arange(beta_min,beta_max,(beta_max-beta_min)/1000)
fitY=lorentzian(cts_beta,fitA,fitB,fitC)

plt.rc('font', size=15)
plt.errorbar(svendsen_beta_array,svendsen_sus_array,yerr=svendsen_sus_SE_array,fmt='o',label='Svendsen Reweighted Data',color='blue',ecolor='yellow',elinewidth=3,capsize=0,markersize=1)
plt.errorbar(beta_array,plaq_sus,yerr=SE_plaq_sus,fmt='o',label='Parallel Tempering Data',color='blue',ecolor='red',elinewidth=3)
plt.plot(cts_beta, fitY, '-', label='Lorentzian Fit R^2='+str(R2),color='blue',alpha=0.5)
plt.legend()
plt.title('Plaquette Susceptibility vs Beta')
plt.xlabel('Beta')
plt.ylabel('Plaquette Susceptibility')  
plt.show()

# --- NEW: adaptive jackknife setup ---
initial_guess_jack = [x0, gamma, a]
delta = (beta_max - beta_min) / 2

critical_beta_array = []
FWHM_array = []
peak_height_array = []

for i in range(1, jacknife_data_array.shape[0]):

    y = jacknife_data_array[i,:]

    # --- NEW: find peak per sample ---
    peak_idx = np.argmax(y)
    beta_peak = beta_array_jacknife[peak_idx]

    # --- NEW: adaptive window ---
    mask_jack = (beta_array_jacknife >= beta_peak - delta) & \
                (beta_array_jacknife <= beta_peak + delta)

    x_fit = beta_array_jacknife[mask_jack]
    y_fit = y[mask_jack]

    # --- NEW: stability check ---
    if len(x_fit) < 5:
        continue

    param,_=curve_fit(lorentzian,x_fit,y_fit,p0=initial_guess_jack,maxfev=10000)

    fitA,fitB,fitC=param

    critical_beta_array.append(fitA)
    FWHM_array.append(2*fitB)
    peak_height_array.append(fitC)

SE_critical_beta=np.sqrt(jackknife_variance(critical_beta_array))
SE_FWHM=np.sqrt(jackknife_variance(FWHM_array))
SE_peak_height=np.sqrt(jackknife_variance(peak_height_array))

print('Critical Beta:', x0, '+/-', SE_critical_beta)
print('FWHM:', 2*gamma, '+/-', SE_FWHM)
print('Peak Height:', a, '+/-', SE_peak_height)

print(str(Spatial_Size)+'^3*'+str(Temporal_Size),',',x0,',',2*gamma,',',a,',', SE_critical_beta,',', SE_FWHM,',', SE_peak_height)