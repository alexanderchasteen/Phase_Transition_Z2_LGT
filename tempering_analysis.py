import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv

Spatial_Size = 3
Temporal_Size = Spatial_Size
Nplaq = Spatial_Size**3 * Temporal_Size * 6
folder = './Multi_Canon_Results/'

df = pd.read_csv('/home/alexa/Z2_Lattice_Gauge_Theory/Z2_CPP_NEW/Raw_MC_Data_3^3_3 (2).csv')

# Beta array
beta_array = df.columns.to_numpy(dtype=float)
CONFIGS = len(beta_array)
beta_nbhd_radius = (beta_array[1] - beta_array[0])/2

blocks = 50

# -------------------
# Simple Weights Class
# -------------------
class SimpleWeights:
    def __init__(self, filename):
        data = np.loadtxt(filename)
        self.plaq_values = data[:, 0]
        self.weights = data[:, 1]
        self.n_bins = len(self.plaq_values)

    def find_bin(self, value):
        idx = np.argmin(np.abs(self.plaq_values - value))
        return idx

    def get_weight(self, value):
        idx = self.find_bin(value)
        return self.weights[idx]

# Load weights TURN ONE OF THEM ON OR OFF

weights = SimpleWeights("/home/alexa/Z2_Lattice_Gauge_Theory/Z2_CPP_NEW/weight_function_normalized.txt")
# weights = SimpleWeights("/home/alexa/Z2_Lattice_Gauge_Theory/Z2_CPP_NEW/No_weight.txt")

# -------------------
# Blocking / Jackknife Functions
# -------------------
def jackknife(array, function):
    array = np.asarray(array)
    n = len(array)
    theta_hat = function(array)
    jackknife_thetas = np.array([function(np.delete(array,i)) for i in range(n)])
    theta_dot = np.mean(jackknife_thetas)
    var_jack = (n-1)/n * np.sum((jackknife_thetas - theta_dot)**2)
    theta_jack = n*theta_hat - (n-1)*theta_dot
    return theta_hat, var_jack, theta_jack

def svendsen_blocking_SPECIAL(array, blocks, weights_array):
    block_size = len(array)//blocks
    blocked_array = []
    for i in range(blocks):
        start = i*block_size
        end = start+block_size
        block = np.array(array[start:end])
        block_weights = np.array(weights_array[start:end])
        block_stat = np.sum(block*block_weights)/np.sum(block_weights)
        blocked_array.append(block_stat)
    return np.array(blocked_array)

def plaq_susceptibility(array):
    return Nplaq * np.var(array)

def fourth_moment(array):
    m = np.mean(array)
    return np.mean((array - m)**4)

def binder_cum(array):
    fourth_mom = fourth_moment(array)
    second_mom = np.var(array)
    return 1 - fourth_mom/(3*second_mom**2)

# -------------------
# Prepare storage arrays
# -------------------
avg_plaq = []
SE_avg_plaq = []
plaq_sus = []
SE_plaq_sus = []
binder = []
SE_Binder = []

svendsen_beta_array = []
svendsen_avg_plaq_array = []
svendsen_avg_plaq_SE_array = []
svendsen_sus_array = []
svendsen_sus_SE_array = []
svendsen_binder = []
svendsen_binder_SE_array = []

sus_jackknife_samples = []
binder_jackknife_samples = []

# -------------------
# Main Analysis Loop
# -------------------
for i in range(CONFIGS):
    print(i)
    array = df.iloc[:,i].to_numpy(dtype=float)

    # Define beta neighborhood
    k_array = np.linspace(-beta_nbhd_radius, beta_nbhd_radius, 10)

    for k in k_array:
        print(k)
        # new_beta=beta_array[i]+k
        # svendsen_beta_array.append(new_beta)
        # # compute weights
        # delta_beta=k
        # logw_array = delta_beta * Nplaq * array
        # maxlogw = np.max(logw_array)
        # w_array = np.exp(np.clip(logw_array - maxlogw, -700, 700))  # prevent overflow




        new_beta = beta_array[i] + k
        svendsen_beta_array.append(new_beta)
        # Compute reweight factors
        reweight_factors = []
        for config in array:
            current_avg_plaq = config
            beta_shift = k * Nplaq * current_avg_plaq
            bias_unundo = -weights.get_weight(current_avg_plaq) 
            exponent = beta_shift + bias_unundo
            reweight_factors.append(np.exp(np.clip(exponent, -700, 700)))
        w_array = np.array(reweight_factors)
        
        # Average plaquette
        sv_block_P = svendsen_blocking_SPECIAL(array, blocks, w_array)
        estimator, var_jack, bias_corr = jackknife(sv_block_P, np.mean)
        svendsen_avg_plaq_array.append(bias_corr)
        svendsen_avg_plaq_SE_array.append(np.sqrt(var_jack))

        if np.abs(k) < 1e-12:
            avg_plaq.append(bias_corr)
            SE_avg_plaq.append(np.sqrt(var_jack))

        # Susceptibility
        sv_block_P2 = svendsen_blocking_SPECIAL(array**2, blocks, w_array)
        sus_samples = []
        for j in range(blocks):
            P_leave_one_out = np.mean(np.delete(sv_block_P,j))
            P2_leave_one_out = np.mean(np.delete(sv_block_P2,j))
            sus_samples.append(Nplaq*(P2_leave_one_out - P_leave_one_out**2))

        sus_samples = np.array(sus_samples)
        full_sus = Nplaq*(np.mean(sv_block_P2)-np.mean(sv_block_P)**2)
        sus_dot = np.mean(sus_samples)
        bias_corr_sus = blocks*full_sus - (blocks-1)*sus_dot
        var_sus = (blocks-1)/blocks * np.sum((sus_samples - sus_dot)**2)

        svendsen_sus_array.append(bias_corr_sus)
        svendsen_sus_SE_array.append(np.sqrt(var_sus))

        if np.abs(k) < 1e-12:
            plaq_sus.append(bias_corr_sus)
            SE_plaq_sus.append(np.sqrt(var_sus))

        # Binder cumulant
        sv_block_P4 = svendsen_blocking_SPECIAL(array**4, blocks, w_array)
        binder_samples = []
        for j in range(blocks):
            P2_leave_one_out = np.mean(np.delete(sv_block_P2,j))
            P4_leave_one_out = np.mean(np.delete(sv_block_P4,j))
            binder_samples.append(1 - P4_leave_one_out/(3*P2_leave_one_out**2))

        binder_samples = np.array(binder_samples)
        full_binder = 1 - np.mean(sv_block_P4)/(3*np.mean(sv_block_P2)**2)
        b_dot = np.mean(binder_samples)
        bias_corr_binder = blocks*full_binder - (blocks-1)*b_dot
        var_binder = (blocks-1)/blocks * np.sum((binder_samples - b_dot)**2)

        svendsen_binder.append(bias_corr_binder)
        svendsen_binder_SE_array.append(np.sqrt(var_binder))

        sus_jackknife_samples.append(sus_samples)
        binder_jackknife_samples.append(binder_samples)

# -------------------
# Plotting
# -------------------

plt.errorbar(beta_array,avg_plaq,yerr=SE_avg_plaq,fmt='o',label='Parallel Tempering Data',color='blue',ecolor='red',elinewidth=3)
plt.errorbar(svendsen_beta_array,svendsen_avg_plaq_array,yerr=svendsen_avg_plaq_SE_array,fmt='o',label='Svendsen Reweighted Data',color='blue',ecolor='yellow',elinewidth=3,capsize=0,markersize=1)
plt.show()
plt.errorbar(beta_array,plaq_sus,yerr=SE_plaq_sus,fmt='o',label='Parallel Tempering Data',color='blue',ecolor='red',elinewidth=3)
plt.errorbar(svendsen_beta_array,svendsen_sus_array,yerr=svendsen_sus_SE_array,fmt='o',label='Svendsen Reweighted Data',color='blue',ecolor='yellow',elinewidth=3,capsize=0,markersize=1)
plt.show()
plt.errorbar(svendsen_beta_array,svendsen_binder,yerr=svendsen_binder_SE_array,fmt='o',label='Svendsen Reweighted Data',color='blue',ecolor='yellow',elinewidth=3,capsize=0,markersize=1)
plt.show()



montecarlo_data=np.transpose(np.array([beta_array,avg_plaq,plaq_sus,SE_avg_plaq,SE_plaq_sus]))
filename=folder + 'PT'+str(Spatial_Size)+'^3*'+str(Temporal_Size)+'_analysis.csv'

header = ["Beta", "Avg Plaq", "Plaq Sus", "Jackknife Action SE", "Jackknife Sus SE"]
with open(filename, 'w', newline='') as csvfile:
    # Create a CSV writer object
    csv_writer = csv.writer(csvfile)

    # Write all rows at once
    csv_writer.writerow(header)
    csv_writer.writerows(montecarlo_data)


print(f"CSV file '{filename}' created successfully.")


svendsen_data=np.transpose(np.array([svendsen_beta_array,svendsen_avg_plaq_array,svendsen_sus_array,svendsen_binder,svendsen_avg_plaq_SE_array,svendsen_sus_SE_array,svendsen_binder_SE_array]))
filename2=folder + 'PT'+str(Spatial_Size)+'^3*'+str(Temporal_Size)+'_svendsen_analysis.csv'

header = ["Beta", "Avg Plaq", "Plaq Sus","Binder Cum", "Jackknife Action SE", "Jackknife Sus SE","Binder Cum SE"]
with open(filename2, 'w', newline='') as csvfile:
    # Create a CSV writer object
    csv_writer = csv.writer(csvfile)

    # Write all rows at once
    csv_writer.writerow(header)
    csv_writer.writerows(svendsen_data)


print(f"CSV file '{filename2}' created successfully.")



jackknife_filename =folder + 'PT'+str(Spatial_Size)+'^3*'+str(Temporal_Size)+'sus_jackknife_samples.csv'


jackknife_matrix = np.transpose(np.array(sus_jackknife_samples))

# Create header row (beta values)
header = [f"{beta:.6f}" for beta in svendsen_beta_array]

with open(jackknife_filename, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(header)
    writer.writerows(jackknife_matrix)

print(f"CSV file '{jackknife_filename}' created successfully with betas as columns.")


jackknife_filename =folder + 'PT'+str(Spatial_Size)+'^3*'+str(Temporal_Size)+'binder_jackknife_samples.csv'


jackknife_matrix = np.transpose(np.array(binder_jackknife_samples))

# Create header row (beta values)
header = [f"{beta:.6f}" for beta in svendsen_beta_array]

with open(jackknife_filename, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(header)
    writer.writerows(jackknife_matrix)

print(f"CSV file '{jackknife_filename}' created successfully with betas as columns.")


