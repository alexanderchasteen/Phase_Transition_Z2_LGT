import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('/home/alexa/Z2_Lattice_Gauge_Theory/Z2_CPP_NEW/MC_Raw_Data_Saved/Raw_MC_Data_M5.csv')
bin_width = 0.01
beta_array = df.columns.to_numpy()

# Select the column near the critical beta
# for x in beta_array:
x = beta_array[16]
print(x)
histogram_data = df[x].dropna().to_numpy()  # convert to numpy array

# Define bins
bins = np.arange(histogram_data.min(), histogram_data.max() + bin_width, bin_width)

# Compute counts and bin edges
counts, bin_edges = np.histogram(histogram_data, bins=bins)

# bin_edges has length n_bins + 1
# counts has length n_bins
# For plotting or further use, you often take the left edge of each bin:
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

# Print the results
for i in range(len(counts)):
    print(f"Bin {i}: {bin_edges[i]:.3f} to {bin_edges[i+1]:.3f}, Count = {counts[i]}")


# Plot histogram using the counts you computed
plt.figure(figsize=(8,5))
plt.bar(bin_centers, counts, width=bin_width, edgecolor='black', alpha=0.7)
plt.xlabel('Average Plaquette',fontsize=16)
plt.ylabel('Counts',fontsize=16)
plt.title(r'Histogram for 5^4 Lattice at $\beta = 0.4393$',fontsize=20)
plt.grid(True, linestyle='--', alpha=0.5)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()
