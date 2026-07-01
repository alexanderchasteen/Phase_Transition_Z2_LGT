import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

# =========================
# Parameters
# =========================
for x in range(3, 11):
    Spatial_Size = int(x)
    Temporal_Size = Spatial_Size
    folder = './Multi_Canon_Results_unbiased/'

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
    def lorentzian(x, x0, gamma, a):
        g = np.abs(gamma)  # Keeps sign handling uniform across fits
        return a * g**2 / ((x - x0)**2 + g**2)

    # =========================
    # Fit window selection
    # =========================
    if Spatial_Size == 3:
        beta_min, beta_max = 0.428, 0.434
        initial_guess = [0.43, 0.0005, 2500]
    elif Spatial_Size == 4:
        beta_min, beta_max = 0.436, 0.439
        initial_guess = [0.437, 0.0005, 2500]
    elif Spatial_Size == 5:
        beta_min, beta_max = 0.439, 0.4396
        initial_guess = [0.439, 0.0005, 2500]
    elif Spatial_Size == 6:
        beta_min, beta_max = 0.4398, 0.4402
        initial_guess = [0.44, 0.0005, 2500]
    elif Spatial_Size == 7:
        beta_min, beta_max = 0.4401, 0.4405
        initial_guess = [0.4403, 0.0005, 2500]
    elif Spatial_Size == 8:
        beta_min, beta_max = 0.4404, 0.4406
        initial_guess = [0.4405, 0.0005, 2500]
    elif Spatial_Size == 9:
        beta_min, beta_max = 0.4405, 0.4406
        initial_guess = [0.44055, 0.0005, 2500]
    elif Spatial_Size == 10:
        beta_min, beta_max = 0.44058, 0.44064
        initial_guess = [0.4406, 0.0005, 2500]

    # Central fit mask over the global array
    mask = (svendsen_beta >= beta_min) & (svendsen_beta <= beta_max)
    x = svendsen_beta[mask]
    y = sus_sv[mask]
    yerr = sus_sv_SE[mask]

    # Master Fit
    param, _ = curve_fit(lorentzian, x, y, p0=initial_guess, maxfev=10000)
    x0, gamma_raw, a = param
    gamma = np.abs(gamma_raw)
    R2 = r2_score(y, lorentzian(x, x0, gamma, a))

    # =========================
    # Corrected Jackknife Parsing
    # =========================
    jack_filepath = folder + f'PT{Spatial_Size}^3*{Temporal_Size}sus_jackknife_samples.csv'

    # 1. Safely extract exact matrix beta coordinates from the header row
    with open(jack_filepath, 'r') as csvfile:
        header_line = csvfile.readline().strip().split(',')
    beta_jack = np.array([float(beta_val) for beta_val in header_line])

    # 2. Load numeric rows skipping the column header labels
    jack = np.loadtxt(jack_filepath, delimiter=',', skiprows=1)

    # 3. Create independent mask tracking strictly for the matrix data coordinate limits
    mask_j = (beta_jack >= beta_min) & (beta_jack <= beta_max)

    crit_beta_samples = []
    FWHM_samples = []
    height_samples = []
    
    def jackknife_var(data_array):
        N = len(data_array)
        mean_val = np.mean(data_array)
        return (N - 1) * np.mean((data_array - mean_val)**2)

    # Loop over true data rows
    for i in range(jack.shape[0]): 
        y_sample = jack[i, :]

        x_fit = beta_jack[mask_j]
        y_fit = y_sample[mask_j]

        if len(x_fit) < 5:
            continue
        
        try:
            param_i, _ = curve_fit(lorentzian, x_fit, y_fit, p0=[x0, gamma, a], maxfev=10000)
            x0_i, gamma_i_raw, a_i = param_i
            gamma_i = np.abs(gamma_i_raw)
            
            # Interactive visualization diagnostics window
            if i % 10 == 20:
                cts_i = np.linspace(np.min(x_fit), np.max(x_fit), 1000)
                plt.figure(figsize=(8, 5))
                plt.plot(x_fit, y_fit, 'o', markersize=3, label=f'JK block {i}')
                plt.plot(cts_i, lorentzian(cts_i, x0_i, gamma_i, a_i), label=f'x0={x0_i:.8f}, FWHM={2*gamma_i:.3e}, h={a_i:.2f}')
                cts_global = np.linspace(beta_min, beta_max, 1000)
                plt.plot(cts_global, lorentzian(cts_global, x0, gamma, a), '--', alpha=0.7, label='Full fit')
                plt.xlabel('Beta')
                plt.ylabel('Susceptibility')
                plt.title(f'L={Spatial_Size}, Jackknife block {i}')
                plt.legend()
                plt.show()

            crit_beta_samples.append(x0_i)
            FWHM_samples.append(2 * gamma_i)
            height_samples.append(a_i)
        except RuntimeError:
            print(f"Fit failed to converge for Jackknife block {i}")
            continue

    # Final error estimation calculations
    SE_beta = np.sqrt(jackknife_var(np.array(crit_beta_samples)))
    SE_FWHM = np.sqrt(jackknife_var(np.array(FWHM_samples)))
    SE_height = np.sqrt(jackknife_var(np.array(height_samples)))

    print(f"{Spatial_Size}, {x0}, {2*gamma},{a}, {SE_beta}, {SE_FWHM}, {SE_height}")

    B = len(crit_beta_samples)
    beta_bar = np.mean(crit_beta_samples)
    FWHM_bar = np.mean(FWHM_samples)
    height_bar = np.mean(height_samples)
    
    # Clean mathematical bias corrections
    beta_bc = B * x0 - (B - 1) * beta_bar
    FWHM_bc = B * (2 * gamma) - (B - 1) * FWHM_bar
    height_bc = B * a - (B - 1) * height_bar

    # print(f"L={Spatial_Size} Final Results:")
    # print(f"Beta_c: {beta_bc:.6f} +/- {SE_beta:.6f}")
    # print(f"FWHM:   {FWHM_bc:.6e} +/- {SE_FWHM:.6e}")
    # print(f"Height: {height_bc:.2f} +/- {SE_height:.2f}\n" + "=" * 50)

    # print(f"{Spatial_Size}, {beta_bc}, {FWHM_bc},{height_bc}, {SE_beta}, {SE_FWHM}, {SE_height}")
    # jack = np.loadtxt(
    #     folder + f'PT{Spatial_Size}^3*{Temporal_Size}sus_jackknife_samples.csv',
    #     delimiter=','
    # )

    # beta_jack = jack[0, :]
    # delta = (beta_max - beta_min) / 2

    # crit_beta = []
    # FWHM = []
    # height = []
    # p_0 = param
    # def jackknife_var(data):
    #     N = len(data)
    #     mean = np.mean(data)
    #     return (N - 1) * np.mean((data - mean)**2)

    # for i in range(1, jack.shape[0]):

    #     y_sample = jack[i, :]

    #     peak_idx = np.argmax(y_sample)
    #     beta_peak = beta_jack[peak_idx]

    #     mask_j = (beta_jack >= beta_peak - delta) & (beta_jack <= beta_peak + delta)

    #     x_fit = beta_jack[mask]
    #     y_fit = y_sample[mask]

    #     if len(x_fit) < 5:
    #         continue

    #     param, _ = curve_fit(lorentzian, x_fit, y_fit,
    #                         p0=p_0, maxfev=10000)

    #     x0_i, gamma_i, a_i= param

    #     crit_beta.append(x0_i)
    #     FWHM.append(2 * gamma_i)
    #     height.append(a_i)


    #     cts_i = np.linspace(np.min(x_fit), np.max(x_fit), 500)

    #         # plot raw jackknife sample
    #     # plt.plot(
    #     #         x_fit,
    #     #         y_fit,
    #     #         'o',
    #     #         markersize=2,
    #     #         alpha=0.3
    #     #     )
        
    #     #     # plot fitted curve
    #     # plt.plot(
    #     #         cts_i,
    #     #         lorentzian(cts_i, x0_i, gamma_i, a_i, c_i),
    #     #         alpha=0.3
    #     #     )
    #     # plt.show()


    # SE_beta = np.sqrt(jackknife_var(crit_beta))
    # SE_FWHM = np.sqrt(jackknife_var(FWHM))
    # SE_height = np.sqrt(jackknife_var(height))

    # # print("Critical beta:", x0, "+/-", SE_beta)
    # # print("FWHM:", 2*gamma, "+/-", SE_FWHM)
    # # print("Peak height:", a, "+/-", SE_height)
    # print(str(Spatial_Size),',',x0,',',2*gamma,',',a,',', SE_beta,',', SE_FWHM,',', SE_height)