import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

# =========================
# Parameters
# =========================
for x in range(3,11):
    Spatial_Size = int(x)
    Temporal_Size = Spatial_Size
    folder = './Multi_Canon_Results/'

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
    # Plot raw susceptibility
    # =========================
    plt.rc('font', size=12)

    plt.errorbar(svendsen_beta, sus_sv, yerr=sus_sv_SE,
                fmt='o', color='blue', ecolor='yellow',
                markersize=1, label='Svendsen')

    plt.errorbar(beta_pt, sus_pt, yerr=sus_pt_SE,
                fmt='o', color='blue', ecolor='red',
                label='PT')

    plt.title('Plaquette Susceptibility vs Beta')
    plt.xlabel('Beta')
    plt.ylabel('Susceptibility')
    plt.legend()
    # plt.show()

    # =========================
    # Model
    # =========================
    def lorentzian(x, x0, gamma, a):
        # return a*np.exp(-4*np.log(2)*((x - x0)**2) / (gamma*gamma))
        return a * gamma**2 / ((x - x0)**2 + gamma**2)



    # =========================
    # # Fit window

    if Spatial_Size==3:
        beta_min = 0.428
        beta_max = 0.434
        initial_guess = [0.43, 0.0005, 2500]

    if Spatial_Size==4:
        beta_min = 0.436
        beta_max = 0.439
        initial_guess = [0.437, 0.0005, 2500]

    if Spatial_Size==5:
        beta_min = 0.43875
        beta_max = 0.44
        initial_guess = [0.439, 0.0005, 2500]

    if Spatial_Size==6:
        beta_min = 0.4396
        beta_max = 0.4404
        initial_guess = [0.44, 0.0005, 2500]

    if Spatial_Size==7:
        beta_min = 0.4401
        beta_max = 0.4405
        initial_guess = [0.4403, 0.0005, 2500]

    if Spatial_Size==8:
        beta_min = 0.4403
        beta_max = 0.4407
        initial_guess = [0.4403, 0.0005, 2500]

    if Spatial_Size==9:
        beta_min = 0.4404
        beta_max = 0.44075
        initial_guess = [0.4405, 0.0005, 2500]

    if Spatial_Size==10:
        beta_min = 0.4405
        beta_max = 0.4407
        initial_guess = [0.4406, 0.0005, 2500]



    mask = (svendsen_beta >= beta_min) & (svendsen_beta <= beta_max)

    x = svendsen_beta[mask]
    y = sus_sv[mask]
    yerr = sus_sv_SE[mask]

    # =========================
    # Fit
    # =========================
    param, _ = curve_fit(lorentzian, x, y, p0=initial_guess,sigma=yerr,absolute_sigma=True,maxfev=10000)

    x0, gamma, a = param
    R2 = r2_score(y, lorentzian(x, x0, gamma, a))

    # =========================
    # Plot fit
    # =========================
    cts = np.linspace(beta_min, beta_max, 1000)

    plt.rc('font', size=15)

    plt.errorbar(x, y, yerr=yerr,
                fmt='o', color='blue', ecolor='yellow',
                label='Data')

    plt.errorbar(beta_pt, sus_pt, yerr=sus_pt_SE,
                fmt='o', color='blue', ecolor='red',
                label='PT')

    plt.plot(cts, lorentzian(cts, x0, gamma, a),
            label=f'Fit R²={R2:.4f}', alpha=0.5)

    plt.title('Susceptibility Fit')
    plt.xlabel('Beta')
    plt.ylabel('Susceptibility')
    plt.legend()
    # plt.show()

    # =========================
    # Jackknife
    # =========================
    jack = np.loadtxt(
        folder + f'PT{Spatial_Size}^3*{Temporal_Size}sus_jackknife_samples.csv',
        delimiter=',',
        skiprows=1 
    )

    # 2. The X-axis for Jackknife fits must be the exact same X-axis used for the main data
    beta_jack = svendsen_beta 

    crit_beta_samples = []
    FWHM_samples = []
    height_samples = []
    
    # Base parameters from your main fit to help the Jackknife fits converge quickly
    p_0 = param 
    
    def jackknife_var(data_array):
        N = len(data_array)
        mean_val = np.mean(data_array)
        # Your variance formula here is mathematically flawless
        return (N - 1) * np.mean((data_array - mean_val)**2)

    # 3. Loop over ALL rows (blocks)
    for i in range(jack.shape[0]): 
        y_sample = jack[i, :]

        # Apply the exact same mask you used for the main fit above
        x_fit = beta_jack[mask]
        y_fit = y_sample[mask]

        if len(x_fit) < 5:
            continue
        
        # We use a try/except because occasionally a Jackknife sample might not converge
        try:
            param_i, _ = curve_fit(lorentzian, x_fit, y_fit, p0=p_0, maxfev=10000)
            x0_i, gamma_i, a_i = param_i

            crit_beta_samples.append(x0_i)
            FWHM_samples.append(2 * gamma_i)
            height_samples.append(a_i)
        except RuntimeError:
            print(f"Fit failed to converge for Jackknife block {i}")
            continue

    # 4. Calculate final standard errors
    SE_beta = np.sqrt(jackknife_var(np.array(crit_beta_samples)))
    SE_FWHM = np.sqrt(jackknife_var(np.array(FWHM_samples)))
    SE_height = np.sqrt(jackknife_var(np.array(height_samples)))

    print(f"{Spatial_Size}, {x0}, {2*gamma}, {a}, {SE_beta}, {SE_FWHM}, {SE_height}")



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