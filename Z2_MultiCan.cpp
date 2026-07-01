#include <iostream>
#include "Headers.h"
#include <fstream>
#include <iomanip>
#include <array>
#include <vector>
#include <random>
#include <omp.h>
#include <cmath>


//  REPLACE ALL  INSTANCES OF MULTICANONICAL WITH update_heatbath(links_at_coupling[i], beta, rng_threads[tid])

// ---------------------- Padding to avoid false sharing ----------------------
struct alignas(64) PaddedDouble {
    double value;
};

int main() {
    // ---------------------- Setup Output Files ----------------------
    std::string filename1 = "Raw_MC_Data_" + std::to_string(Spatial_Size)+"^3*"+std::to_string(Temporal_Size) + ".csv";
    std::ofstream MCfile(filename1);
    if (!MCfile) { std::cerr << "Error creating file!\n"; return 1; }
    MCfile << std::fixed << std::setprecision(6);
    for (int i=0; i<CONFIG; i++){
        double x = beta_min + i * dbeta;
        MCfile << x;
        if (i < CONFIG-1) MCfile << ",";
    }
    MCfile << "\n";
    std::cout << "MC File created.\n";

    // std::string filename2 = "Therm_Data_" + std::to_string(Spatial_Size)+"^3*"+std::to_string(Temporal_Size) + ".csv";
    // std::ofstream Thermfile(filename2);
    // if (!Thermfile) { std::cerr << "Error creating file!\n"; return 1; }
    // Thermfile << std::fixed << std::setprecision(6);
    // for (int i=0; i<CONFIG; i++){
    //     double x = beta_min + i * dbeta;
    //     Thermfile << x;
    //     if (i < CONFIG-1) Thermfile << ",";
    // }
    // Thermfile << "\n";
    // std::cout << "Therm File created.\n";


    // ------------------------Multicanonical Set up --------------------------
    load_weights("weight_function_normalized.txt");

    
    if (weights.empty()) {
        std::cerr << "Error: Weights not loaded. Check file path.\n";
        return 1;
    }

   

    std::cout << "\n=== C++ BIN CHECK ===\n";
    for (int i = 0; i < 15; i++) {
        std::cout << "edge[" << i << "] = " << bin_edges[i]
                << " | weight[" << i << "] = " << weights[i] << std::endl;
    }
    std::cout << "last edge = " << bin_edges.back() << std::endl;
    std::cout << "sizes: edges=" << bin_edges.size()
            << " weights=" << weights.size() << std::endl;

    // ---------------------- RNG Setup for Parallel ----------------------
    int n_threads = omp_get_max_threads();
    std::vector<std::mt19937> rng_threads(n_threads);
    for (int t = 0; t < n_threads; t++) rng_threads[t].seed(rng() + t);

    // ---------------------- Initialize Configuration ----------------------
    std::array<Link_array, CONFIG> links_at_coupling;
    for (int i=0; i<CONFIG; i++){
        links_at_coupling[i] = {};
        cold_hot_start_mix(links_at_coupling[i], rng, i);
        // multicanonical_start_left_cold(links_at_coupling[i], rng);
    }
    std::array<double, CONFIG> beta_array;
    std::array<int, CONFIG> beta_index_array;
    for (int i=0; i<CONFIG; i++){
        beta_array[i] = beta_min + i*dbeta;
        beta_index_array[i] = i;
    }

    // ----------------------Loading in a precomputed configuration----------------------------
    // Make the Necessary Arrays to keep track of the things
    // std::array<double,CONFIG> beta_array=load_PT_Beta_Array("Beta_array_"+ std::to_string(Spatial_Size)+"^3*"+ std::to_string(Temporal_Size));
    // std::array<int,CONFIG> beta_index_array=load_PT_index_Array("Beta_index_array_"+ std::to_string(Spatial_Size)+"^3*"+ std::to_string(Temporal_Size));
    // std::array<Link_array, CONFIG> links_at_coupling=load_links_all("Thermalized_Lattice_"+ std::to_string(Spatial_Size)+"^3*"+ std::to_string(Temporal_Size));
   
    // // ----------------------------------------------------------------------------------------


    // ---------------------- Other Arrays ----------------------
    std::array<PaddedDouble, CONFIG> avg_plaq_array;
    std::array<double, CONFIG> IAT_array;
    
    // ----------------------Measure OG AVG Plaq----------------
    std::array<double,CONFIG> current_avg_plaq;
    for (int i =0; i<CONFIG; i++){
        current_avg_plaq[i] = return_avg_plaq(links_at_coupling[i]);
    }

    // ---------------------- Thermalization ----------------------
   
    for (int j = 0; j < thermal_sweeps; j++) {
        #pragma omp parallel 
        {
            int tid = omp_get_thread_num();
            #pragma omp for
            for (int i = 0; i < CONFIG; i++) {
                double beta = beta_array[i];
                avg_plaq_array[i].value = update_metropolis_multicanonical(links_at_coupling[i], rng_threads[tid], beta, current_avg_plaq[i]);
                }
            }
        static std::mt19937 swap_rng(std::random_device{}());
        if (j % 10 == 5) {
            for (int i = 0; i < CONFIG - 1; i += 2) {
                double delta = N_plaq * (beta_array[i] - beta_array[i + 1]) *
                    (avg_plaq_array[i + 1].value - avg_plaq_array[i].value);

                double Pswap = std::min(1.0, std::exp(delta));
                if (sample_uniform(swap_rng) < Pswap) {
                    std::swap(beta_array[i], beta_array[i + 1]);
                    std::swap(beta_index_array[i], beta_index_array[i + 1]);
                }
            }
        }
        if (j % 10 == 0) {
            for (int i = 1; i < CONFIG - 1; i += 2) {
                double delta = N_plaq * (beta_array[i] - beta_array[i + 1]) *
                    (avg_plaq_array[i + 1].value - avg_plaq_array[i].value);

                double Pswap = std::min(1.0, std::exp(delta));
                if (sample_uniform(swap_rng) < Pswap) {
                    std::swap(beta_array[i], beta_array[i + 1]);
                    std::swap(beta_index_array[i], beta_index_array[i + 1]);
                }
            }
        }
    // ---------------------- SAVE / LOGGING ----------------------
    if (j % 10000 == 0) {
        #pragma omp critical
        save_links_all(
            links_at_coupling,
            "Thermalized_Lattice_" +
            std::to_string(Spatial_Size) + "^3*" +
            std::to_string(Temporal_Size)
        );

        save_PT_Beta_Array(
            beta_array,
            "Beta_array_" +
            std::to_string(Spatial_Size) + "^3*" +
            std::to_string(Temporal_Size)
        );

        save_PT_index_Array(
            beta_index_array,
            "Beta_index_array_" +
            std::to_string(Spatial_Size) + "^3*" +
            std::to_string(Temporal_Size)
        );

        std::cout << j << std::endl;
        }
        if (j % 100 == 0){
        for (int i = 0; i < CONFIG; i++) {
            current_avg_plaq[i] =
                return_avg_plaq(links_at_coupling[i]);
            }
    }
    }


    
    // ---------------------- Save final thermalized state ----------------------
    save_links_all(links_at_coupling,"Thermalized_Lattice_" + std::to_string(Spatial_Size)+"^3*" + std::to_string(Temporal_Size));
    save_PT_Beta_Array(beta_array,"Beta_array_" + std::to_string(Spatial_Size)+"^3*" + std::to_string(Temporal_Size));
    save_PT_index_Array(beta_index_array,"Beta_index_array_" + std::to_string(Spatial_Size)+"^3*" + std::to_string(Temporal_Size));


// ---------------------- Autocorrelation ----------------------
    // #pragma omp parallel for
    // for (int i = 0; i < CONFIG; i++) {
    //     std::array<double, maxlag> rho;
    //     int tid = omp_get_thread_num();
    //     std::array<double, autocorrelation_sweeps> autocorrelation_array;
    //     for (int j = 0; j < autocorrelation_sweeps; j++) {
    //         double beta = beta_array[i];
    //         autocorrelation_array[j] = update_metropolis_multicanonical(links_at_coupling[i], rng_threads[tid], beta, current_avg_plaq[i]);
    //     }

    //     rho = autocorr(autocorrelation_array);
    //     double taucomp = tau_int(rho);
    //     if (taucomp < 1.0) taucomp = 1.0;
    //     IAT_array[i] = std::ceil(taucomp);
    //     #pragma omp critical
    //     std::cout << "Beta " << beta_array[i]
    //             << " IAT " << IAT_array[i]
    //             << "\n";
    // }


    // ---------------------- Measurement Sweeps ----------------------
    for (int j = 0; j < measurment_sweeps; j++) {
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            #pragma omp for
            for (int i = 0; i < CONFIG; i++) {
                double beta = beta_array[i];
                // for (int k = 0; k<IAT_array[i]; k++){
                avg_plaq_array[i].value = update_metropolis_multicanonical(links_at_coupling[i], rng_threads[tid], beta, current_avg_plaq[i]);
                    // }         
                }
        }

        static std::mt19937 swap_rng(std::random_device{}());

        if (j % 10 == 5) {
            for (int i = 0; i < CONFIG - 1; i += 2) {
                double delta =
                    N_plaq * (beta_array[i] - beta_array[i + 1]) *
                    (avg_plaq_array[i + 1].value - avg_plaq_array[i].value);
                double Pswap = std::min(1.0, std::exp(delta));
                if (sample_uniform(swap_rng) < Pswap) {
                    std::swap(beta_array[i], beta_array[i + 1]);
                    std::swap(beta_index_array[i], beta_index_array[i + 1]);
                }
            }
        }

        if (j % 10 == 0) {
            for (int i = 1; i < CONFIG - 1; i += 2) {
                double delta =
                    N_plaq * (beta_array[i] - beta_array[i + 1]) *
                    (avg_plaq_array[i + 1].value - avg_plaq_array[i].value);
                double Pswap = std::min(1.0, std::exp(delta));
                if (sample_uniform(swap_rng) < Pswap) {
                    std::swap(beta_array[i], beta_array[i + 1]);
                    std::swap(beta_index_array[i], beta_index_array[i + 1]);
                }
            }
        }

        #pragma omp single
            {
                std::array<double, CONFIG> store_data;
                for (int i=0; i<CONFIG; i++){
                    store_data[beta_index_array[i]]=current_avg_plaq[i];
                }
                
                for (int i =0; i<CONFIG; i++){
                    MCfile << store_data[i];\
                    if (i < CONFIG - 1) MCfile << ",";
                }
                
                MCfile << "\n";
            }
        if (j % 100 == 0){
        for (int i = 0; i < CONFIG; i++) {
            current_avg_plaq[i] =
                return_avg_plaq(links_at_coupling[i]);
            }
    }
    }
    MCfile.close();
    // Thermfile.close();
    return 0;
}