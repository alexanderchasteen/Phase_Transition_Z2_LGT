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
    // ---------------------- RNG Setup for Parallel ----------------------
    int n_threads = omp_get_max_threads();
    std::vector<std::mt19937> rng_threads(n_threads);
    for (int t = 0; t < n_threads; t++) rng_threads[t].seed(rng() + t);

    // ---------------------- Initialize Configuration ----------------------
    std::array<Link_array, CONFIG> links_at_coupling;
    for (int i=0; i<CONFIG; i++){
        links_at_coupling[i] = {};
        // cold_hot_start_mix(links_at_coupling[i], rng, 0);
        hot_start_array(links_at_coupling[i],rng);
        // multicanonical_start_left_cold(links_at_coupling[i], rng);
    }
    std::array<double, CONFIG> beta_array;
    std::array<int, CONFIG> beta_index_array;
    for (int i=0; i<CONFIG; i++){
        beta_array[i] = beta_min + i*dbeta;
        beta_index_array[i] = i;
    }

    // ----------------------Multicanonical Simulation Globals--------------------------
    double C=1;
    std::array<double, n_bins> weight_function = {};
    std::array<int, n_bins> reference_histogram = {};
   
    // // ----------------------------------------------------------------------------------------


    // ---------------------- Other Arrays ----------------------
    std::array<PaddedDouble, CONFIG> avg_plaq_array;    
    // ----------------------Measure OG AVG Plaq----------------
    std::array<double,CONFIG> current_avg_plaq;
    for (int i =0; i<CONFIG; i++){
        current_avg_plaq[i] = return_avg_plaq(links_at_coupling[i]);
    }

    // ---------------------- Thermalization ----------------------
    int j =0;
    while (C>tolerance) {
        j += 1;
        #pragma omp parallel 
        {
            int tid = omp_get_thread_num();
            #pragma omp for
            for (int i = 0; i < CONFIG; i++) {
                double beta = beta_array[i];
                avg_plaq_array[i].value = multicanonical_update_BUILD_HIST(links_at_coupling[i], rng_threads[tid], beta, current_avg_plaq[i], weight_function,reference_histogram,C);
                }
            }
    // ---------------------- SAVE / LOGGING ----------------------
    if (j % 10000 == 0) {
        std::cout << j << std::endl;
        }
    }
}