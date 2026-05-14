#pragma once
#include <vector>
#include <array>
#include <cmath>
#include <random>
#include <omp.h>
#include <fstream>
#include <algorithm>
#include<iostream>

extern std::mt19937 rng;

// Simulation Parameters


// Lattice Configurations
// constexpr int SIZE = 8;
// constexpr int CONFIG = 20;

// // Beta 
// constexpr double dbeta= 0.000050;
// constexpr double beta_min=0.439500;

// // Measurments
// constexpr int thermal_sweeps=0;  
// constexpr int autocorrelation_sweeps=1000;  
// constexpr int measurment_sweeps=500000;  
// constexpr int maxlag=800;



// Lattice Configurations 
// MEOWWWW DO SOMETHING
// constexpr int Spatial_Size = 7;
// constexpr int Temporal_Size = Spatial_Size;
// constexpr int CONFIG = 1;

// // Beta 
// constexpr double dbeta= 0.000200;
// constexpr double beta_min=0.44;

// // Measurments
// constexpr int thermal_sweeps=10000;  
// constexpr int autocorrelation_sweeps=1000;  
// constexpr int measurment_sweeps=100000;  
// constexpr int maxlag=800;

constexpr int Spatial_Size = 3;
constexpr int Temporal_Size = Spatial_Size;
constexpr int CONFIG = 1;

// // Beta
constexpr double dbeta= 0.00005;
//constexpr double beta_min=0.4390;
constexpr double beta_min=0.431;

// // Measurments
constexpr int thermal_sweeps=1000000;
constexpr int autocorrelation_sweeps=1000;
constexpr int measurment_sweeps=500000;
constexpr int maxlag=800;


// Lattice Configurations 
// constexpr int Spatial_Size = 6;
// constexpr int Temporal_Size = Spatial_Size;
// constexpr int CONFIG = 1;

// // Beta 
// constexpr double dbeta= 0.0001;
// constexpr double beta_min=0.4399;


// // Measurments
// constexpr int thermal_sweeps=100000;  
// constexpr int autocorrelation_sweeps=1000;  
// constexpr int measurment_sweeps=500000;  
// constexpr int maxlag=800;


// Lattice Configurations 
// constexpr int Spatial_Size = 3;
// constexpr int Temporal_Size = Spatial_Size;
// constexpr int CONFIG = 30;

// // Beta 
// constexpr double dbeta= 0.0004;
// constexpr double beta_min=0.423;

// // Measurments
// constexpr int thermal_sweeps=0;  
// constexpr int autocorrelation_sweeps=1000;  
// constexpr int measurment_sweeps=100000;  
// constexpr int maxlag=800;

// Lattice Configurations 
// constexpr int Spatial_Size = 4;
// constexpr int Temporal_Size = Spatial_Size;
// constexpr int CONFIG = 10;

// // Beta 
// constexpr double dbeta= 0.0004;
// constexpr double beta_min=0.435000;

// // Measurments
// constexpr int thermal_sweeps=10000;  
// constexpr int autocorrelation_sweeps=1000;  
// constexpr int measurment_sweeps=500000;  
// constexpr int maxlag=800;



// Lattice Configurations 
// constexpr int Spatial_Size = 5;
// constexpr int Temporal_Size = Spatial_Size;
// constexpr int CONFIG = 10;

// // Beta 
// constexpr double dbeta= 0.0001;
// constexpr double beta_min=0.439;
// // constexpr double beta_min=0.438500;

// // Measurments
// constexpr int thermal_sweeps=1000000;  
// constexpr int autocorrelation_sweeps=1000;  
// constexpr int measurment_sweeps=500000;  
// constexpr int maxlag=800;




// Constants to not be manipulated
constexpr int save_array_size = 4*Spatial_Size*Spatial_Size*Spatial_Size*Temporal_Size*CONFIG;
constexpr int one_config_array_size = 4*Spatial_Size*Spatial_Size*Spatial_Size*Temporal_Size;
constexpr int N_plaq = Spatial_Size*Spatial_Size*Spatial_Size*Temporal_Size*6;
constexpr double beta_max=beta_min+CONFIG*dbeta;




// RNG
double sample_uniform(std::mt19937 &rng_local);
std::array<int,4> sample_coord(std::mt19937 &rng_local);


// Lattice configuration updates

// using Link_array = std::vector<int>;
using Link_array = std::array<int,one_config_array_size>;
// using Link_array = std::vector<int>;
void cold_start_array(Link_array& arr);
void hot_start_array(Link_array& arr, std::mt19937 &rng_local);
void cold_hot_start_mix(Link_array& arr, std::mt19937 &rng_local, int temp);

void moveup(std::array<int,4>& v,int d);
void movedown(std::array<int,4>& v, int d);
double return_avg_plaq(Link_array& link);


// Save Config
void save_links_all(const std::array<Link_array, CONFIG>& links, const std::string& filename); 
void save_PT_Beta_Array(const std::array<double,CONFIG>& array, const std::string& filename);
void save_PT_index_Array(const std::array<int,CONFIG>& array, const std::string& filename);
std::array<double, CONFIG> load_PT_Beta_Array(const std::string& filename);
std::array<int, CONFIG> load_PT_index_Array(const std::string& filename);
std::array<Link_array, CONFIG> load_links_all(const std::string& filename);

// Swap elements
void swap_double(std::array<double,CONFIG>& v, int index1, int index2);
void swap_int(std::array<int,CONFIG>& v, int index1, int index2);

// Monte Carlo updates
double update_heatbath(Link_array& link, double beta,std::mt19937 &rng_local);
double update_metropolis(Link_array& link, double beta,std::mt19937 &rng_local);
void thermalize(Link_array& link, int therm_sweeps, double beta,std::mt19937 &rng_local);



// Autocorrelation analysis
std::vector<double> autocorr(const std::vector<double>& data, int max_lag);
float tau_int(const std::vector<double>& autocorr_data, int maxlag);

// Tensor analysis
int flat_index(int i1, int i2, int i3, int i4, int i5);
int get_array_value(const Link_array& arr, int i1, int i2, int i3, int i4, int i5) ;
void set_array_value(Link_array& arr, int i1, int i2, int i3, int i4, int i5, int value);
std::array<int,5> tensor_index_array(int index);


// Statistics
double mean(const std::array<double,autocorrelation_sweeps>& data);
double variance(const std::array<double,autocorrelation_sweeps>& data);
std::array<double,maxlag> autocorr(const std::array<double,autocorrelation_sweeps>& data);
double tau_int(const std::array<double,maxlag>& rho);


// Multicanonical 
inline std::vector<double> bin_edges;
inline std::vector<double> weights;
int find_bin(double plaq); 
void load_weights(const std::string &filename);
double update_metropolis_multicanonical(Link_array &link, std::mt19937 &rng_local, double beta) ;
void multicanonical_start_left_cold(Link_array& arr, std::mt19937 &rng_local);
void multicanonical_start_right_cold(Link_array& arr, std::mt19937 &rng_local);
void alternating_multicanonical_start(Link_array& arr, std::mt19937 &rng_local, int parity);
void single_link_multicanonical_update(Link_array &link, std::mt19937 &rng_local, double beta,int x0,int x1,int x2,int x3, int d, double &current_avg_plaq);
double update_metropolis_multicanonical(Link_array &link, std::mt19937 &rng_local, double beta, double &current_avg_plaq);



// Multicanonical Weights Getting
constexpr double min_bin = 0.45;
constexpr double max_bin = 0.95;
constexpr int n_bins = 60;
constexpr double bin_size = (max_bin - min_bin) / n_bins;
constexpr double tolerance = 1e-6;
void mark_bin_visited(int idx,std::array<double, n_bins>& weight_function, std::array<int, n_bins>& reference_histogram, double constant);
void all_bins_visited(std::array<double, n_bins>& reference_histogram, double& constant);
void write_lists_to_file(const std::array<double,n_bins>& list_1, const std::array<double,n_bins>& list_2);
void check_C(double constant, std::array<double, n_bins> weight_function);
void single_link_multicanonical_update_BUILD_HIST(Link_array &link, std::mt19937 &rng_local, double beta,int x0,int x1,int x2,int x3, int d,double &current_avg_plaq, std::array<double, n_bins> weight_function); 
double multicanonical_update_BUILD_HIST(Link_array &link, std::mt19937 &rng_local, double beta,double &current_avg_plaq, std::array<double, n_bins>& weight_function, std::array<int, n_bins>& reference_histogram, double& constant);