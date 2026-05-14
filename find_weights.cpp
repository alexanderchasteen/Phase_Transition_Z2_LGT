#include "Headers.h"
#include "Helper.cpp"

constexpr double min_bin = 0.2;
constexpr double max_bin = 1;
constexpr int n_bins = 8;
double bin_size = (max_bin-min_bin)/n_bins;
double C = 1;
constexpr double tolerance = 1e-6;
int meow = 1; //Switch on or off the C read in 

std::array<double,n_bins> weight_function = {};
std::array<int,n_bins> reference_histogram = {};

int find_bin_SIMULATION(double plaq){
    int bin_number;
    if (plaq == 1.0) bin_number=n_bins-1;
    else{
        bin_number = std::floor((plaq-min_bin)/bin_size);
    }
    return bin_number;
}

void mark_bin_visited(int idx){
    weight_function[idx] -= C;
    reference_histogram[idx] += 1;
    return;
}

void all_bins_visited(){
    double min_val = reference_histogram[0]; // Initialize with the first element
    for (int i = 1; i < n_bins; i++) {
        if (reference_histogram[i] < min_val) {
            min_val = reference_histogram[i];
        }
    }
    if (min_val>0){
        C /= 2.0;
        std::cout<<"\n -----Havling C to "<<C<<std::endl;
        reference_histogram = {};
        return;
    }
    return;
}


void write_lists_to_file(const std::array<double,n_bins>& list_1, const std::array<double,n_bins>& list_2)
{
    std::ofstream out("/home/alexa/Z2_Lattice_Gauge_Theory/Z2_CPP_NEW/weight_function_normalized.txt");

    if (!out.is_open()) {
        std::cerr << "Error: could not open file\n";
        return;
    }

    size_t N = std::min(list_1.size(), list_2.size());

    for (size_t i = 0; i < N; i++) {
        out << list_1[i] << " " << list_2[i] << "\n";
    }

    out.close();
}


void check_C(int meow){
    if (C<tolerance){
        if (meow = 1){
            std::cout<<"Done: Stop Running Simulation"<<std::endl;
        std::array<double,n_bins> edges;
        for (int i=0; i<n_bins;i++){
            edges[i]=min_bin+i*bin_size;
        }
        
        int min_val = weight_function[0]; // Initialize with the first element
        for (int i = 1; i < n_bins; i++) {
        if (weight_function[i] < min_val) min_val = weight_function[i];
        }

        for (int i=0; i<n_bins;i++) weight_function[i]=weight_function[i]-min_val;

        write_lists_to_file(edges,weight_function);
        meow=0;
        }
    }
    return;
}



void single_link_multicanonical_update_BUILD_HIST(Link_array &link, std::mt19937 &rng_local, double beta,int x0,int x1,int x2,int x3, int d,double &current_avg_plaq) {
    // 5 coordinates required for the link
    int dperp, staple, staplesum;
    staplesum = 0;
    for (dperp = 0; dperp < 4; dperp++) {
        if (dperp != d) {
            std::array<int, 4> y = {x0,x1,x2,x3};
            movedown(y,dperp);
            staple=get_array_value(link,y[0],y[1],y[2],y[3],dperp);
            staple*=get_array_value(link,y[0],y[1],y[2],y[3],d);
            moveup(y,d);
            staple*=get_array_value(link,y[0],y[1],y[2],y[3],dperp);
            moveup(y,dperp);
            staplesum+=staple;
            staple=get_array_value(link,y[0],y[1],y[2],y[3],dperp);
            moveup(y,dperp);
            movedown(y,d);
            staple*=get_array_value(link,y[0],y[1],y[2],y[3],d);
            movedown(y,dperp);
            staple*=get_array_value(link,y[0],y[1],y[2],y[3],dperp);
            staplesum+=staple;
            }
        }

    int old_link = get_array_value(link, x0, x1, x2, x3, d);
    double delta_S = 2.0 * beta * old_link * staplesum;

    // double current_avg_plaq = return_avg_plaq(link);
    int idx_old = find_bin_SIMULATION(current_avg_plaq);

    double delta_plaq = -2.0*old_link*staplesum/(N_plaq);
    double proposed_avg_plaq = current_avg_plaq+delta_plaq;
    
    int idx_new = find_bin_SIMULATION(proposed_avg_plaq);
    double delta_W = weight_function[idx_new] - weight_function[idx_old];

    double exponent=-delta_S+delta_W;

    if (exponent > 0 || sample_uniform(rng_local) < std::exp(exponent)) {
        set_array_value(link, x0, x1, x2, x3, d, -old_link);
        current_avg_plaq = proposed_avg_plaq;
    }
    int current_idx = find_bin_SIMULATION(current_avg_plaq);
    mark_bin_visited(current_idx);
    all_bins_visited();
    check_C(meow);
    return;
}


double single_link_multicanonical_update_BUILD_HIST(Link_array &link, std::mt19937 &rng_local, double beta, double &current_avg_plaq){
    std::array<int, 4> x;
    int d;
    for (x[0] = 0; x[0] < Spatial_Size; x[0]++)
    for (x[1] = 0; x[1] < Spatial_Size; x[1]++)
    for (x[2] = 0; x[2] < Spatial_Size; x[2]++)
    for (x[3] = 0; x[3] < Temporal_Size; x[3]++)
    for (d = 0; d < 4; d++) {
        single_link_multicanonical_update(link, rng_local, beta,x[0],x[1],x[2],x[3],d,current_avg_plaq);
    }
    return current_avg_plaq;
}