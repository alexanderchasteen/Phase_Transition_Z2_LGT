#include "Headers.h"

// Randomly generate Number 0 to 1
std::mt19937 rng(std::random_device{}());
double sample_uniform(std::mt19937 &rng_local) {
    static thread_local std::uniform_real_distribution<double> dist(0.0, 1.0);
    return dist(rng_local);
}


// Gives the index in an array given the indices in our 6 index tensor for the lattice_config x lattice_data information 
int flat_index(int i1, int i2, int i3, int i4, int i5) {
    return ((((i1 * Spatial_Size + i2) * Spatial_Size + i3) * Temporal_Size + i4) * 4 + i5);
}

// gives the 6 indices in the tensor given the location in the array
std::array<int,5> tensor_index_array(int index){
    int i5 = index % 4;
    index /= Temporal_Size;
    int i4 = index % Temporal_Size;
    index /= Spatial_Size;
    int i3 = index % Spatial_Size;
    index /= Spatial_Size;
    int i2 = index % Spatial_Size;
    index /= Spatial_Size;
    int i1 = index;
    return {i1,i2,i3,i4,i5};
}

void set_array_value(Link_array& arr, int i1, int i2, int i3, int i4, int i5, int value) {
    int idx = flat_index(i1, i2, i3, i4, i5);
    arr[idx] = value;
    return;
}

// Gets the value at a certain point in the array
int get_array_value(const Link_array& arr, int i1, int i2, int i3, int i4, int i5) {
    int idx = flat_index(i1, i2, i3, i4, i5);
    return arr[idx];
}

// Sets all the links to 1. Same as iteratring through the array but this way is to make sure consistint with the array construction
void cold_start_array(Link_array& arr){
    for (int i1 = 0; i1 < Spatial_Size; i1++)
    for (int i2 = 0; i2 < Spatial_Size; i2++)
    for (int i3 = 0; i3 < Spatial_Size; i3++)
    for (int i4 = 0; i4 < Temporal_Size; i4++)
    for (int i5 = 0; i5 < 4; i5++)
    set_array_value(arr,i1,i2,i3,i4,i5,1);
    return;
}

void hot_start_array(Link_array& arr, std::mt19937 &rng_local){
    int x0; 
    for (int i1 = 0; i1 < Spatial_Size; i1++)
    for (int i2 = 0; i2 < Spatial_Size; i2++)
    for (int i3 = 0; i3 < Spatial_Size; i3++)
    for (int i4 = 0; i4 < Temporal_Size; i4++)
    for (int i5 = 0; i5 < 4; i5++){
        x0 = (sample_uniform(rng_local) >= 0.5) ? 1 : -1;
        set_array_value(arr,i1,i2,i3,i4,i5,x0);
        }
    return;
    }

// Tempo=0 is cold and Temp=1 is hot
void cold_hot_start_mix(Link_array& arr, std::mt19937 &rng_local, int temp){
    if (temp % 2 == 0) {
            cold_start_array(arr);
            }
    if (temp % 2 == 1){
            hot_start_array(arr, rng_local);
            }
        return; 
    }

void multicanonical_start_left_cold(Link_array& arr, std::mt19937 &rng_local){
    int x0; 
    for (int i1 = 0; i1 < Spatial_Size; i1++)
    for (int i2 = 0; i2 < Spatial_Size; i2++)
    for (int i3 = 0; i3 < Spatial_Size; i3++)
    for (int i4 = 0; i4 < Temporal_Size; i4++)
    for (int i5 = 0; i5 < 4; i5++){
        if (i4 < Temporal_Size/2) set_array_value(arr,i1,i2,i3,i4,i5,1);
        else{
            x0 = (sample_uniform(rng_local) >= 0.5) ? 1 : -1;
            set_array_value(arr,i1,i2,i3,i4,i5,x0);
        } 
    }
    return;
}

void multicanonical_start_right_cold(Link_array& arr, std::mt19937 &rng_local){
    int x0; 
    for (int i1 = 0; i1 < Spatial_Size; i1++)
    for (int i2 = 0; i2 < Spatial_Size; i2++)
    for (int i3 = 0; i3 < Spatial_Size; i3++)
    for (int i4 = 0; i4 < Temporal_Size; i4++)
    for (int i5 = 0; i5 < 4; i5++){
        if (i4 > Temporal_Size/2) set_array_value(arr,i1,i2,i3,i4,i5,1);
        else{
            x0 = (sample_uniform(rng_local) >= 0.5) ? 1 : -1;
            set_array_value(arr,i1,i2,i3,i4,i5,x0);
        } 
    }
    return;
}

void alternating_multicanonical_start(Link_array& arr, std::mt19937 &rng_local, int parity){
    if (parity % 2 == 0) {
            multicanonical_start_left_cold(arr, rng_local);
            }
    if (parity % 2 == 1){
            multicanonical_start_right_cold(arr, rng_local);
            }
        return;
}



// For a 4 index array representing the lattice points this allows you to move in a positive direction in the lattice to a new node
void moveup(std::array<int,4>& v,int d){
    v[d]+=1;
    if (d==3){
        if (v[d]>=Temporal_Size) v[d]-=Temporal_Size;
    }
    else{
         if (v[d]>=Spatial_Size) v[d]-=Spatial_Size;
    }
    return;
}

// Same as moveup but in negative direction
void movedown(std::array<int,4>& v,int d){
    if (d == 3){
        if (v[d]==0) v[d]+=Temporal_Size-1;
        else v[d]-=1;
    }
    else {
        if (v[d]==0) v[d]+=Spatial_Size-1;
        else v[d]-=1;
    }
    return;
}

// Swaps elements of an array that are doubles
// void swap_double(std::array<double,CONFIG>& arr, int index1, int index2){
//     double tmp=arr[index1];
//     arr[index1]=arr[index2];
//     arr[index2]=tmp;
// }


// // Swaps elements of an array that are non-neg ints (int)
// void swap_int(std::array<int,CONFIG>& arr, int index1, int index2){
//     int tmp=arr[index1];
//     arr[index1]=arr[index2];
//     arr[index2]=tmp;
// }

double update_heatbath(Link_array& link, double beta, std::mt19937 &rng_local){
    std::array<int, 4> x; 
    int d,dperp,staple,staplesum;
    double bplus,bminus,avg_plaq=0.0;
    for (x[0]=0; x[0]<Spatial_Size; x[0]++)
        for (x[1]=0; x[1]<Spatial_Size; x[1]++)
            for (x[2]=0; x[2]<Spatial_Size; x[2]++)
                for (x[3]=0; x[3]<Temporal_Size; x[3]++)
                    for (d=0; d<4; d++) {
                        staplesum=0;
                        for (dperp=0;dperp<4;dperp++){
                            if (dperp!=d){
                                std::array<int, 4> y=x; 
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
                        
            /* the heatbath algorithm */
                        bplus=std::exp(beta*staplesum);
                        bminus=1/bplus;
                        bplus=bplus/(bplus+bminus);

                        if (sample_uniform(rng_local) < bplus){
                            set_array_value(link,x[0],x[1],x[2],x[3],d,1);
                            avg_plaq+=staplesum;
                        }
                        else {
                            set_array_value(link,x[0],x[1],x[2],x[3],d,-1);
                            avg_plaq-=staplesum;
                        }
                                              

                    }
    avg_plaq/=(Spatial_Size*Spatial_Size*Spatial_Size*Temporal_Size*4*6);
    return avg_plaq;
}

double return_avg_plaq(Link_array& link){
    std::array<int, 4> x; 
    int d,dperp,staple,staplesum;
    double avg_plaq=0.0;
    for (x[0]=0; x[0]<Spatial_Size; x[0]++)
        for (x[1]=0; x[1]<Spatial_Size; x[1]++)
            for (x[2]=0; x[2]<Spatial_Size; x[2]++)
                for (x[3]=0; x[3]<Temporal_Size; x[3]++)
                    for (d=0; d<4; d++) {
                        staplesum=0;
                        for (dperp=0;dperp<4;dperp++){
                            if (dperp!=d){
                                std::array<int, 4> y=x; 
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
                        int current_link = get_array_value(link,x[0],x[1],x[2],x[3],d);
                        avg_plaq+=staplesum*current_link;                                             
                    }
    avg_plaq/=(Spatial_Size*Spatial_Size*Spatial_Size*Temporal_Size*4*6);
    return avg_plaq;
}

void thermalize(Link_array& link, int therm_sweeps, double beta, std::mt19937 &rng_local){
    for (int i=0; i<therm_sweeps; i++){
        update_heatbath(link,beta,rng_local);
    }
}

// Autocorrelation Functions
double mean(const std::array<double,autocorrelation_sweeps>& data) {
    double sum=0.0;
    for (double x : data) sum += x;
    return sum / data.size();
}

double variance(const std::array<double,autocorrelation_sweeps>& data){
    double m=mean(data);
    double sum=0.0;
    for (double x : data){
        double diff= x-m;
        sum+= diff*diff;
    }
    return sum / data.size();
}


std::array<double,maxlag> autocorr(const std::array<double,autocorrelation_sweeps>& data) {
    std::array<double,maxlag> rho {};
    double m=mean(data);
    double var=variance(data);
    int N=data.size();
    if (var == 0.0){
        for (int t=0; t<maxlag; t++) rho[t] = (t==0)?1.0:0.0;
        return rho;
    }
    for (int t = 0; t < maxlag; t++){
        double c = 0.0;
         for (int i = 0; i < N - t; i++) {
            c += (data[i] - m) * (data[i + t] - m);
        }
        c /= (N- t);   
        rho[t] = c / var;
    }
    return rho;
}

double tau_int(const std::array<double,maxlag>& rho) {
    double tau = 1;
    for (int t = 1; t < maxlag; t++) {
        if (rho[t] < 0.05) break; 
        tau += rho[t];
    }
    return tau;
}

// Flatten array-of-arrays into a single contiguous array
std::array<int, CONFIG * one_config_array_size> flatten_links(const std::array<std::array<int, one_config_array_size>, CONFIG>& links){
    std::array<int, CONFIG * one_config_array_size> flat{};
    for (int i = 0; i < CONFIG; i++) {
        for (int j = 0; j < one_config_array_size; j++) {
            flat[i * one_config_array_size + j] = links[i][j];
        }
    }
    return flat;
}

// Un-flatten back into array-of-arrays
std::array<std::array<int, one_config_array_size>, CONFIG> unflatten_links(const std::array<int, CONFIG * one_config_array_size>& flat){
    std::array<std::array<int, one_config_array_size>, CONFIG> links{};
    for (int i = 0; i < CONFIG; i++) {
        for (int j = 0; j < one_config_array_size; j++) {
            links[i][j] = flat[i * one_config_array_size + j];
        }
    }
    return links;
}



// Save MC Config
// Saves the lattice Configuration at the end of a thermal cycle if not long enough

// Save all configurations at a coupling
void save_links_all(const std::array<std::array<int, one_config_array_size>, CONFIG>& links, const std::string& filename){
    // Flatten in memory for writing
    std::array<int, CONFIG * one_config_array_size> flat{};
    for (int i = 0; i < CONFIG; i++)
        for (int j = 0; j < one_config_array_size; j++)
            flat[i * one_config_array_size + j] = links[i][j];

    // Save as binary
    std::ofstream out(filename, std::ios::binary);
    if (!out) throw std::runtime_error("Cannot open file for writing: " + filename);

    out.write(reinterpret_cast<const char*>(flat.data()), flat.size() * sizeof(int));
}

// Save the beta array used in tempering
void save_PT_Beta_Array(const std::array<double,CONFIG>& array, const std::string& filename) {
    std::ofstream out(filename, std::ios::binary);
    out.write(reinterpret_cast<const char*>(array.data()),
              array.size() * sizeof(double));
}

// Save the beta index array used in tempering
void save_PT_index_Array(const std::array<int,CONFIG>& array, const std::string& filename) {
    std::ofstream out(filename, std::ios::binary);
    out.write(reinterpret_cast<const char*>(array.data()),
              array.size() * sizeof(int));
}

// Load in the saved beat array
std::array<double, CONFIG> load_PT_Beta_Array(const std::string& filename) {
    std::array<double, CONFIG> array{};
    std::ifstream in(filename, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Cannot open file: " + filename);
    }
    in.read(reinterpret_cast<char*>(array.data()), array.size() * sizeof(double));
    if (!in) {
        throw std::runtime_error("Error reading file: " + filename);
    }
    return array;
}

// Load int array (PT_index_Array)
std::array<int, CONFIG> load_PT_index_Array(const std::string& filename) {
    std::array<int, CONFIG> array{};
    std::ifstream in(filename, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Cannot open file: " + filename);
    }
    in.read(reinterpret_cast<char*>(array.data()), array.size() * sizeof(int));
    if (!in) {
        throw std::runtime_error("Error reading file: " + filename);
    }
    return array;
}

// Load all configurations at a coupling
std::array<Link_array, CONFIG> load_links_all(const std::string& filename){
    std::array<int, CONFIG * one_config_array_size> flat{}; // temporary flat array

    std::ifstream in(filename, std::ios::binary);
    if (!in) throw std::runtime_error("Cannot open file for reading: " + filename);

    in.read(reinterpret_cast<char*>(flat.data()), flat.size() * sizeof(int));
    if (!in) throw std::runtime_error("Error reading file: " + filename);

    // Reconstruct array-of-arrays
    std::array<Link_array, CONFIG> links{};
    for (int i = 0; i < CONFIG; i++)
        for (int j = 0; j < one_config_array_size; j++)
            links[i][j] = flat[i * one_config_array_size + j];

    return links;
}



// --- Find bin index ---
int find_bin(double plaq){
    int bin_number;
    bin_number = std::floor((plaq-min_bin)/bin_size);
    if (bin_number < 0) bin_number = 0;
    if (bin_number >= n_bins) bin_number = n_bins - 1;
    return bin_number;
}

// --- Load weight function ---
void load_weights(const std::string &filename) {
    std::ifstream fin(filename);
    double p, w;
    bin_edges.clear();
    weights.clear();
    while (fin >> p >> w) {
        bin_edges.push_back(p);
        weights.push_back(w);
    }
    // Trim to ensure indexing matches if Python output had extra edge
   if (bin_edges.size() != weights.size()) {
    std::cerr << "ERROR: mismatch -> edges=" 
              << bin_edges.size() 
              << " weights=" 
              << weights.size() << "\n";
    throw std::runtime_error("bin/weight size mismatch");
}
}


void single_link_multicanonical_update(Link_array &link, std::mt19937 &rng_local, double beta,int x0,int x1,int x2,int x3, int d,double &current_avg_plaq) {
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
    
    // Propose Flip
    // int old_link = get_array_value(link, x0, x1, x2, x3, d);
    // set_array_value(link, x0, x1, x2, x3, d, -old_link);
    // double proposed_avg_plaq = return_avg_plaq(link);
    // set_array_value(link, x0, x1, x2, x3, d, old_link);
    // int idx_old = find_bin(current_avg_plaq);
    // int idx_new = find_bin(proposed_avg_plaq);
    // double delta_W = weights[idx_new] - weights[idx_old];
    // double delta_S = -beta*(proposed_avg_plaq-current_avg_plaq)*N_plaq;
    // double exponent = -delta_S + delta_W;
    // plaq = current_avg_plaq;
    // if (exponent > 0) {
    //     set_array_value(link, x0, x1, x2, x3, d, -old_link);
    //     plaq = proposed_avg_plaq;
    // }
    // else {
    //     if (sample_uniform(rng_local) < std::exp(exponent)) {
    //     set_array_value(link, x0, x1, x2, x3, d, -old_link);
    //     plaq = proposed_avg_plaq;
    //     }
    // }
    // return plaq;
    // int old_link = get_array_value(link, x0, x1, x2, x3, d);
    // double delta_S = 2.0 * beta * old_link * staplesum; 
    // double delta_avg_plaq = ((-2.0 * old_link) * staplesum) / ((double)N_plaq);
    // double proposed_avg_plaq = current_avg_plaq + delta_avg_plaq;
    // // int idx_old = find_bin(current_avg_plaq);
    // // int idx_new = find_bin(proposed_avg_plaq);
    // // double delta_W = weights[idx_new] - weights[idx_old];
    // double delta_W = 0;
    // double exponent = -delta_S + delta_W;
    // plaq = current_avg_plaq;
    // if (exponent > 0) {
    //     set_array_value(link, x0, x1, x2, x3, d, -old_link);
    //     plaq = proposed_avg_plaq;
    // }
    // else {
    //     if (sample_uniform(rng_local) < std::exp(exponent)) {
    //     set_array_value(link, x0, x1, x2, x3, d, -old_link);
    //     plaq = proposed_avg_plaq;
    //     }
    // }
    // return plaq;
    
    int old_link = get_array_value(link, x0, x1, x2, x3, d);
    double delta_S = 2.0 * beta * old_link * staplesum;

    // double current_avg_plaq = return_avg_plaq(link);
    int idx_old = find_bin(current_avg_plaq);

    double delta_plaq = -2.0*old_link*staplesum/(N_plaq);
    double proposed_avg_plaq = current_avg_plaq+delta_plaq;
    
    // Check for avg plaq being the same; comment out when running big stuff
    // set_array_value(link, x0, x1, x2, x3, d, -old_link);
    // double computed_new = return_avg_plaq(link);
    // set_array_value(link, x0, x1, x2, x3, d, old_link);
    // if (std::fabs(proposed_avg_plaq - computed_new) > 1e-10) {
    // std::cout << "Mismatch at link " << d << " pos " << x0 << x1 << x2 << x3 
    //           << " | Expected: " << computed_new 
    //           << " | Proposed: " << proposed_avg_plaq 
    //           << " | Diff: " << (proposed_avg_plaq - computed_new) << std::endl;
    // exit(1);
    //     } // Stop immediately to see the first error

    int idx_new = find_bin(proposed_avg_plaq);
    double delta_W = weights[idx_new] - weights[idx_old];

    double exponent=-delta_S+delta_W;

    if (exponent > 0 || sample_uniform(rng_local) < std::exp(exponent)) {
        set_array_value(link, x0, x1, x2, x3, d, -old_link);
        current_avg_plaq = proposed_avg_plaq;
    }
    return;
}



double update_metropolis_multicanonical(Link_array &link, std::mt19937 &rng_local, double beta, double &current_avg_plaq){
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





//-----------------------------------------FINDING THE WEIGHTS--------------------------------------------------------------------------------
void mark_bin_visited(int idx,std::array<double, n_bins>& weight_function, std::array<int, n_bins>& reference_histogram, double constant){
    weight_function[idx] -= constant;
    reference_histogram[idx] += 1;
    return;
}

void all_bins_visited(std::array<int, n_bins>& reference_histogram, double& constant){
    int min_val = reference_histogram[0]; // Initialize with the first element
    for (int i = 1; i < n_bins; i++) {
        if (reference_histogram[i] < min_val) {
            min_val = reference_histogram[i];
        }
    }
    if (min_val>0){
        constant /= 2.0;
        std::cout<<"\n -----Halving C to "<<constant<<std::endl;
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
    return; 
}


void check_C(double constant, std::array<double, n_bins> weight_function){
    static bool done = false;

    if (!done && constant < tolerance){
        done = true;

        std::cout << "Done: Simulation is Jover" << std::endl;

        std::array<double,n_bins> edges;
        for (int i = 0; i < n_bins; i++){
            edges[i] = min_bin + i * bin_size;
        }

        double min_val = weight_function[0];
        for (int i = 1; i < n_bins; i++) {
            if (weight_function[i] < min_val) min_val = weight_function[i];
        }

        for (int i = 0; i < n_bins; i++) {
            weight_function[i] -= min_val;
        }

        write_lists_to_file(edges, weight_function);
    }
    return;
}



void single_link_multicanonical_update_BUILD_HIST(Link_array &link, std::mt19937 &rng_local, double beta,int x0,int x1,int x2,int x3, int d,double &current_avg_plaq, std::array<double, n_bins> weight_function) {
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
    int idx_old = find_bin(current_avg_plaq);

    double delta_plaq = -2.0*old_link*staplesum/(N_plaq);
    double proposed_avg_plaq = current_avg_plaq+delta_plaq;
    
    int idx_new = find_bin(proposed_avg_plaq);
    double delta_W = weight_function[idx_new] - weight_function[idx_old];

    double exponent=-delta_S+delta_W;

    if (exponent > 0 || sample_uniform(rng_local) < std::exp(exponent)) {
        set_array_value(link, x0, x1, x2, x3, d, -old_link);
        current_avg_plaq = proposed_avg_plaq;
    }
    // int current_idx = find_bin_SIMULATION(current_avg_plaq);
    // mark_bin_visited(current_idx,weight_function,reference_histogram,constant);
    // all_bins_visited(reference_histogram,constant);
    // check_C(constant, weight_function);
    // return;
    return;
}


double multicanonical_update_BUILD_HIST(Link_array &link, std::mt19937 &rng_local, double beta,double &current_avg_plaq, std::array<double, n_bins>& weight_function, std::array<int, n_bins>& reference_histogram, double& constant){
    std::array<int, 4> x;
    int d;
    for (x[0] = 0; x[0] < Spatial_Size; x[0]++)
    for (x[1] = 0; x[1] < Spatial_Size; x[1]++)
    for (x[2] = 0; x[2] < Spatial_Size; x[2]++)
    for (x[3] = 0; x[3] < Temporal_Size; x[3]++)
    for (d = 0; d < 4; d++) {
        single_link_multicanonical_update_BUILD_HIST(link, rng_local, beta, x[0], x[1], x[2], x[3],  d,current_avg_plaq, weight_function);
    }
    int current_idx = find_bin(current_avg_plaq);
    mark_bin_visited(current_idx,weight_function,reference_histogram,constant);
    all_bins_visited(reference_histogram,constant);
    check_C(constant, weight_function);
    return current_avg_plaq;
}