#include <iostream>
#include <cstdlib>
#include <string>
#include <sstream>
#include <fstream>
#include <chrono>
#include "utilityCore.hpp"
#include "glslUtility.hpp"
#include "visualizer.h"
#include "kernel.h"
#include "data_handler.h"
#include "sim_cpu.h"

using namespace std;

void print_str(std::string str) {
    printf("%s\n", str.c_str());
}

void print_float(float x) {
    printf("%f\n", x);
}

void print_double(double x) {
    printf("%f\n", x);
}
 
void print_int(int x) {
    printf("%d\n", x);
}

void print_arr(int* l, int size) {
    for (int ind = 0; ind < size; ind++) {
        std::cout << l[ind] << std::endl;
    }
    std::cout << std::endl;
}

void print_Array(ArrayXd m) {
    std::cout << m << std::endl;
}



/**
* C main function.
*/
int main(int argc, char* argv[]) {
    print_str("CUDA_Cardiac_Electrophysiological_Modeling");
    print_str("Reading in data.");
    //simulation_inputs sim_inputs = import_para(string(argv[1]), string(argv[2]));
    simulation_inputs sim_inputs = import_para("data/sim_settings.json", "data/sim_input.json");

    print_str("Running simulation.");
    simulation_outputs sim_output;
    auto start = chrono::high_resolution_clock::now();
    if (sim_inputs.visualization == 1) {
        initGLFW(sim_inputs);
    }
    
    
    if (sim_inputs.use_gpu != 1) {
        sim_output = Simulation_CPU(sim_inputs);
    }
    else {
        //simulation_data_parts data_parts = prepare_parts_naive(sim_inputs);
        //Cardiac::initSimulation_naive(sim_inputs, data_parts);
        //sim_output = Cardiac::runSimulation_naive(sim_inputs);
        //Cardiac::endSimulation_naive();

        if (sim_inputs.use_float == 1) {
            simulation_data_parts data_parts = prepare_parts_float(sim_inputs);
            Cardiac::initSimulation_float(sim_inputs, data_parts);
            sim_output = Cardiac::runSimulation_float(sim_inputs);
            Cardiac::endSimulation_float();
        }
        else {
            simulation_data_parts data_parts = prepare_parts(sim_inputs);
            Cardiac::initSimulation(sim_inputs, data_parts);
            sim_output = Cardiac::runSimulation(sim_inputs);
            Cardiac::endSimulation();
        }
        
    }
    auto duration = chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now() - start);
    cout << "Computation took " << duration.count() << " ms." << endl;

    if (sim_inputs.save_result == 1) {
        print_str("Saving results.");
        save_data(sim_inputs, sim_output, sim_inputs.save_data_min_clip);
    }

    if (sim_inputs.visualization == 1) {
        std::cout << "Running Visualization." << std::endl;
        visualizationLoop(sim_output, sim_inputs);
        Cardiac::endVisulization();
    }
    return 0;
}


