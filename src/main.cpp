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

using namespace std;
using namespace Eigen;

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

ArrayXd sign_int(ArrayXi v) {
    ArrayXd v_ret = (v > 0).select(ArrayXd::Ones(v.rows()), ArrayXd::Zero(v.rows()));
    return v_ret;
}

simulation_outputs Simulation_CPU(simulation_inputs sim_inputs) {

    Map<ArrayXi> px (sim_inputs.voxel_neighborhood[0], sim_inputs.n_voxel);
    Map<ArrayXi> mx (sim_inputs.voxel_neighborhood[1], sim_inputs.n_voxel);
    Map<ArrayXi> py (sim_inputs.voxel_neighborhood[2], sim_inputs.n_voxel);
    Map<ArrayXi> my (sim_inputs.voxel_neighborhood[3], sim_inputs.n_voxel);
    Map<ArrayXi> pz (sim_inputs.voxel_neighborhood[4], sim_inputs.n_voxel);
    Map<ArrayXi> mz (sim_inputs.voxel_neighborhood[5], sim_inputs.n_voxel);
    Map<ArrayXi> pxpy (sim_inputs.voxel_neighborhood[6], sim_inputs.n_voxel);
    Map<ArrayXi> mxpy (sim_inputs.voxel_neighborhood[7], sim_inputs.n_voxel);
    Map<ArrayXi> pxmy (sim_inputs.voxel_neighborhood[8], sim_inputs.n_voxel);
    Map<ArrayXi> mxmy (sim_inputs.voxel_neighborhood[9], sim_inputs.n_voxel);
    Map<ArrayXi> pypz (sim_inputs.voxel_neighborhood[10], sim_inputs.n_voxel);
    Map<ArrayXi> mypz (sim_inputs.voxel_neighborhood[11], sim_inputs.n_voxel);
    Map<ArrayXi> pymz (sim_inputs.voxel_neighborhood[12], sim_inputs.n_voxel);
    Map<ArrayXi> mymz (sim_inputs.voxel_neighborhood[13], sim_inputs.n_voxel);
    Map<ArrayXi> pxpz (sim_inputs.voxel_neighborhood[14], sim_inputs.n_voxel);
    Map<ArrayXi> mxpz (sim_inputs.voxel_neighborhood[15], sim_inputs.n_voxel);
    Map<ArrayXi> pxmz (sim_inputs.voxel_neighborhood[16], sim_inputs.n_voxel);
    Map<ArrayXi> mxmz (sim_inputs.voxel_neighborhood[17], sim_inputs.n_voxel);
    Map<Matrix3d> D(sim_inputs.D);

    ArrayXd part1 = 4.0 * D(0, 0) * sign_int(px);
    ArrayXd part2 = 4.0 * D(0, 0) * sign_int(mx);
    ArrayXd part3 = 4.0 * D(1, 1) * sign_int(py);
    ArrayXd part4 = 4.0 * D(1, 1) * sign_int(my);
    ArrayXd part5 = 4.0 * D(2, 2) * sign_int(pz);
    ArrayXd part6 = 4.0 * D(2, 2) * sign_int(mz);
    ArrayXd part7 = ArrayXd::Zero(part1.rows());
    ArrayXd part8 = ArrayXd::Zero(part1.rows());
    ArrayXd part9 = ArrayXd::Zero(part1.rows());
    ArrayXd part10 = 2.0 * D(0, 1) * sign_int(pxpy) * sign_int(pxmy);
    ArrayXd part11 = 2.0 * D(0, 1) * sign_int(mxpy) * sign_int(mxmy);
    ArrayXd part12 = 2.0 * D(0, 2) * sign_int(pxpz) * sign_int(pxmz);
    ArrayXd part13 = 2.0 * D(0, 2) * sign_int(mxpz) * sign_int(mxmz);
    ArrayXd part14 = 2.0 * D(1, 2) * sign_int(pypz) * sign_int(pymz);
    ArrayXd part15 = 2.0 * D(1, 2) * sign_int(mypz) * sign_int(mymz);

    int total_step = sim_inputs.final_t / sim_inputs.dt;
    ArrayXXd sim_v = ArrayXXd::Zero(part1.rows(), total_step);
    ArrayXd sim_v_current = sim_v(all, 0);
    ArrayXd sim_h = ArrayXd::Ones(part1.rows());
    ArrayXd c_voxel = ArrayXd::Ones(part1.rows());
    ArrayXd sim_h_1 = ArrayXd::Zero(part1.rows());
    ArrayXd sim_h_2 = ArrayXd::Zero(part1.rows());
    ArrayXd diffusion_term = ArrayXd::Zero(part1.rows());
    ArrayXd J_stim_voxel = ArrayXd::Zero(part1.rows());
    double delta = 2.0;
    double v_gate = 0.13;
    int J_stim_min_ind = INT_MAX; 
    int J_stim_max_ind = 0;
    for (int ind = 0; ind < sim_inputs.J_stim.count[sim_inputs.J_stim.n_voxel - 1]; ind++) {
        if (J_stim_min_ind > sim_inputs.J_stim.step[ind]) J_stim_min_ind = sim_inputs.J_stim.step[ind];
        if (J_stim_max_ind < sim_inputs.J_stim.step[ind]) J_stim_max_ind = sim_inputs.J_stim.step[ind];
    }
    simulation_outputs sim_output;
    sim_output.n_step = total_step;
    sim_output.n_voxel = sim_inputs.n_voxel;
    sim_output.action_potentials = new double*[total_step];
    for (int ind = 0; ind < total_step; ind++) {
        sim_output.action_potentials[ind] = new double[sim_inputs.n_voxel];
    }
    print_str("here");
    //int test_list[4] = { 1, 2, 3, 4 };
    ////ArrayXi test = ArrayXi::Ones(4);
    //Map<ArrayXi> test(test_list, 4);
    //int test_list2[4] = { 0, 2, 1, 3 };
    ////ArrayXi test = ArrayXi::Ones(4);
    //Map<ArrayXi> test2(test_list2, 4);
    //std::cout << test << std::endl;
    //std::cout << (test > 2).select(test, test2) << std::endl;


    auto start = chrono::high_resolution_clock::now();
    for (int ind = 0; ind < total_step - 1; ind++) {
        if (ind > 0) {
            sim_v_current = sim_v(all, ind - 1);
        }
        if (ind % 1000 == 0) {
            printf("%d\n", ind);
        }
        J_stim_voxel = ArrayXd::Zero(part1.rows());
        if (ind >= J_stim_min_ind && ind <= J_stim_max_ind) {
            for (int count_ind = 0; count_ind < sim_inputs.J_stim.n_voxel; count_ind++) {
                for (int step_ind = sim_inputs.J_stim.count[count_ind]; step_ind < sim_inputs.J_stim.count[count_ind+1]; step_ind++) {
                    if (sim_inputs.J_stim.step[step_ind] - 1 == ind) {
                        J_stim_voxel(sim_inputs.J_stim.voxel_ind[count_ind]) = sim_inputs.J_stim.value[step_ind];
                    }
                }
            }
        }

        diffusion_term = 1.0 / (4.0 * pow(delta, 2.0)) *
                ((sim_v_current(px) - sim_v_current) * part1 + (sim_v_current(mx) - sim_v_current) * part2 +
                (sim_v_current(py) - sim_v_current) * part3 + (sim_v_current(my) - sim_v_current) * part4 +
                (sim_v_current(pz) - sim_v_current) * part5 + (sim_v_current(mz) - sim_v_current) * part6 +
                (sim_v_current(px) - sim_v_current(mx)) * part7 +
                (sim_v_current(py) - sim_v_current(my)) * part8 +
                (sim_v_current(pz) - sim_v_current(mz)) * part9 +
                (sim_v_current(pxpy) - sim_v_current(pxmy)) * part10 + (sim_v_current(mxmy) - sim_v_current(mxpy)) * part11 +
                (sim_v_current(pxpz) - sim_v_current(pxmz)) * part12 + (sim_v_current(mxmz) - sim_v_current(mxpz)) * part13 +
                (sim_v_current(pypz) - sim_v_current(pymz)) * part14 + (sim_v_current(mymz) - sim_v_current(mypz)) * part15);
        sim_v(all, ind + 1) = ((sim_h * sim_v(all, ind).pow(2.0) * (1 - sim_v(all, ind)) / sim_inputs.tau_in) + (-sim_v(all, ind) / sim_inputs.tau_out) + J_stim_voxel + diffusion_term) * sim_inputs.dt + sim_v(all, ind);
        sim_h_1 = ((1 - sim_h) / sim_inputs.tau_open) * sim_inputs.dt + sim_h;
        sim_h_2 = (-sim_h / sim_inputs.tau_close) * sim_inputs.dt + sim_h;
        sim_h = (sim_v_current < v_gate).select(sim_h_1, sim_h_2);
        
        ArrayXd::Map(sim_output.action_potentials[ind], sim_output.n_voxel) = sim_v(all, ind + 1);
        if (ind == 1503) {
            //for (int ind = 0; ind < sim_output.n_voxel; ind++) {
            //    if (diffusion_term(ind) != 0) {
            print_int(ind);
            //        print_float(diffusion_term(ind));
            std::cout << sim_v_current(seq(90, 100)) << std::endl;
            //    }
            //}
        //    
        //    
        //    
        //    //printf("%f\n", diffusion_term(seq(90, 100)));
        //    //}
        }

    }
    auto duration = chrono::duration_cast<chrono::seconds>(chrono::high_resolution_clock::now() - start);
    cout << duration.count() << " seconds" << endl;
    
    return sim_output;
}

/**
* C main function.
*/
int main(int argc, char* argv[]) {
    simulation_inputs sim_inputs = import_para("data/sim_settings.json",
        "data/volume.json",
        "data/J_stim.json");
    
    simulation_outputs sim_output = Simulation_CPU(sim_inputs);
    //for (int ind = 0; ind < sim_output.n_voxel; ind++) {
    //    std::cout << sim_output.action_potentials[10000][ind] << std::endl;
    //}

    if (init(sim_inputs)) {
        mainLoop(sim_output);
        Cardiac::endSimulation();
        return 0;
    } else {
        return 1;
    }
}


