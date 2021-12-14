#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <string>
using namespace std;

struct J_stim_struct {
    int n_voxel; // number of voxels
    int* voxel_ind;
    int* step;
    double* value;
    float* value_f;
    int* count;
};

struct simulation_inputs {
    float dt_f;
    double dt;
    int final_t;
    int save_result;
    int use_gpu;
    int use_float;
    int use_naive;
    string save_result_filename;
    int visualization;
    int visualization_loop;
    int* visualization_resolution;
    int save_visulization;
    string save_visulization_filename;
    double save_data_min_clip;
    int num_of_dt_per_save;

    double delta_sqr;
    float delta_sqr_f;
    double* tau_in;
    double* tau_out;
    double* tau_open;
    double* tau_close;
    double* v_gate;
    double* c_for_D;
    int n_voxel;
    double** D;
    glm::vec3* voxel;
    int** voxel_neighborhood;
    J_stim_struct J_stim;
};

struct simulation_outputs {
    int n_voxel;
    int n_step;
    double** action_potentials;
    float** action_potentials_f;
    double data_min;
    double data_max;
};

struct simulation_data_parts {
    double* part1;
    double* part2;
    double* part3;
    double* part4;
    double* part5;
    double* part6;
    double* part7;
    double* part8;
    double* part9;
    double* part10;
    double* part11;
    double* part12;
    double* part13;
    double* part14;
    double* part15;
    int* indices;
    double* coefficients;
    double* sim_h_init;
    float* coefficients_f;
    float* sim_h_init_f;
};