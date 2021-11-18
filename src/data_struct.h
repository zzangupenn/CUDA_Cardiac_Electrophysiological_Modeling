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
    int* count;
};

struct simulation_inputs {
    double dt;
    int final_t;
    double tau_in;
    double tau_out;
    double tau_open;
    double tau_close;
    double v_gate;
    double c_for_D;
    int n_voxel;
    double* D = new double[9];
    glm::vec3* voxel;
    int** voxel_neighborhood;
    int* boundary_flag;
    J_stim_struct J_stim;
};

struct simulation_outputs {
    int n_voxel;
    int n_step;
    double** action_potentials;
};

struct simulation_settings {
    bool save_result;
    string save_result_filename;
    bool visualization;
    int visualization_resolution[2];
    bool save_visulization;
    string save_visulization_filename;
};