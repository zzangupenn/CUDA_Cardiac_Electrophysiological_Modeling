#define GLM_FORCE_CUDA
#include <stdio.h>
#include <cuda.h>
#include <cmath>
#include <glm/glm.hpp>
#include "utilityCore.hpp"
#include "kernel.h"

using namespace std;

#define checkCUDAErrorWithLine(msg) checkCUDAError(msg, __LINE__)

/**
* Check for CUDA errors; print and exit if there was a problem.
*/
void checkCUDAError(const char *msg, int line = -1) {
  cudaError_t err = cudaGetLastError();
  if (cudaSuccess != err) {
    if (line >= 0) {
      fprintf(stderr, "Line %d: ", line);
    }
    fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

/*****************
* Configuration *
*****************/

#define SPLIT_NUM 1

/*! Block size used for CUDA kernel launch. */
#define blockSize 32

/*! Size of the starting area in simulation space. */
#define scene_scale 25.0f

/***********************************************
* Kernel state (pointers are device pointers) *
***********************************************/

int numObjects;
dim3 threadsPerBlock(blockSize);

glm::vec3* dev_pos;
glm::vec3* dev_ap_gl;
double* sim_v1;
double* sim_v2;
double* sim_h;
int* indices;
double* coefficients;
double* dt_delta;
int* n_voxel;

int* J_stim_ind_min_max;
int* J_stim_ind_min_max_h = new int[2];
int* J_stim_step;
int* J_stim_n_voxel;
int* J_stim_voxel_ind;
double* J_stim_value;
int* J_stim_count;

float* sim_v1_f;
float* sim_v2_f;
float* sim_h_f;
float* coefficients_f;
float* dt_delta_f;
float* J_stim_value_f;

/******************
* Simulation * FLOAT *****************************************************************************
******************/

void Cardiac::initSimulation_float(simulation_inputs sim_inputs, simulation_data_parts data_parts) {
    int N = sim_inputs.n_voxel;
    numObjects = sim_inputs.n_voxel;
    dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);
    int voxel_n_by_blockSize = (numObjects / blockSize + 1) * blockSize;
    cudaMalloc((void**)&indices, voxel_n_by_blockSize * 18 * sizeof(int));
    cudaMemset((void*)indices, 0, voxel_n_by_blockSize * 18 * sizeof(int));
    cudaMemcpy(indices, data_parts.indices, voxel_n_by_blockSize * 18 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&coefficients_f, voxel_n_by_blockSize * 20 * sizeof(float));
    cudaMemcpy(coefficients_f, data_parts.coefficients_f, voxel_n_by_blockSize * 20 * sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc((void**)&dt_delta_f, 2 * sizeof(float));
    cudaMemcpy(dt_delta_f, &(sim_inputs.dt_f), sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dt_delta_f + 1, &(sim_inputs.delta_sqr_f), sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&n_voxel, sizeof(int));
    cudaMemcpy(n_voxel, &(sim_inputs.n_voxel), sizeof(int), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&sim_v1_f, (numObjects + 1) * sizeof(float));
    cudaMemset((void*)sim_v1_f, 0, (numObjects + 1) * sizeof(float));
    cudaMalloc((void**)&sim_v2_f, (numObjects + 1) * sizeof(float));
    cudaMemset((void*)sim_v2_f, 0, (numObjects + 1) * sizeof(float));
    cudaMalloc((void**)&sim_h_f, N * sizeof(float));
    cudaMemcpy(sim_h_f, data_parts.sim_h_init_f, N * sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc((void**)&J_stim_n_voxel, sizeof(int));
    cudaMemcpy(J_stim_n_voxel, &(sim_inputs.J_stim.n_voxel), sizeof(int), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&J_stim_voxel_ind, sim_inputs.J_stim.n_voxel * sizeof(int));
    cudaMemcpy(J_stim_voxel_ind, sim_inputs.J_stim.voxel_ind, sim_inputs.J_stim.n_voxel * sizeof(int), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&J_stim_count, sim_inputs.J_stim.n_voxel * sizeof(int));
    cudaMemcpy(J_stim_count, sim_inputs.J_stim.count, sim_inputs.J_stim.n_voxel * sizeof(int), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&J_stim_step, sim_inputs.J_stim.count[sim_inputs.J_stim.n_voxel - 1] * sizeof(int));
    cudaMemcpy(J_stim_step, sim_inputs.J_stim.step, sim_inputs.J_stim.count[sim_inputs.J_stim.n_voxel - 1] * sizeof(int), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&J_stim_value_f, N * sizeof(float));
    cudaMemcpy(J_stim_value_f, sim_inputs.J_stim.value_f, sim_inputs.J_stim.count[sim_inputs.J_stim.n_voxel - 1] * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&J_stim_ind_min_max, 2 * sizeof(int));

    J_stim_ind_min_max_h[0] = INT_MAX;
    J_stim_ind_min_max_h[1] = 0;
    for (int ind = 0; ind < sim_inputs.J_stim.count[sim_inputs.J_stim.n_voxel - 1]; ind++) {
        if (J_stim_ind_min_max_h[0] > sim_inputs.J_stim.step[ind]) J_stim_ind_min_max_h[0] = sim_inputs.J_stim.step[ind];
        if (J_stim_ind_min_max_h[1] < sim_inputs.J_stim.step[ind]) J_stim_ind_min_max_h[1] = sim_inputs.J_stim.step[ind];
    }
    cudaMemcpy(J_stim_ind_min_max, J_stim_ind_min_max_h, 2 * sizeof(int), cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();
}

__global__ void simulation_kernel_float(int step, float* sim_v1, float* dt_delta,
    float* sim_v2,
    float* sim_h,
    int* J_stim_step,
    int* J_stim_n_voxel,
    int* J_stim_voxel_ind,
    float* J_stim_value,
    int* J_stim_count,
    int* J_stim_ind_min_max,
    int* n_voxel,
    int* indices_,
    float* coefficients_) {

    for (int sub_ind = 0; sub_ind < SPLIT_NUM; sub_ind++) {
        int blockIdx_x_times_SPLIT_NUM = blockIdx.x * SPLIT_NUM;
        int index = threadIdx.x + ((blockIdx_x_times_SPLIT_NUM + sub_ind) * blockDim.x);
        if (index > n_voxel[0] - 1) {
            return;
        }
        int* indices = indices_ + ((blockIdx_x_times_SPLIT_NUM + sub_ind) * blockDim.x) * 18;
        float* coefficients = coefficients_ + ((blockIdx_x_times_SPLIT_NUM + sub_ind) * blockDim.x) * 20;

        float sim_v1_index = sim_v1[index];
        float diffusion_term = 1.0 / (4.0 * dt_delta[1]) *
            ((sim_v1[indices[threadIdx.x + blockDim.x * 0]] - sim_v1_index) * coefficients[threadIdx.x + blockDim.x * 0] + (sim_v1[indices[threadIdx.x + blockDim.x * 1]] - sim_v1_index) * coefficients[threadIdx.x + blockDim.x * 1] +
                (sim_v1[indices[threadIdx.x + blockDim.x * 2]] - sim_v1_index) * coefficients[threadIdx.x + blockDim.x * 2] + (sim_v1[indices[threadIdx.x + blockDim.x * 3]] - sim_v1_index) * coefficients[threadIdx.x + blockDim.x * 3] +
                (sim_v1[indices[threadIdx.x + blockDim.x * 4]] - sim_v1_index) * coefficients[threadIdx.x + blockDim.x * 4] + (sim_v1[indices[threadIdx.x + blockDim.x * 5]] - sim_v1_index) * coefficients[threadIdx.x + blockDim.x * 5] +
                (sim_v1[indices[threadIdx.x + blockDim.x * 0]] - sim_v1[indices[threadIdx.x + blockDim.x * 1]]) * coefficients[threadIdx.x + blockDim.x * 6] +
                (sim_v1[indices[threadIdx.x + blockDim.x * 2]] - sim_v1[indices[threadIdx.x + blockDim.x * 3]]) * coefficients[threadIdx.x + blockDim.x * 7] +
                (sim_v1[indices[threadIdx.x + blockDim.x * 4]] - sim_v1[indices[threadIdx.x + blockDim.x * 5]]) * coefficients[threadIdx.x + blockDim.x * 8] +
                (sim_v1[indices[threadIdx.x + blockDim.x * 6]] - sim_v1[indices[threadIdx.x + blockDim.x * 8]]) * coefficients[threadIdx.x + blockDim.x * 9] + (sim_v1[indices[threadIdx.x + blockDim.x * 9]] - sim_v1[indices[threadIdx.x + blockDim.x * 7]]) * coefficients[threadIdx.x + blockDim.x * 10] +
                (sim_v1[indices[threadIdx.x + blockDim.x * 14]] - sim_v1[indices[threadIdx.x + blockDim.x * 16]]) * coefficients[threadIdx.x + blockDim.x * 10] + (sim_v1[indices[threadIdx.x + blockDim.x * 17]] - sim_v1[indices[threadIdx.x + blockDim.x * 15]]) * coefficients[threadIdx.x + blockDim.x * 12] +
                (sim_v1[indices[threadIdx.x + blockDim.x * 10]] - sim_v1[indices[threadIdx.x + blockDim.x * 12]]) * coefficients[threadIdx.x + blockDim.x * 13] + (sim_v1[indices[threadIdx.x + blockDim.x * 13]] - sim_v1[indices[threadIdx.x + blockDim.x * 11]]) * coefficients[threadIdx.x + blockDim.x * 14]);

    //    float diffusion_term = 1.0 / (4.0 * dt_delta[1]) *
    //((sim_v1[indices[threadIdx.x + blockSize * 0]] - sim_v1_index) * coefficients[threadIdx.x + blockSize * 0] + (sim_v1[indices[threadIdx.x + blockSize * 1]] - sim_v1_index) * coefficients[threadIdx.x + blockSize * 1] +
    //    (sim_v1[indices[threadIdx.x + blockSize * 2]] - sim_v1_index) * coefficients[threadIdx.x + blockSize * 2] + (sim_v1[indices[threadIdx.x + blockSize * 3]] - sim_v1_index) * coefficients[threadIdx.x + blockSize * 3] +
    //    (sim_v1[indices[threadIdx.x + blockSize * 4]] - sim_v1_index) * coefficients[threadIdx.x + blockSize * 4] + (sim_v1[indices[threadIdx.x + blockSize * 5]] - sim_v1_index) * coefficients[threadIdx.x + blockSize * 5] +
    //    (sim_v1[indices[threadIdx.x + blockSize * 0]] - sim_v1[indices[threadIdx.x + blockSize * 1]]) * coefficients[threadIdx.x + blockSize * 6] +
    //    (sim_v1[indices[threadIdx.x + blockSize * 2]] - sim_v1[indices[threadIdx.x + blockSize * 3]]) * coefficients[threadIdx.x + blockSize * 7] +
    //    (sim_v1[indices[threadIdx.x + blockSize * 4]] - sim_v1[indices[threadIdx.x + blockSize * 5]]) * coefficients[threadIdx.x + blockSize * 8] +
    //    (sim_v1[indices[threadIdx.x + blockSize * 6]] - sim_v1[indices[threadIdx.x + blockSize * 8]]) * coefficients[threadIdx.x + blockSize * 9] + (sim_v1[indices[threadIdx.x + blockSize * 9]] - sim_v1[indices[threadIdx.x + blockSize * 7]]) * coefficients[threadIdx.x + blockSize * 10] +
    //    (sim_v1[indices[threadIdx.x + blockSize * 14]] - sim_v1[indices[threadIdx.x + blockSize * 16]]) * coefficients[threadIdx.x + blockSize * 10] + (sim_v1[indices[threadIdx.x + blockSize * 17]] - sim_v1[indices[threadIdx.x + blockSize * 15]]) * coefficients[threadIdx.x + blockSize * 12] +
    //    (sim_v1[indices[threadIdx.x + blockSize * 10]] - sim_v1[indices[threadIdx.x + blockSize * 12]]) * coefficients[threadIdx.x + blockSize * 13] + (sim_v1[indices[threadIdx.x + blockSize * 13]] - sim_v1[indices[threadIdx.x + blockSize * 11]]) * coefficients[threadIdx.x + blockSize * 14]);


        float J_stim = 0.0;
        [&] {
            if (step >= J_stim_ind_min_max[0] && step <= J_stim_ind_min_max[1]) {
                for (int count_ind = 0; count_ind < J_stim_n_voxel[0]; count_ind++) {
                    if (index == J_stim_voxel_ind[count_ind]) {
                        for (int step_ind = J_stim_count[count_ind]; step_ind < J_stim_count[count_ind + 1]; step_ind++) {
                            if (J_stim_step[step_ind] - 1 == step) {
                                J_stim = J_stim_value[step_ind];
                                return;
                            }
                        }
                    }
                }
            }
        }();

        float sim_h_index = sim_h[index];
        sim_v2[index] = ((sim_h_index * sim_v1_index * sim_v1_index * coefficients[threadIdx.x + blockDim.x * 17] * (1 - sim_v1_index)) + (-sim_v1_index * coefficients[threadIdx.x + blockDim.x * 18]) + J_stim + diffusion_term) * dt_delta[0] + sim_v1_index;
        if (sim_v1_index < coefficients[threadIdx.x + blockDim.x * 19]) {
            sim_h[index] = ((1 - sim_h_index) * coefficients[threadIdx.x + blockDim.x * 15]) * dt_delta[0] + sim_h_index;
        }
        else {
            sim_h[index] = (-sim_h_index * coefficients[threadIdx.x + blockDim.x * 16]) * dt_delta[0] + sim_h_index;
        }

        //float sim_h_index = sim_h[index];
        //sim_v2[index] = ((sim_h_index * sim_v1_index * sim_v1_index * coefficients[threadIdx.x + blockSize * 17] * (1 - sim_v1_index)) + (-sim_v1_index * coefficients[threadIdx.x + blockSize * 18]) + J_stim + diffusion_term) * dt_delta[0] + sim_v1_index;
        //if (sim_v1_index < coefficients[threadIdx.x + blockSize * 19]) {
        //    sim_h[index] = ((1 - sim_h_index) * coefficients[threadIdx.x + blockSize * 15]) * dt_delta[0] + sim_h_index;
        //}
        //else {
        //    sim_h[index] = (-sim_h_index * coefficients[threadIdx.x + blockSize * 16]) * dt_delta[0] + sim_h_index;
        //}

    }

    return;
}

simulation_outputs Cardiac::runSimulation_float(simulation_inputs sim_inputs) {
    dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize / SPLIT_NUM);

    simulation_outputs sim_output;
    int total_step = sim_inputs.final_t / sim_inputs.dt;
    sim_output.n_step = total_step;
    sim_output.n_voxel = sim_inputs.n_voxel;
    sim_output.data_min = sim_inputs.v_gate[0];
    sim_output.data_max = 1.0;
    int total_save = total_step / sim_inputs.num_of_dt_per_save;
    sim_output.action_potentials_f = new float* [total_save];
    int save_ind = 0;

    for (int ind = 0; ind < total_step; ind++) {
        if (ind % 1000 == 0) {
            printf("%d\n", ind);
        }

        simulation_kernel_float << < fullBlocksPerGrid, blockSize >> > (ind, sim_v1_f,
            dt_delta_f,
            sim_v2_f,
            sim_h_f,
            J_stim_step,
            J_stim_n_voxel,
            J_stim_voxel_ind,
            J_stim_value_f,
            J_stim_count,
            J_stim_ind_min_max,
            n_voxel,
            indices,
            coefficients_f);
        cudaDeviceSynchronize();
        std::swap(sim_v1_f, sim_v2_f);
        if (ind % sim_inputs.num_of_dt_per_save == 0 || ind == total_step - 1) {
            sim_output.action_potentials_f[save_ind] = new float[numObjects];
            cudaMemcpy(sim_output.action_potentials_f[save_ind], sim_v1_f, numObjects * sizeof(float), cudaMemcpyDeviceToHost);
            save_ind++;
        }


    }
    return sim_output;
}


void Cardiac::endSimulation_float() {
    cudaFree(sim_v1_f);
    cudaFree(dt_delta_f);
    cudaFree(sim_v2_f);
    cudaFree(sim_h_f);
    cudaFree(indices);
    cudaFree(coefficients_f);
    cudaFree(J_stim_step);
    cudaFree(J_stim_n_voxel);
    cudaFree(J_stim_voxel_ind);
    cudaFree(J_stim_value_f);
    cudaFree(J_stim_count);
    cudaFree(J_stim_ind_min_max);
    cudaDeviceSynchronize();
}

/******************
* Simulation * DOUBLE *****************************************************************************
******************/

void Cardiac::initSimulation(simulation_inputs sim_inputs, simulation_data_parts data_parts) {
    int N = sim_inputs.n_voxel;
    numObjects = sim_inputs.n_voxel;
    dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);
    int voxel_n_by_blockSize = (numObjects / blockSize + 1) * blockSize;
    cudaMalloc((void**)&indices, voxel_n_by_blockSize * 18 * sizeof(int));
    cudaMemset((void*)indices, 0, voxel_n_by_blockSize * 18 * sizeof(int));
    cudaMemcpy(indices, data_parts.indices, voxel_n_by_blockSize * 18 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&coefficients, voxel_n_by_blockSize * 20 * sizeof(double));
    cudaMemcpy(coefficients, data_parts.coefficients, voxel_n_by_blockSize * 20 * sizeof(double), cudaMemcpyHostToDevice);

    cudaMalloc((void**)&dt_delta, 2 * sizeof(double));
    cudaMemcpy(dt_delta, &(sim_inputs.dt), sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dt_delta + 1, &(sim_inputs.delta_sqr), sizeof(double), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&n_voxel, sizeof(int));
    cudaMemcpy(n_voxel, &(sim_inputs.n_voxel), sizeof(int), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&sim_v1, (numObjects + 1) * sizeof(double));
    cudaMemset((void*)sim_v1, 0, (numObjects + 1) * sizeof(double));
    cudaMalloc((void**)&sim_v2, (numObjects + 1) * sizeof(double));
    cudaMemset((void*)sim_v2, 0, (numObjects + 1) * sizeof(double));
    cudaMalloc((void**)&sim_h, N * sizeof(double));
    cudaMemcpy(sim_h, data_parts.sim_h_init, N * sizeof(double), cudaMemcpyHostToDevice);

    cudaMalloc((void**)&J_stim_n_voxel, sizeof(int));
    cudaMemcpy(J_stim_n_voxel, &(sim_inputs.J_stim.n_voxel), sizeof(int), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&J_stim_voxel_ind, sim_inputs.J_stim.n_voxel * sizeof(int));
    cudaMemcpy(J_stim_voxel_ind, sim_inputs.J_stim.voxel_ind, sim_inputs.J_stim.n_voxel * sizeof(int), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&J_stim_count, sim_inputs.J_stim.n_voxel * sizeof(int));
    cudaMemcpy(J_stim_count, sim_inputs.J_stim.count, sim_inputs.J_stim.n_voxel * sizeof(int), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&J_stim_step, sim_inputs.J_stim.count[sim_inputs.J_stim.n_voxel - 1] * sizeof(int));
    cudaMemcpy(J_stim_step, sim_inputs.J_stim.step, sim_inputs.J_stim.count[sim_inputs.J_stim.n_voxel - 1] * sizeof(int), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&J_stim_value, N * sizeof(double));
    cudaMemcpy(J_stim_value, sim_inputs.J_stim.value, sim_inputs.J_stim.count[sim_inputs.J_stim.n_voxel - 1] * sizeof(double), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&J_stim_ind_min_max, 2 * sizeof(int));

    J_stim_ind_min_max_h[0] = INT_MAX;
    J_stim_ind_min_max_h[1] = 0;
    for (int ind = 0; ind < sim_inputs.J_stim.count[sim_inputs.J_stim.n_voxel - 1]; ind++) {
        if (J_stim_ind_min_max_h[0] > sim_inputs.J_stim.step[ind]) J_stim_ind_min_max_h[0] = sim_inputs.J_stim.step[ind];
        if (J_stim_ind_min_max_h[1] < sim_inputs.J_stim.step[ind]) J_stim_ind_min_max_h[1] = sim_inputs.J_stim.step[ind];
    }
    cudaMemcpy(J_stim_ind_min_max, J_stim_ind_min_max_h, 2 * sizeof(int), cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();
}

__global__ void simulation_kernel(int step, double* sim_v1, double* dt_delta,
    double* sim_v2,
    double* sim_h,
    int* J_stim_step,
    int* J_stim_n_voxel,
    int* J_stim_voxel_ind,
    double* J_stim_value,
    int* J_stim_count,
    int* J_stim_ind_min_max,
    int* n_voxel,
    int* indices,
    double* coefficients) {
    int index = threadIdx.x + (blockIdx.x * blockDim.x);
    //indices += (blockIdx.x * blockSize) * 18;
    //coefficients += (blockIdx.x * blockSize) * 20;
    indices += (blockIdx.x * blockDim.x) * 18;
    coefficients += (blockIdx.x * blockDim.x) * 20;

    if (index > n_voxel[0] - 1) {
        return;
    }

    double sim_v1_index = sim_v1[index];          
    //double diffusion_term = 1.0 / (4.0 * dt_delta[1]) *
    //    ((sim_v1[indices[threadIdx.x + blockSize * 0]] - sim_v1_index) * coefficients[threadIdx.x + blockSize * 0] + (sim_v1[indices[threadIdx.x + blockSize * 1]] - sim_v1_index) * coefficients[threadIdx.x + blockSize * 1] +
    //        (sim_v1[indices[threadIdx.x + blockSize * 2]] - sim_v1_index) * coefficients[threadIdx.x + blockSize * 2] + (sim_v1[indices[threadIdx.x + blockSize * 3]] - sim_v1_index) * coefficients[threadIdx.x + blockSize * 3] +
    //        (sim_v1[indices[threadIdx.x + blockSize * 4]] - sim_v1_index) * coefficients[threadIdx.x + blockSize * 4] + (sim_v1[indices[threadIdx.x + blockSize * 5]] - sim_v1_index) * coefficients[threadIdx.x + blockSize * 5] +
    //        (sim_v1[indices[threadIdx.x + blockSize * 0]] - sim_v1[indices[threadIdx.x + blockSize * 1]]) * coefficients[threadIdx.x + blockSize * 6] +
    //        (sim_v1[indices[threadIdx.x + blockSize * 2]] - sim_v1[indices[threadIdx.x + blockSize * 3]]) * coefficients[threadIdx.x + blockSize * 7] +
    //        (sim_v1[indices[threadIdx.x + blockSize * 4]] - sim_v1[indices[threadIdx.x + blockSize * 5]]) * coefficients[threadIdx.x + blockSize * 8] +
    //        (sim_v1[indices[threadIdx.x + blockSize * 6]] - sim_v1[indices[threadIdx.x + blockSize * 8]]) * coefficients[threadIdx.x + blockSize * 9] + (sim_v1[indices[threadIdx.x + blockSize * 9]] - sim_v1[indices[threadIdx.x + blockSize * 7]]) * coefficients[threadIdx.x + blockSize * 10] +
    //        (sim_v1[indices[threadIdx.x + blockSize * 14]] - sim_v1[indices[threadIdx.x + blockSize * 16]]) * coefficients[threadIdx.x + blockSize * 10] + (sim_v1[indices[threadIdx.x + blockSize * 17]] - sim_v1[indices[threadIdx.x + blockSize * 15]]) * coefficients[threadIdx.x + blockSize * 12] +
    //        (sim_v1[indices[threadIdx.x + blockSize * 10]] - sim_v1[indices[threadIdx.x + blockSize * 12]]) * coefficients[threadIdx.x + blockSize * 13] + (sim_v1[indices[threadIdx.x + blockSize * 13]] - sim_v1[indices[threadIdx.x + blockSize * 11]]) * coefficients[threadIdx.x + blockSize * 14]);

    double diffusion_term = 1.0 / (4.0 * dt_delta[1]) *
        ((sim_v1[indices[threadIdx.x + blockDim.x * 0]] - sim_v1_index) * coefficients[threadIdx.x + blockDim.x * 0] + (sim_v1[indices[threadIdx.x + blockDim.x * 1]] - sim_v1_index) * coefficients[threadIdx.x + blockDim.x * 1] +
            (sim_v1[indices[threadIdx.x + blockDim.x * 2]] - sim_v1_index) * coefficients[threadIdx.x + blockDim.x * 2] + (sim_v1[indices[threadIdx.x + blockDim.x * 3]] - sim_v1_index) * coefficients[threadIdx.x + blockDim.x * 3] +
            (sim_v1[indices[threadIdx.x + blockDim.x * 4]] - sim_v1_index) * coefficients[threadIdx.x + blockDim.x * 4] + (sim_v1[indices[threadIdx.x + blockDim.x * 5]] - sim_v1_index) * coefficients[threadIdx.x + blockDim.x * 5] +
            (sim_v1[indices[threadIdx.x + blockDim.x * 0]] - sim_v1[indices[threadIdx.x + blockDim.x * 1]]) * coefficients[threadIdx.x + blockDim.x * 6] +
            (sim_v1[indices[threadIdx.x + blockDim.x * 2]] - sim_v1[indices[threadIdx.x + blockDim.x * 3]]) * coefficients[threadIdx.x + blockDim.x * 7] +
            (sim_v1[indices[threadIdx.x + blockDim.x * 4]] - sim_v1[indices[threadIdx.x + blockDim.x * 5]]) * coefficients[threadIdx.x + blockDim.x * 8] +
            (sim_v1[indices[threadIdx.x + blockDim.x * 6]] - sim_v1[indices[threadIdx.x + blockDim.x * 8]]) * coefficients[threadIdx.x + blockDim.x * 9] + (sim_v1[indices[threadIdx.x + blockDim.x * 9]] - sim_v1[indices[threadIdx.x + blockDim.x * 7]]) * coefficients[threadIdx.x + blockDim.x * 10] +
            (sim_v1[indices[threadIdx.x + blockDim.x * 14]] - sim_v1[indices[threadIdx.x + blockDim.x * 16]]) * coefficients[threadIdx.x + blockDim.x * 10] + (sim_v1[indices[threadIdx.x + blockDim.x * 17]] - sim_v1[indices[threadIdx.x + blockDim.x * 15]]) * coefficients[threadIdx.x + blockDim.x * 12] +
            (sim_v1[indices[threadIdx.x + blockDim.x * 10]] - sim_v1[indices[threadIdx.x + blockDim.x * 12]]) * coefficients[threadIdx.x + blockDim.x * 13] + (sim_v1[indices[threadIdx.x + blockDim.x * 13]] - sim_v1[indices[threadIdx.x + blockDim.x * 11]]) * coefficients[threadIdx.x + blockDim.x * 14]);


    double J_stim = 0.0;
    [&] {
        if (step >= J_stim_ind_min_max[0] && step <= J_stim_ind_min_max[1]) {
            for (int count_ind = 0; count_ind < J_stim_n_voxel[0]; count_ind++) {
                if (index == J_stim_voxel_ind[count_ind]) {
                    for (int step_ind = J_stim_count[count_ind]; step_ind < J_stim_count[count_ind + 1]; step_ind++) {
                        if (J_stim_step[step_ind] - 1 == step) {
                            J_stim = J_stim_value[step_ind];
                            return;
                        }
                    }
                }
            }
        }
    }();

    double sim_h_index = sim_h[index];
    sim_v2[index] = ((sim_h_index * sim_v1_index * sim_v1_index * coefficients[threadIdx.x + blockDim.x * 17] * (1 - sim_v1_index)) + (-sim_v1_index * coefficients[threadIdx.x + blockDim.x * 18]) + J_stim + diffusion_term) * dt_delta[0] + sim_v1_index;
    if (sim_v1_index < coefficients[threadIdx.x + blockDim.x * 19]) {
        sim_h[index] = ((1 - sim_h_index) * coefficients[threadIdx.x + blockDim.x * 15]) * dt_delta[0] + sim_h_index;
    }
    else {
        sim_h[index] = (-sim_h_index * coefficients[threadIdx.x + blockDim.x * 16]) * dt_delta[0] + sim_h_index;
    }

    //double sim_h_index = sim_h[index];
    //sim_v2[index] = ((sim_h_index * sim_v1_index * sim_v1_index * coefficients[threadIdx.x + blockSize * 17] * (1 - sim_v1_index)) + (-sim_v1_index * coefficients[threadIdx.x + blockSize * 18]) + J_stim + diffusion_term) * dt_delta[0] + sim_v1_index;
    //if (sim_v1_index < coefficients[threadIdx.x + blockSize * 19]) {
    //    sim_h[index] = ((1 - sim_h_index) * coefficients[threadIdx.x + blockSize * 15]) * dt_delta[0] + sim_h_index;
    //}
    //else {
    //    sim_h[index] = (-sim_h_index * coefficients[threadIdx.x + blockSize * 16]) * dt_delta[0] + sim_h_index;
    //}

    return;
}

simulation_outputs Cardiac::runSimulation(simulation_inputs sim_inputs) {
    dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);
    double* J_stim = new double[numObjects];
    int temp_ptr;

    simulation_outputs sim_output;
    int total_step = sim_inputs.final_t / sim_inputs.dt;
    sim_output.n_step = total_step;
    sim_output.n_voxel = sim_inputs.n_voxel;
    sim_output.data_min = sim_inputs.v_gate[0];
    sim_output.data_max = 1.0;
    int total_save = total_step / sim_inputs.num_of_dt_per_save;
    sim_output.action_potentials = new double* [total_save];
    int save_ind = 0;

    for (int ind = 0; ind < total_step; ind++) {
        if (ind % 1000 == 0) {
            printf("%d\n", ind);
        }

        simulation_kernel << < fullBlocksPerGrid, blockSize >> > (ind, sim_v1,
            dt_delta,
            sim_v2,
            sim_h,
            J_stim_step,
            J_stim_n_voxel,
            J_stim_voxel_ind,
            J_stim_value,
            J_stim_count,
            J_stim_ind_min_max,
            n_voxel,
            indices,
            coefficients);
        cudaDeviceSynchronize();
        std::swap(sim_v1, sim_v2);
        if (ind % sim_inputs.num_of_dt_per_save == 0 || ind == total_step - 1) {
            sim_output.action_potentials[save_ind] = new double[numObjects];
            cudaMemcpy(sim_output.action_potentials[save_ind], sim_v1, numObjects * sizeof(double), cudaMemcpyDeviceToHost);
            save_ind++;
        }
              
        
    }
    return sim_output;
}

void Cardiac::endSimulation() {
    cudaFree(sim_v1);
    cudaFree(dt_delta);
    cudaFree(sim_v2);
    cudaFree(sim_h);
    cudaFree(indices);
    cudaFree(coefficients);
    cudaFree(J_stim_step);
    cudaFree(J_stim_n_voxel);
    cudaFree(J_stim_voxel_ind);
    cudaFree(J_stim_value);
    cudaFree(J_stim_count);
    cudaFree(J_stim_ind_min_max);
    cudaDeviceSynchronize();
}

/******************
* Simulation * NAIVE *****************************************************************************
******************/

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
double* tau_open;
double* tau_close;
double* tau_in;
double* tau_out;
double* v_gate;
int* px;
int* mx;
int* py;
int* my;
int* pz;
int* mz;
int* pxpy;
int* mxpy;
int* pxmy;
int* mxmy;
int* pypz;
int* mypz;
int* pymz;
int* mymz;
int* pxpz;
int* mxpz;
int* pxmz;
int* mxmz;

void Cardiac::initSimulation_naive(simulation_inputs sim_inputs, simulation_data_parts data_parts) {
    int N = sim_inputs.n_voxel;
    numObjects = sim_inputs.n_voxel;
    dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);

    cudaMalloc((void**)&dt_delta, 2 * sizeof(double));
    cudaMemcpy(dt_delta, &(sim_inputs.dt), sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dt_delta + 1, &(sim_inputs.delta_sqr), sizeof(double), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&n_voxel, sizeof(int));
    cudaMemcpy(n_voxel, &(sim_inputs.n_voxel), sizeof(int), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&sim_v1, (numObjects + 1) * sizeof(double));
    cudaMemset((void*)sim_v1, 0, (numObjects + 1) * sizeof(double));
    cudaMalloc((void**)&sim_v2, (numObjects + 1) * sizeof(double));
    cudaMemset((void*)sim_v2, 0, (numObjects + 1) * sizeof(double));
    cudaMalloc((void**)&sim_h, N * sizeof(double));
    double* sim_h_host = new double[numObjects];
    for (int ind = 0; ind < N; ind++) {
        sim_h_host[ind] = 1.0;
    }
    cudaMemcpy(sim_h, sim_h_host, N * sizeof(double), cudaMemcpyHostToDevice);

    cudaMalloc((void**)&part1, N * sizeof(double));
    cudaMemcpy(part1, data_parts.part1, N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&part2, N * sizeof(double));
    cudaMemcpy(part2, data_parts.part2, N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&part3, N * sizeof(double));
    cudaMemcpy(part3, data_parts.part3, N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&part4, N * sizeof(double));
    cudaMemcpy(part4, data_parts.part4, N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&part5, N * sizeof(double));
    cudaMemcpy(part5, data_parts.part5, N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&part6, N * sizeof(double));
    cudaMemcpy(part6, data_parts.part6, N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&part7, N * sizeof(double));
    cudaMemcpy(part7, data_parts.part7, N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&part8, N * sizeof(double));
    cudaMemcpy(part8, data_parts.part8, N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&part9, N * sizeof(double));
    cudaMemcpy(part9, data_parts.part9, N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&part10, N * sizeof(double));
    cudaMemcpy(part10, data_parts.part10, N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&part11, N * sizeof(double));
    cudaMemcpy(part11, data_parts.part11, N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&part12, N * sizeof(double));
    cudaMemcpy(part12, data_parts.part12, N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&part13, N * sizeof(double));
    cudaMemcpy(part13, data_parts.part13, N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&part14, N * sizeof(double));
    cudaMemcpy(part14, data_parts.part14, N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&part15, N * sizeof(double));
    cudaMemcpy(part15, data_parts.part15, N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&px, N * sizeof(int));
    cudaMemcpy(px, sim_inputs.voxel_neighborhood[0], N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&mx, N * sizeof(int));
    cudaMemcpy(mx, sim_inputs.voxel_neighborhood[1], N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&py, N * sizeof(int));
    cudaMemcpy(py, sim_inputs.voxel_neighborhood[2], N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&my, N * sizeof(int));
    cudaMemcpy(my, sim_inputs.voxel_neighborhood[3], N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&pz, N * sizeof(int));
    cudaMemcpy(pz, sim_inputs.voxel_neighborhood[4], N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&mz, N * sizeof(int));
    cudaMemcpy(mz, sim_inputs.voxel_neighborhood[5], N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&pxpy, N * sizeof(int));
    cudaMemcpy(pxpy, sim_inputs.voxel_neighborhood[6], N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&mxpy, N * sizeof(int));
    cudaMemcpy(mxpy, sim_inputs.voxel_neighborhood[7], N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&pxmy, N * sizeof(int));
    cudaMemcpy(pxmy, sim_inputs.voxel_neighborhood[8], N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&mxmy, N * sizeof(int));
    cudaMemcpy(mxmy, sim_inputs.voxel_neighborhood[9], N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&pypz, N * sizeof(int));
    cudaMemcpy(pypz, sim_inputs.voxel_neighborhood[10], N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&mypz, N * sizeof(int));
    cudaMemcpy(mypz, sim_inputs.voxel_neighborhood[11], N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&pymz, N * sizeof(int));
    cudaMemcpy(pymz, sim_inputs.voxel_neighborhood[12], N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&mymz, N * sizeof(int));
    cudaMemcpy(mymz, sim_inputs.voxel_neighborhood[13], N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&pxpz, N * sizeof(int));
    cudaMemcpy(pxpz, sim_inputs.voxel_neighborhood[14], N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&mxpz, N * sizeof(int));
    cudaMemcpy(mxpz, sim_inputs.voxel_neighborhood[15], N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&pxmz, N * sizeof(int));
    cudaMemcpy(pxmz, sim_inputs.voxel_neighborhood[16], N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&mxmz, N * sizeof(int));
    cudaMemcpy(mxmz, sim_inputs.voxel_neighborhood[17], N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&tau_open, N * sizeof(double));
    cudaMemcpy(tau_open, sim_inputs.tau_open, N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&tau_close, N * sizeof(double));
    cudaMemcpy(tau_close, sim_inputs.tau_close, N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&tau_in, N * sizeof(double));
    cudaMemcpy(tau_in, sim_inputs.tau_in, N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&tau_out, N * sizeof(double));
    cudaMemcpy(tau_out, sim_inputs.tau_out, N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&v_gate, N * sizeof(double));
    cudaMemcpy(v_gate, sim_inputs.v_gate, N * sizeof(double), cudaMemcpyHostToDevice);

    cudaMalloc((void**)&J_stim_n_voxel, sizeof(int));
    cudaMemcpy(J_stim_n_voxel, &(sim_inputs.J_stim.n_voxel), sizeof(int), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&J_stim_voxel_ind, sim_inputs.J_stim.n_voxel * sizeof(int));
    cudaMemcpy(J_stim_voxel_ind, sim_inputs.J_stim.voxel_ind, sim_inputs.J_stim.n_voxel * sizeof(int), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&J_stim_count, sim_inputs.J_stim.n_voxel * sizeof(int));
    cudaMemcpy(J_stim_count, sim_inputs.J_stim.count, sim_inputs.J_stim.n_voxel * sizeof(int), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&J_stim_step, sim_inputs.J_stim.count[sim_inputs.J_stim.n_voxel-1] * sizeof(int));
    cudaMemcpy(J_stim_step, sim_inputs.J_stim.step, sim_inputs.J_stim.count[sim_inputs.J_stim.n_voxel - 1] * sizeof(int), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&J_stim_value, N * sizeof(double));
    cudaMemcpy(J_stim_value, sim_inputs.J_stim.value, sim_inputs.J_stim.count[sim_inputs.J_stim.n_voxel - 1] * sizeof(double), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&J_stim_ind_min_max, 2 * sizeof(int));

    cudaMalloc((void**)&dev_pos, N * sizeof(glm::vec3));
    checkCUDAErrorWithLine("cudaMalloc dev_pos failed!");
    cudaMemcpy(dev_pos, sim_inputs.voxel, N * sizeof(glm::vec3), cudaMemcpyHostToDevice);

    cudaMalloc((void**)&dev_ap_gl, N * sizeof(glm::vec3));
    checkCUDAErrorWithLine("cudaMalloc dev_ap_gl failed!");
    
    J_stim_ind_min_max_h[0] = INT_MAX;
    J_stim_ind_min_max_h[1] = 0;
    for (int ind = 0; ind < sim_inputs.J_stim.count[sim_inputs.J_stim.n_voxel - 1]; ind++) {
        if (J_stim_ind_min_max_h[0] > sim_inputs.J_stim.step[ind]) J_stim_ind_min_max_h[0] = sim_inputs.J_stim.step[ind];
        if (J_stim_ind_min_max_h[1] < sim_inputs.J_stim.step[ind]) J_stim_ind_min_max_h[1] = sim_inputs.J_stim.step[ind];
    }
    cudaMemcpy(J_stim_ind_min_max, J_stim_ind_min_max_h, 2 * sizeof(int), cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();
}

__global__ void simulation_kernel_naive(int step, double* sim_v1, double* dt_delta,
    double* sim_v2,
    double* sim_h,
    double* part1,
    double* part2,
    double* part3,
    double* part4,
    double* part5,
    double* part6,
    double* part7,
    double* part8,
    double* part9,
    double* part10,
    double* part11,
    double* part12,
    double* part13,
    double* part14,
    double* part15,
    double* tau_open,
    double* tau_close,
    double* tau_in,
    double* tau_out,
    double* v_gate,
    int* px,
    int* mx,
    int* py,
    int* my,
    int* pz,
    int* mz,
    int* pxpy,
    int* mxpy,
    int* pxmy,
    int* mxmy,
    int* pypz,
    int* mypz,
    int* pymz,
    int* mymz,
    int* pxpz,
    int* mxpz,
    int* pxmz,
    int* mxmz,
    int* J_stim_step,
    int* J_stim_n_voxel,
    int* J_stim_voxel_ind,
    double* J_stim_value,
    int* J_stim_count,
    int* J_stim_ind_min_max,
    int* n_voxel) {
    int index = threadIdx.x + (blockIdx.x * blockDim.x);
    if (index == n_voxel[0]-1) {
        return;
    }
    double sim_v1_index = sim_v1[index];
    double diffusion_term = 1.0 / (4.0 * dt_delta[1]) *
        ((sim_v1[px[index]] - sim_v1_index) * part1[index] + (sim_v1[mx[index]] - sim_v1_index) * part2[index] +
            (sim_v1[py[index]] - sim_v1_index) * part3[index] + (sim_v1[my[index]] - sim_v1_index) * part4[index] +
            (sim_v1[pz[index]] - sim_v1_index) * part5[index] + (sim_v1[mz[index]] - sim_v1_index) * part6[index] +
            (sim_v1[px[index]] - sim_v1[mx[index]]) * part7[index] +
            (sim_v1[py[index]] - sim_v1[my[index]]) * part8[index] +
            (sim_v1[pz[index]] - sim_v1[mz[index]]) * part9[index] +
            (sim_v1[pxpy[index]] - sim_v1[pxmy[index]]) * part10[index] + (sim_v1[mxmy[index]] - sim_v1[mxpy[index]]) * part11[index] +
            (sim_v1[pxpz[index]] - sim_v1[pxmz[index]]) * part12[index] + (sim_v1[mxmz[index]] - sim_v1[mxpz[index]]) * part13[index] +
            (sim_v1[pypz[index]] - sim_v1[pymz[index]]) * part14[index] + (sim_v1[mymz[index]] - sim_v1[mypz[index]]) * part15[index]);
    
    double J_stim = 0;
    [&] {
    if (step >= J_stim_ind_min_max[0] && step <= J_stim_ind_min_max[1]) {
        for (int count_ind = 0; count_ind < J_stim_n_voxel[0]; count_ind++) {
            if (index == J_stim_voxel_ind[count_ind]) {
                for (int step_ind = J_stim_count[count_ind]; step_ind < J_stim_count[count_ind + 1]; step_ind++) {
                    if (J_stim_step[step_ind] - 1 == step) {
                        J_stim = J_stim_value[step_ind];
                        return;
                    }
                }
            }
        }
    }
    }();

    double sim_h_index = sim_h[index];
    sim_v2[index] = ((sim_h_index * sim_v1_index * sim_v1_index * (1 - sim_v1_index) / tau_in[index]) + (-sim_v1_index / tau_out[index]) + J_stim + diffusion_term) * dt_delta[0] + sim_v1_index;
    if (sim_v1_index < v_gate[index]) {
        sim_h[index] = ((1 - sim_h_index) / tau_open[index]) * dt_delta[0] + sim_h_index;
    }
    else {
        sim_h[index] = (-sim_h_index / tau_close[index]) * dt_delta[0] + sim_h_index;
    }
    return;
}

simulation_outputs Cardiac::runSimulation_naive(simulation_inputs sim_inputs) {
    dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);
    double* J_stim = new double[numObjects];
    int temp_ptr;

    simulation_outputs sim_output;
    int total_step = sim_inputs.final_t / sim_inputs.dt;
    sim_output.n_step = total_step;
    sim_output.n_voxel = sim_inputs.n_voxel;
    sim_output.data_min = sim_inputs.v_gate[0];
    sim_output.data_max = 1.0;
    int total_save = total_step / sim_inputs.num_of_dt_per_save;
    sim_output.action_potentials = new double* [total_save];
    int save_ind = 0;

    for (int ind = 0; ind < total_step; ind++) {
        if (ind % 1000 == 0) {
            printf("%d\n", ind);
        }

        simulation_kernel_naive << < fullBlocksPerGrid, blockSize >> > (ind, sim_v1,
            dt_delta,
            sim_v2,
            sim_h,
            part1,
            part2,
            part3,
            part4,
            part5,
            part6,
            part7,
            part8,
            part9,
            part10,
            part11,
            part12,
            part13,
            part14,
            part15,
            tau_open,
            tau_close,
            tau_in,
            tau_out,
            v_gate,
            px,
            mx,
            py,
            my,
            pz,
            mz,
            pxpy,
            mxpy,
            pxmy,
            mxmy,
            pypz,
            mypz,
            pymz,
            mymz,
            pxpz,
            mxpz,
            pxmz,
            mxmz,
            J_stim_step,
            J_stim_n_voxel,
            J_stim_voxel_ind,
            J_stim_value,
            J_stim_count,
            J_stim_ind_min_max,
            n_voxel);

        cudaDeviceSynchronize();
        checkCUDAErrorWithLine("runSimulation failed!");
        if (ind % sim_inputs.num_of_dt_per_save == 0 || ind == total_step - 1) {
            sim_output.action_potentials[save_ind] = new double[numObjects];
            cudaMemcpy(sim_output.action_potentials[save_ind], sim_v1, numObjects * sizeof(double), cudaMemcpyDeviceToHost);
            save_ind++;
        }
        std::swap(sim_v1, sim_v2);
    }
    return sim_output;
}

void Cardiac::endSimulation_naive() {
    cudaFree(sim_v1);
    cudaFree(dt_delta);
    cudaFree(sim_v2);
    cudaFree(sim_h);
    cudaFree(part1);
    cudaFree(part2);
    cudaFree(part3);
    cudaFree(part4);
    cudaFree(part5);
    cudaFree(part6);
    cudaFree(part7);
    cudaFree(part8);
    cudaFree(part9);
    cudaFree(part10);
    cudaFree(part11);
    cudaFree(part12);
    cudaFree(part13);
    cudaFree(part14);
    cudaFree(part15);
    cudaFree(tau_open);
    cudaFree(tau_close);
    cudaFree(tau_in);
    cudaFree(tau_out);
    cudaFree(v_gate);
    cudaFree(px);
    cudaFree(mx);
    cudaFree(py);
    cudaFree(my);
    cudaFree(pz);
    cudaFree(mz);
    cudaFree(pxpy);
    cudaFree(mxpy);
    cudaFree(pxmy);
    cudaFree(mxmy);
    cudaFree(pypz);
    cudaFree(mypz);
    cudaFree(pymz);
    cudaFree(mymz);
    cudaFree(pxpz);
    cudaFree(mxpz);
    cudaFree(pxmz);
    cudaFree(mxmz);
    cudaFree(J_stim_step);
    cudaFree(J_stim_n_voxel);
    cudaFree(J_stim_voxel_ind);
    cudaFree(J_stim_value);
    cudaFree(J_stim_count);
    cudaFree(J_stim_ind_min_max);
    cudaDeviceSynchronize();
}


/******************
* Visulization *
******************/

void Cardiac::initVisulization(int N, glm::vec3* voxel) {
  numObjects = N;
  dim3 fullBlocksPerGrid((N + blockSize - 1) / blockSize);

  cudaMalloc((void**)&dev_pos, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_pos failed!");
  cudaMemcpy(dev_pos, voxel, N * sizeof(glm::vec3), cudaMemcpyHostToDevice);

  cudaMalloc((void**)&dev_ap_gl, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_ap_gl failed!");

  cudaDeviceSynchronize();
}

__global__ void kernCopyPositionsToVBO(int N, glm::vec3 *pos, float *vbo, float s_scale) {
  int index = threadIdx.x + (blockIdx.x * blockDim.x);
  float c_scale = -1.0f / s_scale;
  if (index < N) {
    vbo[4 * index + 0] = pos[index].x * c_scale;
    vbo[4 * index + 1] = pos[index].y * c_scale;
    vbo[4 * index + 2] = pos[index].z * c_scale;
    vbo[4 * index + 3] = 1.0f;
  }
}

__global__ void kernCopyPotentialToVBO(int N, glm::vec3 *vel, float *vbo, float s_scale) {
  int index = threadIdx.x + (blockIdx.x * blockDim.x);

  if (index < N) {
    vbo[4 * index + 0] = vel[index].x + 0.3f;
    vbo[4 * index + 1] = vel[index].y + 0.3f;
    vbo[4 * index + 2] = vel[index].z + 0.3f;
    vbo[4 * index + 3] = 1.0f;
  }
}

/**
* Wrapper for call to the kernCopyCardiacToVBO CUDA kernel.
*/
void Cardiac::copyCardiacToVBO(float *vbodptr_positions, float *vbodptr_potential) {
  dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);

  kernCopyPositionsToVBO << <fullBlocksPerGrid, blockSize >> >(numObjects, dev_pos, vbodptr_positions, scene_scale);
  checkCUDAErrorWithLine("copyCardiacToVBO failed!");
  kernCopyPotentialToVBO << <fullBlocksPerGrid, blockSize >> >(numObjects, dev_ap_gl, vbodptr_potential, scene_scale);

  cudaDeviceSynchronize();
}

void Cardiac::copySimOutput(int N, glm::vec3* output_vec3) {
    cudaMemcpy(dev_ap_gl, output_vec3, N * sizeof(glm::vec3), cudaMemcpyHostToDevice);
}

void Cardiac::endVisulization() {
    cudaFree(dev_ap_gl);
    cudaFree(dev_pos);
}
