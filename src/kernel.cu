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

/*! Block size used for CUDA kernel launch. */
#define blockSize 256

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
double* DT;
int* n_voxel;

int* J_stim_ind_min_max;
int* J_stim_ind_min_max_h = new int[2];
double* J_stim_voxel;
int* J_stim_step;
int* J_stim_n_voxel;
int* J_stim_voxel_ind;
double* J_stim_value;
int* J_stim_count;



/******************
* Simulation *
******************/

void Cardiac::initSimulation(simulation_inputs sim_inputs, simulation_data_parts data_parts) {
    int N = sim_inputs.n_voxel;
    numObjects = sim_inputs.n_voxel;
    dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);

    cudaMalloc((void**)&DT, sizeof(double));
    cudaMemcpy(DT, &(sim_inputs.dt), sizeof(double), cudaMemcpyHostToDevice);
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

__global__ void simulation_kernel_optimized(int step, double* sim_v1, double* dt,
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
    if (index == n_voxel[0]) {
        return;
    }
    double sim_v1_index = sim_v1[index];
    double diffusion_term = 1.0 / (4.0 * 4.0) *
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
                        step_ind = J_stim_count[count_ind + 1];
                        count_ind = J_stim_n_voxel[0];
                        return;
                    }
                }
            }
        }
    }
    }();

    double sim_h_index = sim_h[index];
    sim_v2[index] = ((sim_h_index * sim_v1_index * sim_v1_index * (1 - sim_v1_index) / tau_in[index]) + (-sim_v1_index / tau_out[index]) + J_stim + diffusion_term) * dt[0] + sim_v1_index;
    if (sim_v1_index < v_gate[index]) {
        sim_h[index] = ((1 - sim_h_index) / tau_open[index]) * dt[0] + sim_h_index;
    }
    else {
        sim_h[index] = (-sim_h_index / tau_close[index]) * dt[0] + sim_h_index;
    }
    return;
}

simulation_outputs Cardiac::runSimulation_optimized(simulation_inputs sim_inputs) {
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

        simulation_kernel_optimized << < fullBlocksPerGrid, blockSize >> > (ind, sim_v1,
            DT,
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

void Cardiac::endSimulation() {
    cudaFree(sim_v1);
    cudaFree(DT);
    cudaFree(sim_v2);
    cudaFree(sim_h);
    cudaFree(J_stim_voxel);
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

void Cardiac::unitTest() {
  // LOOK-1.2 Feel free to write additional tests here.

  // test unstable sort
  int *dev_intKeys;
  int *dev_intValues;
  int N = 10;

  std::unique_ptr<int[]>intKeys{ new int[N] };
  std::unique_ptr<int[]>intValues{ new int[N] };

  intKeys[0] = 0; intValues[0] = 0;
  intKeys[1] = 1; intValues[1] = 1;
  intKeys[2] = 0; intValues[2] = 2;
  intKeys[3] = 3; intValues[3] = 3;
  intKeys[4] = 0; intValues[4] = 4;
  intKeys[5] = 2; intValues[5] = 5;
  intKeys[6] = 2; intValues[6] = 6;
  intKeys[7] = 0; intValues[7] = 7;
  intKeys[8] = 5; intValues[8] = 8;
  intKeys[9] = 6; intValues[9] = 9;

  cudaMalloc((void**)&dev_intKeys, N * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_intKeys failed!");

  cudaMalloc((void**)&dev_intValues, N * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_intValues failed!");

  dim3 fullBlocksPerGrid((N + blockSize - 1) / blockSize);

  std::cout << "before unstable sort: " << std::endl;
  for (int i = 0; i < N; i++) {
    std::cout << "  key: " << intKeys[i];
    std::cout << " value: " << intValues[i] << std::endl;
  }

  // How to copy data to the GPU
  cudaMemcpy(dev_intKeys, intKeys.get(), sizeof(int) * N, cudaMemcpyHostToDevice);
  cudaMemcpy(dev_intValues, intValues.get(), sizeof(int) * N, cudaMemcpyHostToDevice);

  // Wrap device vectors in thrust iterators for use with thrust.
  thrust::device_ptr<int> dev_thrust_keys(dev_intKeys);
  thrust::device_ptr<int> dev_thrust_values(dev_intValues);
  // LOOK-2.1 Example for using thrust::sort_by_key
  thrust::sort_by_key(dev_thrust_keys, dev_thrust_keys + N, dev_thrust_values);

  // How to copy data back to the CPU side from the GPU
  cudaMemcpy(intKeys.get(), dev_intKeys, sizeof(int) * N, cudaMemcpyDeviceToHost);
  cudaMemcpy(intValues.get(), dev_intValues, sizeof(int) * N, cudaMemcpyDeviceToHost);
  checkCUDAErrorWithLine("memcpy back failed!");

  std::cout << "after unstable sort: " << std::endl;
  for (int i = 0; i < N; i++) {
    std::cout << "  key: " << intKeys[i];
    std::cout << " value: " << intValues[i] << std::endl;
  }

  // cleanup
  cudaFree(dev_intKeys);
  cudaFree(dev_intValues);
  checkCUDAErrorWithLine("cudaFree failed!");
  return;
}
