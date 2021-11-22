#pragma once

#include <stdio.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/device_vector.h>
#include <iostream>
#include <cuda.h>
#include <cmath>
#include <vector>
#include "data_struct.h"


#ifndef imax
#define imax( a, b ) ( ((a) > (b)) ? (a) : (b) )
#endif

#ifndef imin
#define imin( a, b ) ( ((a) < (b)) ? (a) : (b) )
#endif

namespace Cardiac {
    simulation_outputs runSimulation_naive(simulation_inputs sim_inputs);
    simulation_outputs runSimulation_optimized(simulation_inputs sim_inputs);
    void initSimulation(simulation_inputs sim_inputs, simulation_data_parts data_parts);
    void endSimulation();
    void initVisulization(int N, glm::vec3* voxel);
    void copyCardiacToVBO(float *vbodptr_positions, float *vbodptr_potential);
    void endVisulization();
    void unitTest();
    void copySimOutput(int N, glm::vec3* output_vec3);
}

