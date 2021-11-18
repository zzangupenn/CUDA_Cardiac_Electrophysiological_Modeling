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


#ifndef imax
#define imax( a, b ) ( ((a) > (b)) ? (a) : (b) )
#endif

#ifndef imin
#define imin( a, b ) ( ((a) < (b)) ? (a) : (b) )
#endif

namespace Cardiac {
    void initSimulation(int N, glm::vec3* voxel);
    void copyCardiacToVBO(float *vbodptr_positions, float *vbodptr_potential);
    void endSimulation();
    void unitTest();
    void copySimOutput(int N, glm::vec3* output_vec3);
}

