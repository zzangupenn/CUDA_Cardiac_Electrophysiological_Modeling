#pragma once
#include <iostream>
#include <cstdlib>
#include <string>
#include <sstream>
#include <fstream>
#include <chrono>
#include <Eigen/Dense>
#include "data_struct.h"

using namespace std;
using namespace Eigen;

ArrayXd sign_int(ArrayXi v);
simulation_outputs Simulation_CPU(simulation_inputs sim_inputs);
simulation_data_parts prepare_parts(simulation_inputs sim_inputs);
