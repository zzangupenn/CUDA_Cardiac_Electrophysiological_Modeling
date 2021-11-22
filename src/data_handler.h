#pragma once

#include "json/json.h"
#include <sstream>
#include <fstream>
#include <iostream>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <Eigen/Dense>
#include "data_struct.h"

simulation_inputs import_para(string setting_path, string input_path);
int save_data(simulation_inputs sim_input, simulation_outputs sim_output, double save_data_min_clip);