#pragma once

#include "json/json.h"
#include <sstream>
#include <fstream>
#include <iostream>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <Eigen/Dense>
#include "data_struct.h"

simulation_inputs import_para(string setting_path, string vol_path, string j_stim_path);