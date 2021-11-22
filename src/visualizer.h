#pragma once

#include <iostream>
#include <cstdlib>
#include <string>
#include <sstream>
#include <fstream>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "utilityCore.hpp"
#include "glslUtility.hpp"
#include "json/json.h"
#include "data_handler.h"


void visualizationLoop(simulation_outputs sim_inputs);
bool initGLFW(simulation_inputs sim_inputs);
bool initCUDA();
void updateCamera();
void errorCallback(int error, const char* description);
void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);
void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods);
void mousePositionCallback(GLFWwindow* window, double xpos, double ypos);
void initShaders(GLuint* program);
void initVAO(int N_FOR_VIS);