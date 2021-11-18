/**
* @file      main.cpp
* @brief     Example Cardiac flocking simulation for CIS 565
* @authors   Liam Boone, Kai Ninomiya, Kangning (Gary) Li
* @date      2013-2017
* @copyright University of Pennsylvania
*/
#include "visualizer.h"
#include "kernel.h"
#include "data_struct.h"
using namespace std;

const char* projectName;
std::string deviceName;
#define VISUALIZE 1

//====================================
// GL Stuff
//====================================

GLuint positionLocation = 0;   // Match results from glslUtility::createProgram.
GLuint potentialLocation = 1; // Also see attribtueLocations below.
const char* attributeLocations[] = { "Position", "Potential" };

GLuint cardiacVAO = 0;
GLuint cardiacVBO_positions = 0;
GLuint cardiacVBO_potential = 0;
GLuint cardiacIBO = 0;
GLuint displayImage;
GLuint program[2];

const int N_FOR_VIS = 13288;
const float DT = 0.2f;

const unsigned int PROG_CARDIAC = 0;

const float fovy = (float)(PI / 4);
const float zNear = 0.10f;
const float zFar = 100.0f;
int width = 1280;
int height = 720;
int pointSize = 2;

// For camera controls
bool leftMousePressed = false;
bool rightMousePressed = false;
double lastX;
double lastY;
float theta = 1.040833f;
float phi = -0.842187f;
float zoom = 6.0f;
glm::vec3 lookAt = glm::vec3(0.0f, -3.0f, -5.0f);
glm::vec3 cameraPosition = glm::vec3(0.0f, 0.0f, 0.0f);
glm::mat4 projection;
GLFWwindow* window;

//void print_str(std::string str) {
//    printf("%s\n", str.c_str());
//}

//void print_float(float x) {
//    printf("%f\n", x);
//}


//====================================
// Main loop
//====================================
void runCUDA(int N, glm::vec3* output_vec3) {
    // Map OpenGL buffer object for writing from CUDA on a single GPU
    // No data is moved (Win & Linux). When mapped to CUDA, OpenGL should not
    // use this buffer

    float4* dptr = NULL;
    float* dptrVertPositions = NULL;
    float* dptrVertPotential = NULL;

    cudaGLMapBufferObject((void**)&dptrVertPositions, cardiacVBO_positions);
    cudaGLMapBufferObject((void**)&dptrVertPotential, cardiacVBO_potential);

    // execute the kernel
    Cardiac::copySimOutput(N, output_vec3);
    //for (int ind = 0; ind < N; ind++) {
    //    printf("%f\n", float(output_vec3[ind].x));
    //}

    //#if VISUALIZE
    Cardiac::copyCardiacToVBO(dptrVertPositions, dptrVertPotential);
    //#endif
    // unmap buffer object
    cudaGLUnmapBufferObject(cardiacVBO_positions);
    cudaGLUnmapBufferObject(cardiacVBO_potential);
}

void mainLoop(simulation_outputs sim_output) {
    double fps = 0;
    double timebase = 0;
    int frame = 0;
    int sim_frame = 0;
    int sim_frame_step = 10;
    glm::vec3* output_vec3 = new glm::vec3[sim_output.n_voxel];
    
    while (!glfwWindowShouldClose(window)) {
        
        glfwPollEvents();

        frame++;
        sim_frame += 10;
        double time = glfwGetTime();

        if (time - timebase > 1.0) {
            fps = frame / (time - timebase);
            timebase = time;
            frame = 0;
        }
        double scale = 1.0;
        for (int ind = 0; ind < sim_output.n_voxel; ind++) {
            output_vec3[ind] = glm::vec3(sim_output.action_potentials[sim_frame][ind] * scale, 0.0, 0.0);
            //printf("%f\n", float(sim_output.action_potentials[10000][ind] * scale));
        }
        runCUDA(sim_output.n_voxel, output_vec3);

        std::ostringstream ss;
        ss << "[";
        ss.precision(1);
        ss << std::fixed << fps;
        ss << " fps] " << "frame " << to_string(sim_frame) << " " << deviceName;
        glfwSetWindowTitle(window, ss.str().c_str());

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

#if VISUALIZE
        glUseProgram(program[PROG_CARDIAC]);
        glBindVertexArray(cardiacVAO);
        glPointSize((GLfloat)pointSize);
        glDrawElements(GL_POINTS, N_FOR_VIS + 1, GL_UNSIGNED_INT, 0);
        glPointSize(1.0f);

        glUseProgram(0);
        glBindVertexArray(0);

        glfwSwapBuffers(window);
        if (frame >= sim_output.n_step - sim_frame_step) break;
#endif
    }
    glfwDestroyWindow(window);
    glfwTerminate();
}

/**
* Initialization of CUDA and GLFW.
*/
bool init(simulation_inputs _sim_inputs) {
    // Set window title to "Student Name: [SM 2.0] GPU Name"
    
    projectName = "CUDA_Cardiac_Electrophysiological_Modeling";
    cudaDeviceProp deviceProp;
    int gpuDevice = 0;
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    if (gpuDevice > device_count) {
        std::cout
            << "Error: GPU device number is greater than the number of devices!"
            << " Perhaps a CUDA-capable GPU is not installed?"
            << std::endl;
        return false;
    }
    cudaGetDeviceProperties(&deviceProp, gpuDevice);
    int major = deviceProp.major;
    int minor = deviceProp.minor;
    
    std::ostringstream ss;
    ss << projectName << " [SM " << major << "." << minor << " " << deviceProp.name << "]";
    deviceName = ss.str();

    // Window setup stuff
    glfwSetErrorCallback(errorCallback);

    if (!glfwInit()) {
        std::cout
            << "Error: Could not initialize GLFW!"
            << " Perhaps OpenGL 3.3 isn't available?"
            << std::endl;
        return false;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    window = glfwCreateWindow(width, height, deviceName.c_str(), NULL, NULL);
    if (!window) {
        glfwTerminate();
        return false;
    }
    glfwMakeContextCurrent(window);
    glfwSetKeyCallback(window, keyCallback);
    glfwSetCursorPosCallback(window, mousePositionCallback);
    glfwSetMouseButtonCallback(window, mouseButtonCallback);

    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) {
        return false;
    }

    // Initialize drawing state
    initVAO();

    // Default to device ID 0. If you have more than one GPU and want to test a non-default one,
    // change the device ID.
    cudaGLSetGLDevice(0);
    cudaGLRegisterBufferObject(cardiacVBO_positions);
    cudaGLRegisterBufferObject(cardiacVBO_potential);
    Cardiac::initSimulation(_sim_inputs.n_voxel, _sim_inputs.voxel);

    updateCamera();
    initShaders(program);
    glEnable(GL_DEPTH_TEST);

    return true;
}

void initVAO() {

    std::unique_ptr<GLfloat[]> bodies{ new GLfloat[4 * (N_FOR_VIS)] };
    std::unique_ptr<GLuint[]> bindices{ new GLuint[N_FOR_VIS] };

    //glm::vec4 ul(-1.0, -1.0, 1.0, 1.0);
    //glm::vec4 lr(1.0, 1.0, 0.0, 0.0);

    for (int i = 0; i < N_FOR_VIS; i++) {
        bodies[4 * i + 0] = 0.0f;
        bodies[4 * i + 1] = 0.0f;
        bodies[4 * i + 2] = 0.0f;
        bodies[4 * i + 3] = 1.0f;
        bindices[i] = i;
    }

    glGenVertexArrays(1, &cardiacVAO); // Attach everything needed to draw a particle to this
    glGenBuffers(1, &cardiacVBO_positions);
    glGenBuffers(1, &cardiacVBO_potential);
    glGenBuffers(1, &cardiacIBO);

    glBindVertexArray(cardiacVAO);

    // Bind the positions array to the cardiacVAO by way of the cardiacVBO_positions
    glBindBuffer(GL_ARRAY_BUFFER, cardiacVBO_positions); // bind the buffer
    glBufferData(GL_ARRAY_BUFFER, 4 * (N_FOR_VIS) * sizeof(GLfloat), bodies.get(), GL_DYNAMIC_DRAW); // transfer data

    glEnableVertexAttribArray(positionLocation);
    glVertexAttribPointer((GLuint)positionLocation, 4, GL_FLOAT, GL_FALSE, 0, 0);

    // Bind the potential array to the cardiacVAO by way of the cardiacVBO_potential
    glBindBuffer(GL_ARRAY_BUFFER, cardiacVBO_potential);
    glBufferData(GL_ARRAY_BUFFER, 4 * (N_FOR_VIS) * sizeof(GLfloat), bodies.get(), GL_DYNAMIC_DRAW);
    glEnableVertexAttribArray(potentialLocation);
    glVertexAttribPointer((GLuint)potentialLocation, 4, GL_FLOAT, GL_FALSE, 0, 0);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, cardiacIBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, (N_FOR_VIS) * sizeof(GLuint), bindices.get(), GL_STATIC_DRAW);

    glBindVertexArray(0);
}

void initShaders(GLuint* program) {
    GLint location;

    program[PROG_CARDIAC] = glslUtility::createProgram(
        "shaders/cardiac.vert.glsl",
        "shaders/cardiac.geom.glsl",
        "shaders/cardiac.frag.glsl", attributeLocations, 2);
    glUseProgram(program[PROG_CARDIAC]);

    if ((location = glGetUniformLocation(program[PROG_CARDIAC], "u_projMatrix")) != -1) {
        glUniformMatrix4fv(location, 1, GL_FALSE, &projection[0][0]);
    }
    if ((location = glGetUniformLocation(program[PROG_CARDIAC], "u_cameraPos")) != -1) {
        glUniform3fv(location, 1, &cameraPosition[0]);
    }
}

void errorCallback(int error, const char* description) {
    fprintf(stderr, "error %d: %s\n", error, description);
}

void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
        glfwSetWindowShouldClose(window, GL_TRUE);
    }
}

void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {
    leftMousePressed = (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS);
    rightMousePressed = (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS);
}

void mousePositionCallback(GLFWwindow* window, double xpos, double ypos) {
    if (leftMousePressed) {
        // compute new camera parameters
        phi += (xpos - lastX) / width;
        theta -= (ypos - lastY) / height;
        theta = std::fmax(0.01f, std::fmin(theta, 3.14f));
        updateCamera();
    }
    else if (rightMousePressed) {
        zoom += (ypos - lastY) / height;
        zoom = std::fmax(0.1f, std::fmin(zoom, 5.0f));
        updateCamera();
    }

    lastX = xpos;
    lastY = ypos;
}

void updateCamera() {
    //print(theta);
    //print(phi);
    //print_str(" ");
    cameraPosition.x = zoom * sin(phi) * sin(theta);
    cameraPosition.z = zoom * cos(theta);
    cameraPosition.y = zoom * cos(phi) * sin(theta);
    cameraPosition += lookAt;

    projection = glm::perspective(fovy, float(width) / float(height), zNear, zFar);
    glm::mat4 view = glm::lookAt(cameraPosition, lookAt, glm::vec3(0, 0, 1));
    projection = projection * view;

    GLint location;

    glUseProgram(program[PROG_CARDIAC]);
    if ((location = glGetUniformLocation(program[PROG_CARDIAC], "u_projMatrix")) != -1) {
        glUniformMatrix4fv(location, 1, GL_FALSE, &projection[0][0]);
    }
}