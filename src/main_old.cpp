

#include "main.hpp"
#include "visualizer.h"
#include "data_handler.h"
#include "kernel.h"
using namespace std;
using namespace Eigen;



//void print_str(std::string str) {
//    printf("%s\n", str.c_str());
//}

//void print(float x) {
//    printf("%f\n", x);
//}

void print_arr(int* l, int size) {
    for (int ind = 0; ind < size; ind++) {
        std::cout << l[ind] << std::endl;
    }
    std::cout << std::endl;
}

void print_mat(MatrixXi m) {
    std::cout << m << std::endl;
}

VectorXi sign_int(VectorXi v) {
    VectorXi v_ret = (v.array() > 0).select(VectorXi::Ones(v.rows()), VectorXi::Zero(v.rows()));
    return v_ret;
}

/**
* C main function.
*/
int main(int argc, char* argv[]) {

    simulation_inputs sim_inputs = import_para("data/sim_settings.json", 
                                               "data/volume.json", 
                                               "data/J_stim.json");
    VectorXi px = sim_inputs.voxel_neighborhood(all, 0);
    VectorXi mx = sim_inputs.voxel_neighborhood(all, 1);
    VectorXi py = sim_inputs.voxel_neighborhood(all, 2);
    VectorXi my = sim_inputs.voxel_neighborhood(all, 3);
    VectorXi pz = sim_inputs.voxel_neighborhood(all, 4);
    VectorXi mz = sim_inputs.voxel_neighborhood(all, 5);
    VectorXi pxpy = sim_inputs.voxel_neighborhood(all, 6);
    VectorXi mxpy = sim_inputs.voxel_neighborhood(all, 7);
    VectorXi pxmy = sim_inputs.voxel_neighborhood(all, 8);
    VectorXi mxmy = sim_inputs.voxel_neighborhood(all, 9);
    VectorXi pypz = sim_inputs.voxel_neighborhood(all, 10);
    VectorXi mypz = sim_inputs.voxel_neighborhood(all, 11);
    VectorXi pymz = sim_inputs.voxel_neighborhood(all, 12);
    VectorXi mymz = sim_inputs.voxel_neighborhood(all, 13);
    VectorXi pxpz = sim_inputs.voxel_neighborhood(all, 14);
    VectorXi mxpz = sim_inputs.voxel_neighborhood(all, 15);
    VectorXi pxmz = sim_inputs.voxel_neighborhood(all, 16);
    VectorXi mxmz = sim_inputs.voxel_neighborhood(all, 17);

    VectorXf part1 = 4.0 * sim_inputs.D(0, 0) * sign_int(px).cast<float>();
    VectorXf part2 = 4.0 * sim_inputs.D(0, 0) * sign_int(mx).cast<float>();
    VectorXf part3 = 4.0 * sim_inputs.D(1, 1) * sign_int(py).cast<float>();
    VectorXf part4 = 4.0 * sim_inputs.D(1, 1) * sign_int(my).cast<float>();
    VectorXf part5 = 4.0 * sim_inputs.D(2, 2) * sign_int(pz).cast<float>();
    VectorXf part6 = 4.0 * sim_inputs.D(2, 2) * sign_int(mz).cast<float>();
    VectorXf part7 = VectorXf::Zero(part1.rows());
    VectorXf part8 = VectorXf::Zero(part1.rows());
    VectorXf part9 = VectorXf::Zero(part1.rows());
    VectorXf part10 = 2.0 * sim_inputs.D(1, 2) * sign_int(pxpy).cast<float>() * sign_int(pxmy).cast<float>();
    VectorXf part11 = 2.0 * sim_inputs.D(1, 2) * sign_int(mxpy).cast<float>() * sign_int(mxmy).cast<float>();
    VectorXf part12 = 2.0 * sim_inputs.D(1, 3) * sign_int(pxpz).cast<float>() * sign_int(pxmz).cast<float>();
    VectorXf part13 = 2.0 * sim_inputs.D(1, 3) * sign_int(mxpz).cast<float>() * sign_int(mxmz).cast<float>();
    VectorXf part14 = 2.0 * sim_inputs.D(1, 3) * sign_int(pypz).cast<float>() * sign_int(pymz).cast<float>();
    VectorXf part15 = 2.0 * sim_inputs.D(1, 3) * sign_int(mypz).cast<float>() * sign_int(mymz).cast<float>();

    VectorXf sim_v = VectorXf::Zero(part1.rows());
    VectorXf sim_h = VectorXf::Ones(part1.rows());



    if (init(sim_inputs)) {
        mainLoop();
        Cardiac::endSimulation();
        return 0;
    } else {
        return 1;
    }
}


