#include "data_handler.h"
using namespace std;


//void print_str(std::string str) {
//    printf("%s\n", str.c_str());
//}


simulation_inputs import_para(string setting_path, string vol_path,
                              string j_stim_path) {
	simulation_inputs ret;
    ifstream file;
    Json::Value json_in;
    file.open(setting_path);
    file >> json_in;
    ret.dt = json_in["dt"].asDouble();
    ret.final_t = json_in["final_t"].asInt();
    ret.tau_in = json_in["tau_in"].asDouble();
    ret.tau_out = json_in["tau_out"].asDouble();
    ret.tau_open = json_in["tau_open"].asDouble();
    ret.tau_close = json_in["tau_close"].asDouble();
    ret.v_gate = json_in["v_gate"].asDouble();
    ret.c_for_D = json_in["c_for_D"].asDouble();

    for (int ind = 0; ind < 9; ind++) {
        ret.D[ind] = json_in["D"][ind].asDouble();
    }
    file.close();
    file.clear();
    json_in.clear();
    
    // load volume
    file.open(vol_path);
    file >> json_in;
    ret.voxel = new glm::vec3[json_in["voxel"].size()];
    ret.n_voxel = json_in["voxel"].size();
    ret.voxel_neighborhood = new int*[18];
   
    for (int ind2 = 0; ind2 < 18; ind2++) {
        ret.voxel_neighborhood[ind2] = new int[ret.n_voxel];
        
    }
    ret.boundary_flag = new int[ret.n_voxel];
    for (int ind = 0; ind < json_in["voxel"].size(); ind++) {
        ret.voxel[ind] = glm::vec3(json_in["voxel"][ind][0].asDouble(), json_in["voxel"][ind][1].asDouble(), json_in["voxel"][ind][2].asDouble());
        for (int ind2 = 0; ind2 < 18; ind2++) {
            ret.voxel_neighborhood[ind2][ind] = json_in["voxel_neighborhood"][ind][ind2].asInt() - 1;
        }
        ret.boundary_flag[ind] = json_in["boundary_flag"][ind].asInt();
    }
    file.close();
    file.clear();
    
    json_in.clear();
    
    // load j_stim
    file.open(j_stim_path);
    file >> json_in;
    ret.J_stim.n_voxel = json_in["voxel_ind"].size();
    ret.J_stim.voxel_ind = new int[ret.J_stim.n_voxel];
    ret.J_stim.count = new int[ret.J_stim.n_voxel + 1];
    ret.J_stim.count[0] = 0;
    for (int ind = 0; ind < ret.J_stim.n_voxel; ind++) {
        ret.J_stim.voxel_ind[ind] = json_in["voxel_ind"][ind].asInt() - 1;
        ret.J_stim.count[ind + 1] = json_in["count"][ind].asInt();
    }
    
    int step_length = ret.J_stim.count[ret.J_stim.n_voxel - 1];
    ret.J_stim.step = new int[step_length];
    ret.J_stim.value = new double[step_length];
    for (int ind = 0; ind < step_length; ind++) {
        ret.J_stim.step[ind] = json_in["step"][ind].asInt();
        ret.J_stim.value[ind] = json_in["value"][ind].asDouble();
    }
    
    file.close();
    file.clear();
    json_in.clear();
    
    return ret;
}

