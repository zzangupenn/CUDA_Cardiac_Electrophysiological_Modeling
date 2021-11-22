#include "data_handler.h"
using namespace std;


simulation_inputs import_para(string setting_path, string input_path) {
    
	simulation_inputs ret;
    ifstream file;
    Json::Value json_in;
    file.open(setting_path);
    file >> json_in;
    ret.dt = json_in["dt"].asDouble();
    ret.final_t = json_in["final_t"].asInt();
    ret.use_gpu = json_in["use_gpu"].asInt();
    ret.visualization = json_in["visualization"].asInt();
    ret.visualization_resolution = new int[2];
    ret.visualization_resolution[0] = json_in["visualization_resolution"][0].asInt();
    ret.visualization_resolution[1] = json_in["visualization_resolution"][1].asInt();
    ret.save_visulization = json_in["save_visulization"].asInt();
    ret.save_result = json_in["save_result"].asInt();
    ret.save_result_filename = json_in["save_result_filename"].asString();
    ret.save_data_min_clip = json_in["save_data_min_clip"].asDouble();

    file.close();
    file.clear();
    json_in.clear();
    
    // load volume
    file.open(input_path);
    file >> json_in;
    ret.voxel = new glm::vec3[json_in["volume"]["voxel"].size()];
    ret.n_voxel = json_in["volume"]["voxel"].size();
    ret.voxel_neighborhood = new int*[18];
    
    for (int ind2 = 0; ind2 < 18; ind2++) {
        ret.voxel_neighborhood[ind2] = new int[ret.n_voxel];
    }
    ret.boundary_flag = new int[ret.n_voxel];
    ret.tau_in = new double[ret.n_voxel];
    ret.tau_out = new double[ret.n_voxel];
    ret.tau_open = new double[ret.n_voxel];
    ret.tau_close = new double[ret.n_voxel];
    ret.v_gate = new double[ret.n_voxel];
    ret.c_for_D = new double[ret.n_voxel];
    ret.D = new double*[ret.n_voxel];


    for (int ind = 0; ind < json_in["volume"]["voxel"].size(); ind++) {

        ret.voxel[ind] = glm::vec3(json_in["volume"]["voxel"][ind][0].asDouble(), json_in["volume"]["voxel"][ind][1].asDouble(), json_in["volume"]["voxel"][ind][2].asDouble());
        ret.boundary_flag[ind] = json_in["volume"]["boundary_flag"][ind].asInt();
        ret.delta = json_in["volume"]["delta"].asDouble();
        ret.c_for_D[ind] = json_in["c_for_D"][ind].asDouble();
        ret.tau_in[ind] = json_in["tau_in"][ind].asDouble();
        ret.tau_out[ind] = json_in["tau_out"][ind].asDouble();
        ret.tau_open[ind] = json_in["tau_open"][ind].asDouble();
        ret.tau_close[ind] = json_in["tau_close"][ind].asDouble();
        ret.v_gate[ind] = json_in["v_gate"][ind].asDouble();

        ret.D[ind] = new double[9];
        for (int ind2 = 0; ind2 < 9; ind2++) {
            ret.D[ind][ind2] = json_in["D"][ind][ind2].asDouble();
        }

        for (int ind2 = 0; ind2 < 18; ind2++) {
            ret.voxel_neighborhood[ind2][ind] = json_in["volume"]["voxel_neighborhood"][ind][ind2].asInt() - 1;
        }
    }
    
    // load j_stim
    ret.J_stim.n_voxel = json_in["J_stim"]["voxel_ind"].size();
    ret.J_stim.voxel_ind = new int[ret.J_stim.n_voxel];
    ret.J_stim.count = new int[ret.J_stim.n_voxel + 1];
    ret.J_stim.count[0] = 0;
    for (int ind = 0; ind < ret.J_stim.n_voxel; ind++) {
        ret.J_stim.voxel_ind[ind] = json_in["J_stim"]["voxel_ind"][ind].asInt() - 1;
        ret.J_stim.count[ind + 1] = json_in["J_stim"]["count"][ind].asInt();
    }
    int step_length = ret.J_stim.count[ret.J_stim.n_voxel];
    ret.J_stim.step = new int[step_length];
    ret.J_stim.value = new double[step_length];

    for (int ind = 0; ind < step_length; ind++) {
        ret.J_stim.step[ind] = json_in["J_stim"]["step"][ind].asInt();
        ret.J_stim.value[ind] = json_in["J_stim"]["value"][ind].asDouble();
    }
    
    file.close();
    file.clear();
    json_in.clear();
    return ret;
}

int save_data(simulation_inputs sim_input, simulation_outputs sim_output, double save_data_min_clip) {

    std::ofstream fd;
    //fd.open(sim_input.save_result_filename + "0.json");
    fd.open(sim_input.save_result_filename + ".json");
    fd << "{\"final_t\":";
    fd << to_string(sim_input.final_t);
    fd << ",\"n_voxel\":";
    fd << to_string(sim_output.n_voxel);
    fd << ",\"action_potentials\":[";

    //fd << "{\"action_potentials\":";
    //int count = 0;
    for (int ind = 0; ind < sim_output.n_step; ind += int(1 / sim_input.dt)) {
    //for (int ind = 0; ind < 2; ind += 1) {
        //if (ind % 1000 == 0 && ind != 0) {
        //    fd << "]}";
        //    fd.close();
        //    count++;
        //    fd.open(sim_input.save_result_filename + to_string(count) + ".json");
        //    fd << "{\"action_potentials\":";
        //}
        fd << "[";
        for (int ind2 = 0; ind2 < sim_output.n_voxel; ind2++) {
            if (sim_output.action_potentials[ind][ind2] < save_data_min_clip) {
                fd << "0";
            }
            else {
                fd << to_string(sim_output.action_potentials[ind][ind2]);
            }
            if (ind2 != sim_output.n_voxel - 1) {
                fd << ",";
            }
        }
        if (ind != sim_output.n_step - int(1 / sim_input.dt)) {
            fd << "],";
        }
        else {
            fd << "]";
        }
    }
    fd << "]}";
    fd.close();

    // not usable, preparing the json file take too much memory
    //Json::Value json_out;
    //Json::Value json_array(Json::arrayValue);
    //json_out["n_step"] = sim_output.n_step;
    //json_out["n_voxel"] = sim_output.n_voxel;
    //for (int ind = 0; ind < sim_output.n_step; ind++) {
    //    Json::Value json_array2(Json::arrayValue);
    //    for (int ind2 = 0; ind2 < sim_output.n_voxel; ind2++) {
    //        json_array2.append(sim_output.action_potentials[ind][ind2]);
    //    }
    //    json_array.append(json_array2);
    //}
    //json_out["action_potentials"] = json_array;
    //
    //Json::StreamWriterBuilder builder;
    //builder["commentStyle"] = "None";
    //builder["indentation"] = "";
    //std::unique_ptr<Json::StreamWriter> writer(builder.newStreamWriter());
    //std::ofstream outputFileStream("sim_result.json");
    //writer->write(json_out, &outputFileStream);


}