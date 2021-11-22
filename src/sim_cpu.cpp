#include "sim_cpu.h"

using namespace std;
using namespace Eigen;

ArrayXd sign_int(ArrayXi v) {
    ArrayXd v_ret = (v > 0).select(ArrayXd::Ones(v.rows()), ArrayXd::Zero(v.rows()));
    return v_ret;
}

simulation_data_parts prepare_parts(simulation_inputs sim_inputs) {
    Map<ArrayXi> px(sim_inputs.voxel_neighborhood[0], sim_inputs.n_voxel);
    Map<ArrayXi> mx(sim_inputs.voxel_neighborhood[1], sim_inputs.n_voxel);
    Map<ArrayXi> py(sim_inputs.voxel_neighborhood[2], sim_inputs.n_voxel);
    Map<ArrayXi> my(sim_inputs.voxel_neighborhood[3], sim_inputs.n_voxel);
    Map<ArrayXi> pz(sim_inputs.voxel_neighborhood[4], sim_inputs.n_voxel);
    Map<ArrayXi> mz(sim_inputs.voxel_neighborhood[5], sim_inputs.n_voxel);
    Map<ArrayXi> pxpy(sim_inputs.voxel_neighborhood[6], sim_inputs.n_voxel);
    Map<ArrayXi> mxpy(sim_inputs.voxel_neighborhood[7], sim_inputs.n_voxel);
    Map<ArrayXi> pxmy(sim_inputs.voxel_neighborhood[8], sim_inputs.n_voxel);
    Map<ArrayXi> mxmy(sim_inputs.voxel_neighborhood[9], sim_inputs.n_voxel);
    Map<ArrayXi> pypz(sim_inputs.voxel_neighborhood[10], sim_inputs.n_voxel);
    Map<ArrayXi> mypz(sim_inputs.voxel_neighborhood[11], sim_inputs.n_voxel);
    Map<ArrayXi> pymz(sim_inputs.voxel_neighborhood[12], sim_inputs.n_voxel);
    Map<ArrayXi> mymz(sim_inputs.voxel_neighborhood[13], sim_inputs.n_voxel);
    Map<ArrayXi> pxpz(sim_inputs.voxel_neighborhood[14], sim_inputs.n_voxel);
    Map<ArrayXi> mxpz(sim_inputs.voxel_neighborhood[15], sim_inputs.n_voxel);
    Map<ArrayXi> pxmz(sim_inputs.voxel_neighborhood[16], sim_inputs.n_voxel);
    Map<ArrayXi> mxmz(sim_inputs.voxel_neighborhood[17], sim_inputs.n_voxel);
    ArrayXXd D = ArrayXXd::Zero(sim_inputs.n_voxel, 9);
    for (int ind = 0; ind < sim_inputs.n_voxel; ind++) {
        for (int ind2 = 0; ind2 < 9; ind2++) {
            D(ind, ind2) = sim_inputs.D[ind][ind2];
        }
    };

    ArrayXd part1 = 4.0 * D(all, 0) * sign_int(px);
    ArrayXd part2 = 4.0 * D(all, 0) * sign_int(mx);
    ArrayXd part3 = 4.0 * D(all, 4) * sign_int(py);
    ArrayXd part4 = 4.0 * D(all, 4) * sign_int(my);
    ArrayXd part5 = 4.0 * D(all, 8) * sign_int(pz);
    ArrayXd part6 = 4.0 * D(all, 8) * sign_int(mz);
    ArrayXd part7 = ((D(all, 0)(px) - D(all, 0)(mx)) * sign_int(px) * sign_int(mx) + (D(all, 3)(py) - D(all, 3)(my)) * sign_int(py) * sign_int(my) + (D(all, 6)(pz) - D(all, 6)(mz)) * sign_int(pz) * sign_int(mz)) * sign_int(px) * sign_int(mx);
    ArrayXd part8 = ((D(all, 1)(px) - D(all, 1)(mx)) * sign_int(px) * sign_int(mx) + (D(all, 4)(py) - D(all, 4)(my)) * sign_int(py) * sign_int(my) + (D(all, 7)(pz) - D(all, 7)(mz)) * sign_int(pz) * sign_int(mz)) * sign_int(py) * sign_int(my);
    ArrayXd part9 = ((D(all, 2)(px) - D(all, 2)(mx)) * sign_int(px) * sign_int(mx) + (D(all, 5)(py) - D(all, 5)(my)) * sign_int(py) * sign_int(my) + (D(all, 8)(pz) - D(all, 8)(mz)) * sign_int(pz) * sign_int(mz)) * sign_int(pz) * sign_int(mz);
    ArrayXd part10 = 2.0 * D(all, 1) * sign_int(pxpy) * sign_int(pxmy);
    ArrayXd part11 = 2.0 * D(all, 1) * sign_int(mxpy) * sign_int(mxmy);
    ArrayXd part12 = 2.0 * D(all, 2) * sign_int(pxpz) * sign_int(pxmz);
    ArrayXd part13 = 2.0 * D(all, 2) * sign_int(mxpz) * sign_int(mxmz);
    ArrayXd part14 = 2.0 * D(all, 5) * sign_int(pypz) * sign_int(pymz);
    ArrayXd part15 = 2.0 * D(all, 5) * sign_int(mypz) * sign_int(mymz);
    
    simulation_data_parts ret;
    //ret.parts[0] = new double[sim_inputs.n_voxel];
    //Map<ArrayXd>(ret.parts[0], sim_inputs.n_voxel) = part1;
    //ret.parts[1] = new double[sim_inputs.n_voxel];
    //Map<ArrayXd>(ret.parts[1], sim_inputs.n_voxel) = part2;
    //ret.parts[2] = new double[sim_inputs.n_voxel];
    //Map<ArrayXd>(ret.parts[2], sim_inputs.n_voxel) = part3;
    //ret.parts[3] = new double[sim_inputs.n_voxel];
    //Map<ArrayXd>(ret.parts[3], sim_inputs.n_voxel) = part4;
    //ret.parts[4] = new double[sim_inputs.n_voxel];
    //Map<ArrayXd>(ret.parts[4], sim_inputs.n_voxel) = part5;
    //ret.parts[5] = new double[sim_inputs.n_voxel];
    //Map<ArrayXd>(ret.parts[5], sim_inputs.n_voxel) = part6;
    //ret.parts[6] = new double[sim_inputs.n_voxel];
    //Map<ArrayXd>(ret.parts[6], sim_inputs.n_voxel) = part7;
    //ret.parts[7] = new double[sim_inputs.n_voxel];
    //Map<ArrayXd>(ret.parts[7], sim_inputs.n_voxel) = part8;
    //ret.parts[8] = new double[sim_inputs.n_voxel];
    //Map<ArrayXd>(ret.parts[8], sim_inputs.n_voxel) = part9;
    //ret.parts[9] = new double[sim_inputs.n_voxel];
    //Map<ArrayXd>(ret.parts[9], sim_inputs.n_voxel) = part10;
    //ret.parts[10] = new double[sim_inputs.n_voxel];
    //Map<ArrayXd>(ret.parts[10], sim_inputs.n_voxel) = part11;
    //ret.parts[11] = new double[sim_inputs.n_voxel];
    //Map<ArrayXd>(ret.parts[11], sim_inputs.n_voxel) = part12;
    //ret.parts[12] = new double[sim_inputs.n_voxel];
    //Map<ArrayXd>(ret.parts[12], sim_inputs.n_voxel) = part13;
    //ret.parts[13] = new double[sim_inputs.n_voxel];
    //Map<ArrayXd>(ret.parts[13], sim_inputs.n_voxel) = part14;
    //ret.parts[14] = new double[sim_inputs.n_voxel];
    //Map<ArrayXd>(ret.parts[14], sim_inputs.n_voxel) = part15;
    
    ret.part1 = new double[sim_inputs.n_voxel];
    Map<ArrayXd>(ret.part1, sim_inputs.n_voxel) = part1;
    ret.part2 = new double[sim_inputs.n_voxel];
    Map<ArrayXd>(ret.part2, sim_inputs.n_voxel) = part2;
    ret.part3 = new double[sim_inputs.n_voxel];
    Map<ArrayXd>(ret.part3, sim_inputs.n_voxel) = part3;
    ret.part4 = new double[sim_inputs.n_voxel];
    Map<ArrayXd>(ret.part4, sim_inputs.n_voxel) = part4;
    ret.part5 = new double[sim_inputs.n_voxel];
    Map<ArrayXd>(ret.part5, sim_inputs.n_voxel) = part5;
    ret.part6 = new double[sim_inputs.n_voxel];
    Map<ArrayXd>(ret.part6, sim_inputs.n_voxel) = part6;
    ret.part7 = new double[sim_inputs.n_voxel];
    Map<ArrayXd>(ret.part7, sim_inputs.n_voxel) = part7;
    ret.part8 = new double[sim_inputs.n_voxel];
    Map<ArrayXd>(ret.part8, sim_inputs.n_voxel) = part8;
    ret.part9 = new double[sim_inputs.n_voxel];
    Map<ArrayXd>(ret.part9, sim_inputs.n_voxel) = part9;
    ret.part10 = new double[sim_inputs.n_voxel];
    Map<ArrayXd>(ret.part10, sim_inputs.n_voxel) = part10;
    ret.part11 = new double[sim_inputs.n_voxel];
    Map<ArrayXd>(ret.part11, sim_inputs.n_voxel) = part11;
    ret.part12 = new double[sim_inputs.n_voxel];
    Map<ArrayXd>(ret.part12, sim_inputs.n_voxel) = part12;
    ret.part13 = new double[sim_inputs.n_voxel];
    Map<ArrayXd>(ret.part13, sim_inputs.n_voxel) = part13;
    ret.part14 = new double[sim_inputs.n_voxel];
    Map<ArrayXd>(ret.part14, sim_inputs.n_voxel) = part14;
    ret.part15 = new double[sim_inputs.n_voxel];
    Map<ArrayXd>(ret.part15, sim_inputs.n_voxel) = part15;

    return ret;
}

simulation_outputs Simulation_CPU(simulation_inputs sim_inputs) {

    Map<ArrayXi> px(sim_inputs.voxel_neighborhood[0], sim_inputs.n_voxel);
    Map<ArrayXi> mx(sim_inputs.voxel_neighborhood[1], sim_inputs.n_voxel);
    Map<ArrayXi> py(sim_inputs.voxel_neighborhood[2], sim_inputs.n_voxel);
    Map<ArrayXi> my(sim_inputs.voxel_neighborhood[3], sim_inputs.n_voxel);
    Map<ArrayXi> pz(sim_inputs.voxel_neighborhood[4], sim_inputs.n_voxel);
    Map<ArrayXi> mz(sim_inputs.voxel_neighborhood[5], sim_inputs.n_voxel);
    Map<ArrayXi> pxpy(sim_inputs.voxel_neighborhood[6], sim_inputs.n_voxel);
    Map<ArrayXi> mxpy(sim_inputs.voxel_neighborhood[7], sim_inputs.n_voxel);
    Map<ArrayXi> pxmy(sim_inputs.voxel_neighborhood[8], sim_inputs.n_voxel);
    Map<ArrayXi> mxmy(sim_inputs.voxel_neighborhood[9], sim_inputs.n_voxel);
    Map<ArrayXi> pypz(sim_inputs.voxel_neighborhood[10], sim_inputs.n_voxel);
    Map<ArrayXi> mypz(sim_inputs.voxel_neighborhood[11], sim_inputs.n_voxel);
    Map<ArrayXi> pymz(sim_inputs.voxel_neighborhood[12], sim_inputs.n_voxel);
    Map<ArrayXi> mymz(sim_inputs.voxel_neighborhood[13], sim_inputs.n_voxel);
    Map<ArrayXi> pxpz(sim_inputs.voxel_neighborhood[14], sim_inputs.n_voxel);
    Map<ArrayXi> mxpz(sim_inputs.voxel_neighborhood[15], sim_inputs.n_voxel);
    Map<ArrayXi> pxmz(sim_inputs.voxel_neighborhood[16], sim_inputs.n_voxel);
    Map<ArrayXi> mxmz(sim_inputs.voxel_neighborhood[17], sim_inputs.n_voxel);
    Map<ArrayXd> tau_open(sim_inputs.tau_open, sim_inputs.n_voxel);
    Map<ArrayXd> tau_close(sim_inputs.tau_close, sim_inputs.n_voxel);
    Map<ArrayXd> tau_in(sim_inputs.tau_in, sim_inputs.n_voxel);
    Map<ArrayXd> tau_out(sim_inputs.tau_out, sim_inputs.n_voxel);
    Map<ArrayXd> v_gate(sim_inputs.v_gate, sim_inputs.n_voxel);
    ArrayXXd D = ArrayXXd::Zero(sim_inputs.n_voxel, 9);
    for (int ind = 0; ind < sim_inputs.n_voxel; ind++) {
        for (int ind2 = 0; ind2 < 9; ind2++) {
            D(ind, ind2) = sim_inputs.D[ind][ind2];
        }
    }
    double delta = sim_inputs.delta;

    ArrayXd part1 = 4.0 * D(all, 0) * sign_int(px);
    ArrayXd part2 = 4.0 * D(all, 0) * sign_int(mx);
    ArrayXd part3 = 4.0 * D(all, 4) * sign_int(py);
    ArrayXd part4 = 4.0 * D(all, 4) * sign_int(my);
    ArrayXd part5 = 4.0 * D(all, 8) * sign_int(pz);
    ArrayXd part6 = 4.0 * D(all, 8) * sign_int(mz);
    ArrayXd part7 = ((D(all, 0)(px) - D(all, 0)(mx)) * sign_int(px) * sign_int(mx) + (D(all, 3)(py) - D(all, 3)(my)) * sign_int(py) * sign_int(my) + (D(all, 6)(pz) - D(all, 6)(mz)) * sign_int(pz) * sign_int(mz)) * sign_int(px) * sign_int(mx);
    ArrayXd part8 = ((D(all, 1)(px) - D(all, 1)(mx)) * sign_int(px) * sign_int(mx) + (D(all, 4)(py) - D(all, 4)(my)) * sign_int(py) * sign_int(my) + (D(all, 7)(pz) - D(all, 7)(mz)) * sign_int(pz) * sign_int(mz)) * sign_int(py) * sign_int(my);
    ArrayXd part9 = ((D(all, 2)(px) - D(all, 2)(mx)) * sign_int(px) * sign_int(mx) + (D(all, 5)(py) - D(all, 5)(my)) * sign_int(py) * sign_int(my) + (D(all, 8)(pz) - D(all, 8)(mz)) * sign_int(pz) * sign_int(mz)) * sign_int(pz) * sign_int(mz);
    ArrayXd part10 = 2.0 * D(all, 1) * sign_int(pxpy) * sign_int(pxmy);
    ArrayXd part11 = 2.0 * D(all, 1) * sign_int(mxpy) * sign_int(mxmy);
    ArrayXd part12 = 2.0 * D(all, 2) * sign_int(pxpz) * sign_int(pxmz);
    ArrayXd part13 = 2.0 * D(all, 2) * sign_int(mxpz) * sign_int(mxmz);
    ArrayXd part14 = 2.0 * D(all, 5) * sign_int(pypz) * sign_int(pymz);
    ArrayXd part15 = 2.0 * D(all, 5) * sign_int(mypz) * sign_int(mymz);

    int total_step = sim_inputs.final_t / sim_inputs.dt;
    ArrayXd sim_v = ArrayXd::Zero(sim_inputs.n_voxel);
    ArrayXd sim_v_current = sim_v;
    ArrayXd sim_h = ArrayXd::Ones(sim_inputs.n_voxel);
    ArrayXd c_voxel = ArrayXd::Ones(sim_inputs.n_voxel);
    ArrayXd sim_h_1 = ArrayXd::Zero(sim_inputs.n_voxel);
    ArrayXd sim_h_2 = ArrayXd::Zero(sim_inputs.n_voxel);
    ArrayXd diffusion_term = ArrayXd::Zero(sim_inputs.n_voxel);
    ArrayXd J_stim_voxel = ArrayXd::Zero(sim_inputs.n_voxel);
    int J_stim_min_ind = INT_MAX;
    int J_stim_max_ind = 0;
    for (int ind = 0; ind < sim_inputs.J_stim.count[sim_inputs.J_stim.n_voxel - 1]; ind++) {
        if (J_stim_min_ind > sim_inputs.J_stim.step[ind]) J_stim_min_ind = sim_inputs.J_stim.step[ind];
        if (J_stim_max_ind < sim_inputs.J_stim.step[ind]) J_stim_max_ind = sim_inputs.J_stim.step[ind];
    }
    simulation_outputs sim_output;
    sim_output.n_step = total_step;
    sim_output.n_voxel = sim_inputs.n_voxel;
    sim_output.data_min = v_gate(0);
    sim_output.data_max = 1.0;
    sim_output.action_potentials = new double* [total_step];
    for (int ind = 0; ind < total_step; ind++) {
        sim_output.action_potentials[ind] = new double[sim_inputs.n_voxel];
    }

    //auto start = chrono::high_resolution_clock::now();
    for (int ind = 0; ind < total_step; ind++) {
        sim_v_current = sim_v;
        if (ind % 1000 == 0) {
            printf("%d\n", ind);
        }

        J_stim_voxel = ArrayXd::Zero(part1.rows());
        if (ind >= J_stim_min_ind && ind <= J_stim_max_ind) {
            for (int count_ind = 0; count_ind < sim_inputs.J_stim.n_voxel; count_ind++) {
                for (int step_ind = sim_inputs.J_stim.count[count_ind]; step_ind < sim_inputs.J_stim.count[count_ind + 1]; step_ind++) {
                    if (sim_inputs.J_stim.step[step_ind] - 1 == ind) {
                        J_stim_voxel(sim_inputs.J_stim.voxel_ind[count_ind]) = sim_inputs.J_stim.value[step_ind];
                    }
                }
            }
        }

        diffusion_term = 1.0 / (4.0 * pow(delta, 2.0)) *
            ((sim_v_current(px) - sim_v_current) * part1 + (sim_v_current(mx) - sim_v_current) * part2 +
                (sim_v_current(py) - sim_v_current) * part3 + (sim_v_current(my) - sim_v_current) * part4 +
                (sim_v_current(pz) - sim_v_current) * part5 + (sim_v_current(mz) - sim_v_current) * part6 +
                (sim_v_current(px) - sim_v_current(mx)) * part7 +
                (sim_v_current(py) - sim_v_current(my)) * part8 +
                (sim_v_current(pz) - sim_v_current(mz)) * part9 +
                (sim_v_current(pxpy) - sim_v_current(pxmy)) * part10 + (sim_v_current(mxmy) - sim_v_current(mxpy)) * part11 +
                (sim_v_current(pxpz) - sim_v_current(pxmz)) * part12 + (sim_v_current(mxmz) - sim_v_current(mxpz)) * part13 +
                (sim_v_current(pypz) - sim_v_current(pymz)) * part14 + (sim_v_current(mymz) - sim_v_current(mypz)) * part15);
        sim_v = ((sim_h * sim_v_current.pow(2.0) * (1 - sim_v_current) / tau_in) + (-sim_v_current / tau_out) + J_stim_voxel + diffusion_term) * sim_inputs.dt + sim_v_current;
        sim_h_1 = ((1 - sim_h) / tau_open) * sim_inputs.dt + sim_h;
        sim_h_2 = (-sim_h / tau_close) * sim_inputs.dt + sim_h;

        sim_h = (sim_v_current < v_gate).select(sim_h_1, sim_h_2);

        ArrayXd::Map(sim_output.action_potentials[ind], sim_output.n_voxel) = sim_v;
    }
    //auto duration = chrono::duration_cast<chrono::seconds>(chrono::high_resolution_clock::now() - start);
    //cout << "Computation took " << duration.count() << " seconds" << endl;

    return sim_output;
}