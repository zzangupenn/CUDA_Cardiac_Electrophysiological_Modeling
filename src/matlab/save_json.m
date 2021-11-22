simulation_input_json = {};

% save the J_stim
J_stim_struct = {};
J_stim_struct.voxel_ind = simulation_input.stimulus.voxel_id;
step_list = [];
value_list = [];
count_list = [];
count = 0;
for voxel_ind = 1:length(simulation_input.stimulus.voxel_id)
    for ind = 1:length(simulation_input.stimulus.signal(voxel_ind, :))
        if simulation_input.stimulus.signal(voxel_ind, ind) ~= 0
            step_list = [step_list, ind];
            value_list = [value_list, simulation_input.stimulus.signal(voxel_ind, ind)];
            count = count + 1;
        end
    end
    count_list = [count_list, count];
end
J_stim_struct.step = step_list;
J_stim_struct.value = value_list;
J_stim_struct.count = count_list;
simulation_input_json.J_stim = J_stim_struct;

simulation_input_json.tau_in = simulation_input.tau_in_voxel;
simulation_input_json.tau_close = simulation_input.tau_close_voxel;
simulation_input_json.tau_open = simulation_input.tau_open_voxel;
simulation_input_json.tau_out = simulation_input.tau_out_voxel;
simulation_input_json.v_gate = simulation_input.v_gate_voxel;
simulation_input_json.c_for_D = simulation_input.c_voxel;

D_json = [];
for ind = 1:simulation_input.n_voxel
    D_json = [D_json, [simulation_input.D0{ind}(:)]];
end
D_json = D_json';
simulation_input_json.D = D_json;

% save volume
volume_struct = {};
volume_struct.voxel = simulation_input.geometry.volume.voxel;
volume_struct.voxel_neighborhood = simulation_input.geometry.volume.voxel_based_voxels;
volume_struct.boundary_flag = simulation_input.geometry.volume.boundary_flag;
volume_struct.delta = simulation_input.geometry.volume.delta;
simulation_input_json.volume = volume_struct;

JSONFILE_name = 'simulation_input.json'; 
fid = fopen(JSONFILE_name, 'w');
encodedJSON = jsonencode(simulation_input_json); 
fprintf(fid, encodedJSON); 