function output_info = detect(sensor_data, target_index, adapt_model)

% sensor_data: N_sensor x 1 matrix
% target_index: sensor to detect
% adapt_model: N_model_length x 1 matrix

% output_info: 2 x 1 matrix
% 1st value is fail_conf: [0,1] 
% 2nd value is fail_type: [1,5]

fail_conf = rand * 0.7;
fail_type = randi(5);

output_info = [fail_conf, fail_type];
