function adapt_info = adapt(sensor_data, target_index, adapt_model)

% sensor_data: N_sensor x 1 matrix
% target_index: sensor to adapt
% adapt_model: N_model_length x 1 matrix

% adapt_info 2 x 1 matrix
% 1st value is adapt_val: double
% 2nd value is adapt_err: double

adapt_info = zeros(2,1);

adapt_info = [sensor_data(target_index) + randn * 0.01, randn * 0.01];
