function model = train_model(sensor_data)

%% sensor_data is a matrix: N_sensor * N_sample
%% model: N_model_length x 1 matrix

X = sensor_data(1:2,:);
Y = sensor_data(3,:);
d = 2;
lambda = 0.01;

[weight, min_val, max_val] = train_ridge_reg(X, Y, d, lambda);

model = [min_val; max_val; weight];
