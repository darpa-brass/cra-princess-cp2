To run the code, it will need:
Python 3.6
Octave 4.4.1
Python modules: numpy 1.12.1, oct2py 4.0.6


train_model(train_data_filename, model_filename)
Create a file to save the model trained by training data.
train_data_filename: file path and filename of training data
model_filename: file path and filename of the file to save model

train_model('data/train_data.txt', 'model.txt');
call: model = train_model(sensor_data);
write: model to model.txt


failure_detection(sensor_data, model, sensor_index)
Return failure confidence(possibility) and failure type of data of a specified sensor given one tuple sensor data
sensor_data: one tuple sensor data to be detected
model: trained model
sensor_index: index of the sensor to be detected

adaptation(sensor_data, model, sensor_index)
Return adapted data and adaptation error of data of a specified sensor given one tuple sensor data
model: trained model
sensor_index: index of the sensor to be detected


test_model(test_data_filename, model_filename, adapted_data_filename)
Create a file of adapted data derived from testing data by the model
test_data_filename: file path and filename of the testing data file
model_filename: file path and filename of the model file
adapted_data_filename: file path and filename of the file to save the adapted data

read from test_data_filename
read from model_file_name 

for each sample X
    for each target_index
        do failure_detection(X, model, target_index)   call: output_info = detect(X, target_index, model) 
        if failure_confidemce > 0.5 . % failure
            do adaptation(X, model, target_index)  call: adapt_info = adapt(X, target_index, model)
            adapted_value -> X(target_index)
        end
    end
end
write adapted_data_filename
