To run the code, it will need:
Python 3.6
Python modules: numpy 1.12.1, matplotlib 3.0.2


train_model(training_data_files, model_file):
Train a model by the given training_data_files, save the model in one txt file model_file.
training_data_files: 2D 3*3 list, contains (path and) names for training files:
[[base_ch1, base_ch2, base_ch3],
[front_ch1, front_ch2, front_ch3],
[right_ch1, right_ch2, right_ch3]]
model_file: string, file path and filename to save the model file


failure_detection(data, model)
Return [failure_mode, orientation, failure_channel]
data: 1D numpy array, one tuple channel data to be detected
model: trained model


adaptation(data, detection_info, model)
Return adapted data
data: 1D numpy array, one tuple channel data to be adapted
detection_info: the list returned by failure_detection
model: trained model
