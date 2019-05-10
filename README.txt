To run the code, it will need:
Python 3.6
Python modules: numpy 1.12.1, scipy 0.13.3, scikit-learn 0.20.2

data_preprocess(folder_path, data_filenames, valid_data_length=500)
Read signal channel data from original data files, and then clean data by extracting valid part of data and do some signal alignments. Return the cleaned data.
folder_path: string, the path of the folder that contains data files
data_filenames: list of string, names of each data file


get_target_channle(data, channel_index)
Divide cleaned data into two part: the data of the specified channel wanted to be reconstrued later, and the data of other channels that are used to reconstrut the data of the specified channel. Return those two part data, X and Y.
data: 2D numpy array, all cleaned data returned by data_preprocess()
channel_index: int, the index of the channel to be specified


train_model(train_X, train_Y, model_filename):
Train a machine learing model by the train_X and train_Y, store the model in the file model_filename. Return the model.
train_X: 2D numpy array, features value for training data
train_Y: 1D numpy array, target value for training data
model_filename: string, file path and filename to save the model file


failure_detection(channels_datapoint, model, channel_index)
Return failure confidence a specified channel given one tuple channel data
channels_datapoint: 1D numpy array, one tuple channel data to be detected
model: trained model
channel_index: int, index of the channel to be detected


adaptation(channels_datapoint, model, channel_index)
Return adapted data of a specified channel given one tuple channel data
channels_datapoint: 1D numpy array, one tuple channel data to be adapted
model: trained model
channel_index: int, index of the channel to be detected


test_model(folder_path, data_filenames, model_filename, adapted_data_filename, channel_index)
Create a file of adapted data derived from testing data by the model
folder_path: string, the path of the folder that contains testing data files
data_filenames: list of string, names of each testing data file
model_filename: string, file path and filename of the model file
adapted_data_filename: string, file path and filename of the file to save the adapted data
channel_index: int, the index of the channle whose data needs to be reconstruced

read from test_data_filename
read from model_file_name 

for each sample X
    failure_confidence = failure_detection(X, model, channel_index)
        if failure_confidemce > 2 
            do adaptation(X, model, channel_index)  
            adapted_value -> X(channel_index)
        end
    end
end
write adapted_data_filename
