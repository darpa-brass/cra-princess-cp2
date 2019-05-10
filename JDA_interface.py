import numpy as np
from numpy import genfromtxt
from sklearn.svm import SVR
from sklearn.externals import joblib

def data_preprocess(folder_path, data_filenames, valid_data_length=500):
    channels_data = []
    for filename in data_filenames:
        channel_data = genfromtxt(folder_path + filename, delimiter=',')
        start_time  = np.sum(np.where(channel_data[:, 0] == 0))
        channel_data = channel_data[start_time: start_time + valid_data_length, 1]
        channels_data.append(channel_data)

    channel1 = channels_data[0]
    for channel_index in range(1, len(channels_data)):
        channel_i = channels_data[channel_index]
        min_diff = float('infinity')
        for shift in range(channel_i.size):
            shift_channel_i = np.roll(channel_i, -shift)
            cur_diff = np.mean(np.power((channel1 - shift_channel_i), 2))
            if cur_diff < min_diff:
                min_diff = cur_diff
                shift_step = -shift
        shift_channel_i = np.roll(channel_i, shift_step)
        channels_data[channel_index] = shift_channel_i

    start_point = 80
    for channel_index in range(len(channels_data)):
        channels_data[channel_index] = channels_data[channel_index][start_point:]

    data = np.vstack(channels_data)
    data = data.T
    return data

def get_target_channle(data, channel_index):
    reshaped_data = data.T
    Y = reshaped_data[channel_index]
    X = np.vstack([reshaped_data[0 : channel_index, :], reshaped_data[channel_index + 1 : , :]])
    X = X.reshape([X.shape[1], X.shape[0]])
    return X, Y

def train_model(train_X, train_Y, model_filename):
    model = SVR(kernel='rbf', C=1e3, gamma=0.1)
    model.fit(train_X, train_Y)
    joblib.dump(model, model_filename)
    return model


def failure_detection(channels_datapoint, model, channel_index):
    y = channels_datapoint[channel_index]
    x = np.append(channels_datapoint[0 : channel_index], channels_datapoint[channel_index + 1 :])
    x = x.reshape([1, x.size])
    predict_y = model.predict(x)
    failure_confidence = abs((predict_y[0] - y))
    return failure_confidence


def adaptation(channels_datapoint, model, channel_index):
    x = np.append(channels_datapoint[0 : channel_index], channels_datapoint[channel_index + 1 :])
    x = x.reshape([1, x.size])
    predict_y = model.predict(x)
    adapted_channels_value = np.append(channels_datapoint[0 : channel_index], predict_y)
    adapted_channels_value = np.append(adapted_channels_value, channels_datapoint[channel_index + 1 :])
    return adapted_channels_value


def test_model(folder_path, data_filenames, model_filename, adapted_data_filename, channel_index):
    model = joblib.load(model_filename)
    test_data = data_preprocess(folder_path, data_filenames)
    channel_num = test_data.shape[1]
    test_num = test_data.shape[0]

    std = np.std(test_data[:, channel_index])
    for i in range(test_num):
        if(np.random.random() > 0.8):
            test_data[i, channel_index] += 3 * std
            
    adapted_data = np.zeros((test_num, channel_num))
    for data_i in range(test_num):
        one_test_data = test_data[data_i]
        failure_conf = failure_detection(one_test_data, model, channel_index)

        if failure_conf > 2 * std:
            adp_val = adaptation(one_test_data, model, channel_index)
            adapted_data[data_i] = adp_val
        else:
            adapted_data[data_i] = test_data[data_i]

    adapted_data_text = ""
    for data in adapted_data:
        for channel_data in data:
            adapted_data_text +=  str(channel_data) + ","
        adapted_data_text = adapted_data_text[:-1]
        adapted_data_text += '\n'
    adapted_data_text = adapted_data_text.strip()

    with open(adapted_data_filename, 'w') as adapted_data_file:
        adapted_data_file.write(adapted_data_text)
    return


if __name__ == "__main__":

    folder_path = "Samples/Sample1/Working/"
    data_filenames = ['ch1.CSV', 'ch2.CSV', 'ch3.CSV']
    train_data = data_preprocess(folder_path, data_filenames)
    train_X, train_Y = get_target_channle(train_data, 0)
    model = train_model(train_X, train_Y, 'model.m')

    test_model(folder_path, data_filenames, "model.m", "Samples/adapted_data.txt", 0)
