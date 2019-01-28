import numpy as np
import matlab
import matlab.engine
eng = matlab.engine.start_matlab()


def train_model(train_data_filename, model_filename):
    train_data = np.loadtxt(train_data_filename, delimiter=",").tolist()
    train_data = matlab.double(train_data)
    model = eng.train_model(train_data)

    model_text = ""
    for value in model:
        model_text += str(value[0]) + '\n'
    model_text = model_text.strip()

    with open(model_filename, 'w') as model_file:
        model_file.write(model_text)
    return


def failure_detection(sensor_data, model, sensor_index):
    result = eng.detect(sensor_data, sensor_index + 1, model)
    failure_confidence = result[0][0]
    failure_type = result[0][1]
    return failure_confidence, failure_type


def adaptation(sensor_data, model, sensor_index):
    result = eng.adapt(sensor_data, sensor_index + 1, model)
    adapted_sensor_value = result[0][0]
    adaptation_error = result[0][1]
    return adapted_sensor_value, adaptation_error


def test_model(test_data_filename, model_filename, adapted_data_filename):
    model = np.loadtxt(model_filename)
    model = model.reshape(model.shape[0], 1)
    model = model.tolist()
    model = matlab.double(model)

    test_data = np.loadtxt(test_data_filename, delimiter=",")
    test_num = test_data.shape[1]
    sensor_num = test_data.shape[0]

    adapted_data = np.zeros((sensor_num, test_num))
    for data_i in range(test_num):
        one_test_data = matlab.double((test_data[:, data_i].reshape(sensor_num, 1)).tolist())
        for index in range(sensor_num):
            failure_conf, failure_type = failure_detection(one_test_data, model, index)
            if failure_conf > 0.5:
                adp_val, adp_err = adaptation(one_test_data, model, index)
                adapted_data[index, data_i] = adp_val
            else:
                adapted_data[index, data_i] = test_data[index, data_i]

    adapted_data_text = ""
    for sensor_data in range(sensor_num):
        for time_stamp in range(test_num):
            adapted_data_text += str(adapted_data[sensor_data, time_stamp])
            if time_stamp + 1 < test_num:
                adapted_data_text += ","
        adapted_data_text += '\n'
    adapted_data_text = adapted_data_text.strip()

    with open(adapted_data_filename, 'w') as adapted_data_file:
        adapted_data_file.write(adapted_data_text)
    return


if __name__ == "__main__":
    train_model("data/train_data.txt", "model.txt")
    test_model("data/test_data.txt", "model.txt", "data/adapted_data.txt")
