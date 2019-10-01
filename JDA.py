import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
import os

NO_FAILURE = 0
DISCONNECTION = 1

ANY_ORIENTATION = 0
BASE = 1
FRONT = 2
RIGHT = 3

def train_model(training_data_files, model_file):
	base_ch1 = genfromtxt(training_data_files[0][0], delimiter=',')
	base_ch2 = genfromtxt(training_data_files[0][1], delimiter=',')
	base_ch3 = genfromtxt(training_data_files[0][2], delimiter=',')

	front_ch1 = genfromtxt(training_data_files[1][0], delimiter=',')
	front_ch2 = genfromtxt(training_data_files[1][1], delimiter=',')
	front_ch3 = genfromtxt(training_data_files[1][2], delimiter=',')

	right_ch1 = genfromtxt(training_data_files[2][0], delimiter=',')
	right_ch2 = genfromtxt(training_data_files[2][1], delimiter=',')
	right_ch3 = genfromtxt(training_data_files[2][2], delimiter=',')

	model = ""
	model += str(np.mean(base_ch1)) + ","
	model += str(np.mean(base_ch2)) + ","
	model += str(np.mean(base_ch3)) + "\n"

	model += str(np.mean(front_ch1)) + ","
	model += str(np.mean(front_ch2)) + ","
	model += str(np.mean(front_ch3)) + "\n"

	model += str(np.mean(right_ch1)) + ","
	model += str(np.mean(right_ch2)) + ","
	model += str(np.mean(right_ch3)) + "\n"

	base_shift_ch12 = base_ch1 - base_ch2
	base_shift_ch13 = base_ch1 - base_ch3
	base_shift_ch23 = base_ch2 - base_ch3

	front_shift_ch12 = front_ch1 - front_ch2
	front_shift_ch13 = front_ch1 - front_ch3
	front_shift_ch23 = front_ch2 - front_ch3

	right_shift_ch12 = right_ch1 - right_ch2
	right_shift_ch13 = right_ch1 - right_ch3
	right_shift_ch23 = right_ch2 - right_ch3

	model += str(np.mean(base_shift_ch12)) + ","
	model += str(np.mean(base_shift_ch13)) + ","
	model += str(np.mean(base_shift_ch23)) + "\n"

	model += str(np.mean(front_shift_ch12)) + ","
	model += str(np.mean(front_shift_ch13)) + ","
	model += str(np.mean(front_shift_ch23)) + "\n"

	model += str(np.mean(right_shift_ch12)) + ","
	model += str(np.mean(right_shift_ch13)) + ","
	model += str(np.mean(right_shift_ch23)) + "\n"

	model += str(normalized_var(base_shift_ch12)) + ","
	model += str(normalized_var(base_shift_ch13)) + ","
	model += str(normalized_var(base_shift_ch23)) + "\n"

	model += str(normalized_var(front_shift_ch12)) + ","
	model += str(normalized_var(front_shift_ch13)) + ","
	model += str(normalized_var(front_shift_ch23)) + "\n"

	model += str(normalized_var(right_shift_ch12)) + ","
	model += str(normalized_var(right_shift_ch13)) + ","
	model += str(normalized_var(right_shift_ch23))

	with open(model_file, 'w') as build_model:
		build_model.write(model)

def normalized_var(array):
	abs_array = np.abs(array)
	return np.var(abs_array / np.max(abs_array))


def failure_detection(data, model):
	failure_mode = NO_FAILURE
	orientation = ANY_ORIENTATION
	failed_ch = 0

	ch1 = data[0]
	ch2 = data[1]
	ch3 = data[2]

	shift_ch12 = ch1 - ch2
	shift_ch13 = ch1 - ch3
	shift_ch23 = ch2 - ch3

	max_deviation = 2000
	if np.max(np.abs(np.array([model[0][0], model[1][0], model[2][0]]) - ch1)) > max_deviation:
		failure_mode = DISCONNECTION
		failed_ch = 1
	elif np.max(np.abs(np.array([model[0][1], model[1][1], model[2][1]]) - ch2)) > max_deviation:
		failure_mode = DISCONNECTION
		failed_ch = 2
	elif np.max(np.abs(np.array([model[0][2], model[1][2], model[2][2]]) - ch3)) > max_deviation:
		failure_mode = DISCONNECTION
		failed_ch = 3

	if failure_mode == NO_FAILURE:
		max_shift_deviation = 200
		if (max(np.abs(shift_ch12 - model[3:6, 0])) > max_shift_deviation and max(np.abs(shift_ch13 - model[3:6, 1])) > max_shift_deviation):
			failure_mode = DISCONNECTION
			failed_ch = 1
		elif (max(np.abs(shift_ch12 - model[3:6, 0])) > max_shift_deviation and max(np.abs(shift_ch23 - model[3:6, 2])) > max_shift_deviation):
			failure_mode = DISCONNECTION
			failed_ch = 2
		elif (max(np.abs(shift_ch13 - model[3:6, 1])) > max_shift_deviation and max(np.abs(shift_ch23 - model[3:6, 2])) > max_shift_deviation):
			failure_mode = DISCONNECTION
			failed_ch = 3

	if failure_mode == NO_FAILURE:
		return failure_mode, orientation, failed_ch


	orientation_differences = np.zeros(3)
	if failed_ch == 1:
		orientation_differences[0] = abs(shift_ch23 - model[3][2])
		orientation_differences[1] = abs(shift_ch23 - model[4][2])
		orientation_differences[2] = abs(shift_ch23 - model[5][2])
		orientation = np.argmin(orientation_differences) + 1
	elif failed_ch == 2:
		orientation_differences[0] = abs(shift_ch13 - model[3][1])
		orientation_differences[1] = abs(shift_ch13 - model[4][1])
		orientation_differences[2] = abs(shift_ch13 - model[5][1])
		orientation = np.argmin(orientation_differences) + 1
	elif failed_ch == 3:
		orientation_differences[0] = abs(shift_ch12 - model[3][0])
		orientation_differences[1] = abs(shift_ch12 - model[4][0])
		orientation_differences[2] = abs(shift_ch12 - model[5][0])
		orientation = np.argmin(orientation_differences) + 1

		if orientation != RIGHT:
			avg_differences = np.zeros(2)
			avg_differences[0] = abs(ch1 - model[0][0]) + abs(ch2 - model[0][1])
			avg_differences[1] = abs(ch1 - model[1][0]) + abs(ch2 - model[1][1])
			orientation = np.argmin(avg_differences) + 1

	return failure_mode, orientation, failed_ch

def adaptation(data, detection_info, model):
	failure_mode = detection_info[0]
	if failure_mode == NO_FAILURE:
		return data

	orientation = detection_info[1]
	selected_shifts = model[orientation + 2]

	adapted_data = data
	failed_ch = detection_info[2]
	if failed_ch == 1:
		adapted_ch_data = ((selected_shifts[0] + data[1]) + (selected_shifts[1] + data[2])) / 2
	elif failed_ch == 2:
		adapted_ch_data = ((-selected_shifts[0] + data[0]) + (selected_shifts[2] + data[2])) / 2
	else:
		adapted_ch_data = ((-selected_shifts[1] + data[0]) + (-selected_shifts[2] + data[1])) / 2

	adapted_data[failed_ch - 1] = adapted_ch_data
	return adapted_data

def plot_chs(chs_data, setting):
	ch1 = chs_data[0]
	ch2 = chs_data[1]
	ch3 = chs_data[2]

	time_idx = [i for i in range(len(ch1))]
	plt.ion()
	plt.title(setting)
	plt.plot(time_idx, ch1, label='CH 1')
	plt.plot(time_idx, ch2, label='CH 2')
	plt.plot(time_idx, ch3, label='CH 3')
	plt.xlabel('Time Ticks (~122usecs)')
	plt.ylabel('Counts')
	plt.legend()
	plt.show()
	plt.savefig(setting + '.png')


if __name__ == "__main__":
	root_path = "instrumentation-test-data-master/mkz/data/"

	base_path = root_path + "base/"
	front_path = root_path + "front/"
	right_path = root_path + "right/"

	base_normal_path = base_path + "normal_01/"
	front_normal_path = front_path + "normal_01/"
	right_normal_path = right_path + "normal_01/"

	base_ch1_training_path = base_normal_path + "data_ch1.csv"
	base_ch2_training_path = base_normal_path + "data_ch2.csv"
	base_ch3_training_path = base_normal_path + "data_ch3.csv"

	front_ch1_training_path = front_normal_path + "data_ch1.csv"
	front_ch2_training_path = front_normal_path + "data_ch2.csv"
	front_ch3_training_path = front_normal_path + "data_ch3.csv"

	right_ch1_training_path = right_normal_path + "data_ch1.csv"
	right_ch2_training_path = right_normal_path + "data_ch2.csv"
	right_ch3_training_path = right_normal_path + "data_ch3.csv"

	training_data_files = ([[base_ch1_training_path, base_ch2_training_path, base_ch3_training_path], 
		[front_ch1_training_path, front_ch2_training_path, front_ch3_training_path], 
		[right_ch1_training_path, right_ch2_training_path, right_ch3_training_path]])

	model_file = "model.txt"
	train_model(training_data_files, model_file)
	model = genfromtxt(model_file, delimiter=',')


	selected_orientaion = "base"
	mode = "bad"
	test_root_path = root_path + selected_orientaion + "/" + mode + "_01/"
	ch1_testing_path = test_root_path + "data_ch1.csv"
	ch2_testing_path = test_root_path + "data_ch2.csv"
	ch3_testing_path = test_root_path + "data_ch3.csv"

	ch1_testing_data = genfromtxt(ch1_testing_path, delimiter=',')
	ch2_testing_data = genfromtxt(ch2_testing_path, delimiter=',')
	ch3_testing_data = genfromtxt(ch3_testing_path, delimiter=',')

	testing_data = np.vstack([ch1_testing_data, ch2_testing_data, ch3_testing_data])
	testing_data = testing_data.T

	adapted_data_set = np.zeros(testing_data.shape)
	for index in range(len(testing_data)):
		testing_data_point = np.copy(testing_data[index])
		detection_info = failure_detection(testing_data_point, model)
		adapted_data_point = adaptation(testing_data_point, detection_info, model)
		adapted_data_set[index] = adapted_data_point

	output_file = 'adapted_data.txt'
	np.savetxt(output_file, adapted_data_set, delimiter=',')

	# Save somewhere SwRI can access
	docker_output_dir = '/data/cp2/'
	os.makedirs(docker_output_dir, exist_ok=True)
	copyfile(output_file, docker_output_dir + output_file)
	#plot_chs(adapted_data_set.T, selected_orientaion + '_' + mode)




