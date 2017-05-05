import tensorflow as tf 
import numpy as np
from environment import *
from collections import defaultdict
from deepQ import *
from Q import *
from Qfunction_approx import *
import csv # for reading
import sys
import multiprocess # for multithreading



if __name__ == "__main__":

	# define method params
	doubleQbackup = {
		'name': 'doubleQ'
	}
	samplingBackup = {
		'name': 'sampling'
	}
	replayBufferBackup = { 'name': 'replay buffer',
					'buff_size': 50,
					'replays': 5
	}
	ob_file = "../data/10_GOOG.csv"
	V = 2500
	H = 2500
	T = 10 
	I = 10
	S = 10000
	L = 10
	window = 10
	ob_size = L
	hidden_size = 5
	depth = 3
	actions = L + 1
	batch = 1
	epochs = 3
	rnn_params = Params(window, ob_size, hidden_size, depth, actions, batch)
	test_steps = 100000
	divs = 10

	environment = Environment(ob_file, setup=False)

	# tables
	doubleQProcess = multiprocess.Process(target=dp_algo, args=(ob_file, H, V, I, T, L, doubleQbackup, S, divs, test_steps), kwargs={'env': environment})
	samplingProcess = multiprocess.Process(target=dp_algo, args=(ob_file, H, V, I, T, L, samplingBackup, S, divs, test_steps), kwargs={'env': environment})
	replayBufferProcess = multiprocess.Process(target=dp_algo, args=(ob_file, H, V, I, T, L, replayBufferBackup, S, divs, test_steps), kwargs={'env': environment})
	# start
	#doubleQProcess.start()
	#samplingProcess.start()
	#replayBufferProcess.start()

	# function approx
	func_doubleQProcess = multiprocess.Process(target=dp_algo, args=(ob_file, H, V, I, T, L, doubleQbackup, S, divs, test_steps), kwargs={'env': environment, 'func_approx': "linear"})
	func_samplingProcess = multiprocess.Process(target=dp_algo, args=(ob_file, H, V, I, T, L, samplingBackup, S, divs, test_steps), kwargs={'env': environment, 'func_approx': "linear"})
	func_replayBufferProcess = multiprocess.Process(target=dp_algo, args=(ob_file, H, V, I, T, L, replayBufferBackup, S, divs, test_steps), kwargs={'env': environment, 'func_approx': "linear"})
	# start
	#func_doubleQProcess.start()
	#func_samplingProcess.start()
	#func_replayBufferProcess.start()

	# deep learning
	DQN_process = multiprocess.Process(target=train_DQN, args=(epochs, ob_file, H, V, I, T, L, S, test_steps, rnn_params), kwargs={'env': environment})
	# start
	DQN_process.start()
