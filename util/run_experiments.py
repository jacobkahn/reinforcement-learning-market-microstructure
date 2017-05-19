import tensorflow as tf 
import numpy as np
import os
from environment import *
from collections import defaultdict
from deepQ import *
from Q import *
from Qfunction_approx import *
import csv # for reading
import sys
import multiprocess # for multithreading



def produce_envs(ticker, path):
	envs = {}
	filelist = os.listdir(path)
	filtered_data_sources = [path + '/' + item for item in filelist if ticker in item]
	test_env = Environment(filtered_data_sources[-1], setup=False, time=True)
	for orderbook_file in filtered_data_sources[:-1]:
		envs[orderbook_file] = Environment(filtered_data_sources[-1], setup=False, time=True)
	return envs, test_env


if __name__ == "__main__":

	# Get data from all files that contain string (i.e. AAPL)
	TICKER = 'AAPL'
	# Server
	PATH_TO_DATA = '../../data-output-unzipped'
	# Local
	# PATH_TO_DATA = '../data'
	# define method Paramsms'
	envs, test_env = produce_envs(TICKER, PATH_TO_DATA)


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
	V = 10000
	H = 1000
	T = 10 
	I = 10
	w_S = 1
	S = 1
	L = 5
	window = 10
	ob_size = L
	hidden_size = 5
	depth = 2
	actions = L + 1
	batch = 1
	epochs = 3
	continuous = True
	stateful = True
	test_steps = len(test_env.books) - H - 1
	divs = 10
	params = {
		'backup': 'sampling',
		'network': 'CNN',
		'advantage': True,
		'replay': True,
		'replay_size': 10,
		'replays': 0,
		'window': window,
		'ob_size': ob_size,
		'hidden_size': hidden_size, 
		'depth': depth, 
		'actions': actions, 
		'batch': batch,
		'continuous': continuous,
		'stateful': stateful,
		'length': 1,
		'H': H, 
		'V': V,
		'T': T,
		'I': I,
		'T': T,
		'S': S,
		'w_S': w_S,
		'L': L
	}

	# tables
	doubleQProcess = multiprocess.Process(target=dp_algo, args=(ob_file, H, V, I, T, L, doubleQbackup, S, divs, test_steps), kwargs={'envs': envs, 'test_env':test_env})
	samplingProcess = multiprocess.Process(target=dp_algo, args=(ob_file, H, V, I, T, L, samplingBackup, S, divs, test_steps), kwargs={'envs': envs, 'test_env':test_env})
	#replayBufferProcess = multiprocess.Process(target=dp_algo, args=(ob_file, H, V, I, T, L, replayBufferBackup, S, divs, test_steps), kwargs={'env': environment})
	# start
	doubleQProcess.start()
	samplingProcess.start()
	#replayBufferProcess.start()

	# function approx
	func_doubleQProcess = multiprocess.Process(target=dp_algo, args=(ob_file, H, V, I, T, L, doubleQbackup, S, divs, test_steps), kwargs={'envs': envs, 'test_env':test_env, 'func_approx': "linear"})
	func_samplingProcess = multiprocess.Process(target=dp_algo, args=(ob_file, H, V, I, T, L, samplingBackup, S, divs, test_steps), kwargs={'envs': envs, 'test_env':test_env, 'func_approx': "linear"})
	#func_replayBufferProcess = multiprocess.Process(target=dp_algo, args=(ob_file, H, V, I, T, L, replayBufferBackup, S, divs, test_steps), kwargs={'env': environment, 'func_approx': "linear"})
	# start
	func_doubleQProcess.start()
	func_samplingProcess.start()
	#func_replayBufferProcess.start()

	# deep learning
	#DQN_process = multiprocess.Process(target=train_DQN, args=(epochs, ob_file, H, V, I, T, L, S, test_steps, rnn_params), kwargs={'env': environment})
	# start
	#DQN_process.start()
