import tensorflow as tf 
import numpy as np
import os
from environment import *
from q_learners import *
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
	H = 10000
	T = 2 
	I = 2
	w_S = 1
	S = 10
	L = 5
	window = 50
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
		'replay_size': 1000,
		'replays': 10,
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

	layers = {
		'A-conv1': {
			'type': 'conv',
			'size': 2,
			'stride': 1,
			'num': 30
		},
		'C-pool1': {
			'type': 'pool',
			'stride': 2,
			'size': 2,
			'pool_type': 'max'
		},
		'B-conv2': {
			'type': 'conv',
			'size': 3,
			'stride': 2,
			'num': 20
		}
	}
	params['layers'] = layers

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
	params['network'] = 'RNN'
	RNN_DQN_process_dp = multiprocess.Process(target=train_DQN_DP, args=(epochs, ob_file, params, test_steps), kwargs={'envs': envs, 'test_env':test_env, test_file_name:'RNN_DP-trades'})
	params['network'] = 'CNN'
	CNN_DQN_process_dp = multiprocess.Process(target=train_DQN_DP, args=(epochs, ob_file, params, test_steps), kwargs={'envs': envs, 'test_env':test_env, test_file_name:'CNN_DP-trades'})

	params['network'] = 'RNN'
	RNN_DQN_process_warmup = multiprocess.Process(target=train_DQN_DP_warmup, args=(epochs, ob_file, params, test_steps), kwargs={'envs': envs, 'test_env':test_env, test_file_name:'RNN_warmup-trades'})
	params['network'] = 'CNN'
	CNN_DQN_process_warmup = multiprocess.Process(target=train_DQN_DP_warmup, args=(epochs, ob_file, params, test_steps), kwargs={'envs': envs, 'test_env':test_env, test_file_name:'CNN_warmup-trades'})

	# start
	RNN_DQN_process_dp.start()
	CNN_DQN_process_dp.start()
	RNN_DQN_process_warmup.start()
	CNN_DQN_process_warmup.start()	




