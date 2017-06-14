from environment import *
from agent import *
from q_learners import Q_CNN
from q_learners import Q_RNN
from q_learners import Params
from collections import defaultdict
from random import shuffle
import tensorflow as tf
import numpy as np
import csv
import os



class Q_Approx:
	def __init__(self, params):
		self.params = params
		b = params['backup']
		a = params['network']
		self.inp_buff = []
		self.targ_buff = []
		self.replay = []
		self.batchs = []
		self.input_batch = None
		self.targ_batch = None
		self.counter = 0
		n = ''.join(random.choice('abcdefhgijklmnop') for _ in range(15))
		self.Q = self.create_network(a, params, 'Q-' + n)
		self.Q_target = self.create_network(a, params, 'Q_target-' + n)
		self.buff = []
		self.updateTargetOperation = self.Q_target.copy_Q_Op(self.Q)
		self.choose_backup_networks()


	def create_network(self, arch, params, name):
		if self.params['backup'] == 'replay_buffer':
			self.replay = []
		if arch == 'CNN':
			return Q_CNN(params,  name)
		elif arch == 'RNN':
			return Q_RNN(params, name)
		else:
			print 'Illegal Architecture type!'
			return None

	def choose_backup_networks(self):
		self.target = self.Q_target
		self.fit = self.Q


	def batch_ready(self):
		return ((self.input_batch != None) and (self.targ_batch != None))

	def update_networks(self, sess):
		Q = self.fit
		inp = self.input_batch
		targ = self.targ_batch
		q_vals, loss, min_score, update = sess.run(
				(Q.predictions, Q.loss, Q.min_score, Q.updateWeights), 
				feed_dict={Q.input_place_holder: inp, Q.target_values: targ})
		self.input_batch = None
		self.targ_batch = None
		return q_vals, loss, min_score

	def calculate_target(self, sess, env, state, action, length, use_Q=None, reset=True):
		if use_Q is None:
			use_Q = self.target
		b = self.params['backup']
		ob_size = self.params['ob_size']
		window = self.params['window']
		batch = self.params['batch']
		stateful = self.params['stateful']
		continuous = self.params['continuous']
		time_unit = self.params['H']/self.params['T']
		vol_unit = self.params['V'] / self.params['I']
		T = self.params['T']
		i = state['inv']
		t = state['t']
		ts = state['ts']

		if self.counter % (batch * 5) == 0:
			sess.run(self.updateTargetOperation)

		backup = 0
		curr_action = action
		tgt_price = env.mid_spread(ts + time_unit * (self.params['T']- t))
		leftover = i

		states = []
		rewards = []
		trades_cost = 0

		if reset:
			left = (1.0 * leftover / self.params['V']) if continuous else int(round(leftover / vol_unit))
			s = create_input_window_stateless(env, ts, window, ob_size, t, left)
			if self.params['stateful']:
				env.get_timesteps(ts, ts + time_unit*length+ 1, self.params['I'], self.params['V'])
			states.append(s)
		# rollout bellman operator for multiple action steps
		for idx in range(length):
			# draw out current state and limit price
			old_leftover = leftover
			if stateful:
				curr_book = env.curr_book
			else:
				curr_book = env.get_book(ts + idx*time_unit)
			actions = sorted(curr_book.a.keys())
			actions.append(0)
			limit_price = float("inf") if t == T else actions[action]

			# execute
			spent, leftover = env.limit_order(0, limit_price, old_leftover)
			if t >= T:
				spent += leftover * actions[-2]
				argmin = 0
				leftover = 0
				states.append(None)
			else:
				left = (1.0 * leftover / self.params['V']) if continuous else int(round(leftover / vol_unit))
				if stateful:
					next_book_vec = create_input_window_stateful(env, window, ob_size, t + 1, left, time_unit)
				else:
					next_book_vec = create_input_window_stateless(env, ts + time_unit, window, ob_size, t + 1, left)
				next_scores, argmin, action = self.predict(sess, next_book_vec)
				states.append(next_book_vec)
			# update state and record value of transaction
			t += 1
			ts = ts + time_unit
			diff = old_leftover - leftover
			price_paid = tgt_price if diff == 0 else spent / diff
			t_cost =  (float(price_paid) - tgt_price)/tgt_price * 100 # the share prices are blown up - this is decimal
			backup += t_cost * diff if i!=0 else 0
			rewards.append([t_cost, diff])
		trades_cost = 0 if (i - leftover) == 0 else (1.0*backup/(i - leftover))
		rewards.append([argmin, leftover])
		return trades_cost, states, rewards

	def predict_train(self, sess, state):
		if self.params['backup'] != 'doubleQ':
			# select action and evaluate value both with target
			scores, argmin, action = sess.run((self.target.predictions, self.target.min_score, self.target.min_action), feed_dict={
											self.target.input_place_holder: state})
		else:
			# use online network to select policy
			_, _, action = sess.run((self.fit.predictions, self.fit.min_score, self.fit.min_action), feed_dict={
											self.fit.input_place_holder: state})
			# use offline network to predict argmin value
			scores, _, _ = sess.run((self.target.predictions, self.target.min_score, self.target.min_action), feed_dict={
											self.target.input_place_holder: state})
			argmin = scores[0][action]
		return scores, argmin, action

	def predict(self, sess, state):
		scores, argmin, action = sess.run((self.fit.predictions, self.fit.min_score, self.fit.min_action), feed_dict={
											self.fit.input_place_holder: state})
		return scores, argmin, action


	def backup_actions(self, sess, inp, actions, costs, next_states):
		targ, _, _ = self.predict(sess, inp)
		for idx in range(len(actions)):
			targ[0][actions[idx]] = self.targ_action_value(sess, inp, costs[idx], next_states[idx])
		return targ

	def targ_action_value(self, sess, inp, cost, next_state):
		if next_state == None:
			return cost
		inp_shares = inp[0][-1][-1]
		leftover_shares = next_state[0][-1][-1]
		diff = inp_shares - leftover_shares
		if inp_shares == 0:
			return 0
		else:
			_, argmin, _ = self.predict_train(sess, next_state)
			return (cost*diff + leftover_shares*argmin)/inp_shares

	def submit_to_batch(self, sess, inp, actions, costs, next_states):
		target = self.backup_actions(sess, inp, actions, costs, next_states)
		## redo batching stuff - done
		self.inp_buff.append(inp)
		self.targ_buff.append(target)
		if self.params['replay'] == True:
			self.replay.append((inp, actions, costs, next_states))
			if len(self.replay) > self.params['replay_size']:
				self.replay.pop(0)
			for idx in range(self.params['replays'] - 1):
				if len(self.inp_buff) == self.params['batch']:
					self.input_batch = np.concatenate(self.inp_buff, axis=0)
					self.targ_batch = np.concatenate(self.targ_buff, axis=0)
					self.inp_buff = []
					self.targ_buff = []
				inp, actions, costs, next_state = random.choice(self.replay)
				self.inp_buff.append(inp)
				self.targ_buff.append(self.backup_actions(sess,inp, actions, costs, next_state))
			if len(self.inp_buff) == self.params['batch']:
				self.input_batch = np.concatenate(self.inp_buff, axis=0)
				self.targ_batch = np.concatenate(self.targ_buff, axis=0)
				self.inp_buff = []
				self.targ_buff = []
		else:
			if len(self.inp_buff) == self.params['batch']:
				self.input_batch = np.concatenate(self.inp_buff, axis=0)
				self.targ_batch = np.concatenate(self.targ_buff, axis=0)
				self.inp_buff = []
				self.targ_buff = []
				

# to do: policy gradients
# class A3C:




def create_input_window_test(env, window, ob_size, t, i):
	vecs = []
	for idx in range(0, window):
		book_vec = env.get_next_state().vectorize_book(ob_size, t, i).reshape(1,1,ob_size * 4 + 2)
		vecs.append(book_vec)
	window_vec = np.concatenate(vecs, axis=1)
	return window_vec


def create_input_window_stateless(env, ts, window, ob_size, t, i):
	if(ts < window - 1):
		return np.zeros(shape=[1, window, ob_size * 4 + 2])
	else:
		vecs = []
		for idx in range(ts - window + 1, ts + 1):
			book_vec = env.get_book(ts).vectorize_book(ob_size, t, i).reshape(1,1,ob_size * 4 + 2)
			vecs.append(book_vec)
		window_vec = np.concatenate(vecs, axis=1)
		return window_vec

def create_input_window_stateful(env, window, ob_size, t, i, time_unit):
	vecs = []
	# wind forward to the window
	for idx in range(0, time_unit - window):
		env.get_next_state()
	for idx in range(0, window):
		book_vec = env.get_next_state().vectorize_book(ob_size, t, i).reshape(1,1,ob_size * 4 + 2)
		vecs.append(book_vec)
	window_vec = np.concatenate(vecs, axis=1)
	return window_vec


def execute_algo(agent, params, session, env, steps):
	H, V, I, T, S = params['H'], params['V'], params['I'], params['T'], params['S']
	divs = 10
	env.get_timesteps(1, S+2, I, V)
	spreads, misbalances, imm_costs, signed_vols = create_variable_divs(divs, env)
	window = agent.params['window']
	ob_size = agent.params['ob_size']
	# remaining volume and list of trades made by algorithm
	executions = []
	volume = V
	# number of shares per unit of inventory
	vol_unit = V/I
	# number of timesteps in between decisions
	time_unit = H/T
	# number of decisions possible during test steps set
	decisions = steps / time_unit - 1
	for x in range(10):
		offset = random.randint(0, time_unit - 1)
		for ts in range(0, decisions+1):
			# update the state of the algorithm based on the current book and timestep
			rounded_unit = 1.0 * volume / vol_unit
			t_left =  ts % (T + 1)
			# regenerate orderbook simulation for the next time horizon of decisions

			if ts % (T+1) == 0:
				env.get_timesteps(ts*time_unit + offset, ts*time_unit+T*time_unit+window+offset+1, T, V)
				volume = V
			if ts % (T + 1) != 0:
				for i in range(0, time_unit - window):
					env.get_next_state()
			input_book = create_input_window_test(env, 	window, ob_size, rounded_unit, t_left)
			curr_book = env.curr_book
			spread = compute_bid_ask_spread(curr_book, spreads)
			volume_misbalance = compute_volume_misbalance(curr_book, misbalances, env)
			immediate_cost = compute_imm_cost(curr_book, volume, imm_costs)
			signed_vol = compute_signed_vol(env.running_vol, signed_vols)
			# ideal price is mid-spread end of the period
			perfect_price = env.mid_spread(ts*time_unit + time_unit * (T- t_left))
			actions = sorted(curr_book.a.keys())
			actions.append(0)
			# compute and execute the next action using the table
			scores, argmin, min_action = agent.predict(session, input_book)
			paid, leftover = env.limit_order(0, actions[min_action], volume)
			print min_action
			# if we are at the last time step, have to submit everything remaining to OB
			if t_left == T:
				additional_spent, overflow = env.limit_order(0, float("inf"), leftover)
				paid += overflow * actions[-2] + additional_spent
				leftover = 0
			if leftover == volume:
				reward = [t_left, rounded_unit, spread, volume_misbalance, signed_vol, min_action, 'no trade ', 0]
			else:
				price_paid = paid / (volume - leftover)
				basis_p = (float(price_paid) - perfect_price)/perfect_price * 100
				reward = [t_left, rounded_unit, spread, volume_misbalance, immediate_cost, signed_vol, min_action, basis_p, volume - leftover]
				print str(perfect_price) + ' ' + str(price_paid)
			executions.append(reward)
			volume = leftover
	return executions

def write_trades(executions, tradesOutputFilename="DQN"):
	trade_file = open(tradesOutputFilename + '.csv', 'wb')
	# write trades executed
	w = csv.writer(trade_file)
	executions.insert(0, ['Time Left', 'Rounded Units Left', 'Bid Ask Spread', 'Volume Misbalance', 'Immediate Cost', 'Signed Transcation Volume' ,'Action', 'Reward', 'Volume'])
	w.writerows(executions)

def run_sampling_DQN(sess, envs, agent, params):
	ob_size = params['ob_size']
	window = params['window']
	H = params['H']
	V = params['V']
	I = params['I']
	T = params['T']
	L = params['L']
	S = params['S']
	length = params['length']
	epsilon = 0.2
	losses = []
	vol_unit = V / I
	time_unit = H / T
	state = {}
	agent.choose_backup_networks()
	averages = []
	diffs = []
	costs = []
	for ts in range(11, S+11):
		if isinstance(envs, dict):
			days = envs.keys()
			day = random.choice(days)
			env = envs[day]
		else:
			env = envs
		order_books = len(env.books)
		sample = random.randint(0, order_books - (H + 1))
		i = V
		t = 0
		curr_state = create_input_window_stateless(env, sample, window, ob_size, t, (1.0 * i)/ V)	
		env.get_timesteps(sample, sample + time_unit * T + 1, I, V)
		while t <= T:
			scores, argmin, action = agent.predict(sess, curr_state)
			if np.random.rand(1) < epsilon:
				a = np.array([random.randint(0, params['L'])])[0]
			else:
				a = action
			state['inv'] = i
			state['t'] = t
			state['ts'] = sample + t *time_unit
			cost, states, rewards = agent.calculate_target(sess, env, state, a, length, reset=False)
			argmin, leftover = rewards[-1]

			costs.append(cost)
			inp = curr_state
			agent.submit_to_batch(sess, inp, [a], [cost], [states[-1]])
			t = t + length
			i = rewards[-1][1]
			curr_state = states[-1]
			if agent.batch_ready():
				averages.append(np.mean(costs))
				costs = []
				b_in = agent.input_batch
				b_targ = agent.targ_batch
				q_vals, loss, min_score = agent.update_networks(sess)
				diffs.append(loss)
				agent.choose_backup_networks()
				#losses.append([q_vals, loss, min_score, b_in, b_targ])
				#print_stuff(agent, q_vals, loss, b_in, b_targ)
				if len(diffs) == 100:
					print 'TS %d'.format(ts)
					print np.mean(diffs)
					print np.mean(averages)
					diffs = []
					averages = []
	print 'Epoch Over'

def run_dp(sess, envs, agent, params):
	ob_size = params['ob_size']
	window = params['window']
	H = params['H']
	V = params['V']
	I = params['I']
	T = params['T']
	L = params['L']
	S = params['S']
	length = params['length']
	losses = []
	diffs = []
	vol_unit = V / I
	time_unit = H / T
	agent.choose_backup_networks()

	for ts in range(1, S+1):
		if isinstance(envs, dict):
			days = envs.keys()
			day = random.choice(days)
			env = envs[day]
		else:
			env = envs
		order_books = len(env.books)
		sample = random.randint(0, order_books - (H + 1))
		for t in range(T+1)[::-1]:
			for i in range(I+1):
				t_costs = np.zeros(shape=[L+1])
				next_states = []
				for a in range(L+1):
					state = {}
					state['inv'] = i * vol_unit
					state['t'] = t
					state['ts'] = sample 
					costs, states, rewards = agent.calculate_target(sess, env, state, a, length)
					argmin, leftover = rewards[-1]
					t_costs[a] = costs
					next_states.append(states[-1])

				inp = states[0]
				agent.submit_to_batch(sess, inp, range(L+1), t_costs, next_states)
				if agent.batch_ready():
					b_in = agent.input_batch
					b_targ = agent.targ_batch
					q_vals, loss, min_score = agent.update_networks(sess)
					agent.choose_backup_networks()
					diffs.append(loss)
					#losses.append([q_vals, loss, min_score, b_in, b_targ])
					# print_stuff(agent, q_vals, loss, b_in, b_targ)
					if len(diffs) == 10:
						print 'TS %d'.format(ts)
						print np.mean(diffs)
						diffs = []
	print 'Epoch Over'

def print_stuff(agent, q_vals, loss, inputs, targets):
	#print 'inputs'
	print inputs[:,:,-2:]
	#print 'Q'
	print q_vals[-1]
	#print 'targets'
	print targets[-1]
	#print 'loss'
	print np.mean(loss)



def train_DQN_sampling(epochs, ob_file, params, test_steps, test_file_name='DQN-trades', envs=None, test_env=None):
	if envs is None:
		envs = Environment(ob_file,setup=False)
	agent = Q_Approx(params)
	init = tf.initialize_all_variables()
	with tf.Session() as sess:
		sess.run(init)
		if params['backup'] == 'sampling':	
			sess.run(agent.updateTargetOperation)
		for i in range(epochs):
			run_sampling_DQN(sess, envs, agent, params)
		if test_env is None:
			test_env = envs
		executions = execute_algo(agent, params, sess, test_env, test_steps)
		write_trades(executions, tradesOutputFilename=test_file_name)

def train_DQN_DP(epochs, ob_file, params, test_steps, test_file_name='DQN-trades', envs=None, test_env=None):
	if envs is None:
		envs = Environment(ob_file,setup=False)

	agent = Q_Approx(params) 
	init = tf.initialize_all_variables()
	with tf.Session() as sess:
		sess.run(init)
		if params['backup'] == 'sampling':
			sess.run(agent.updateTargetOperation)
		for i in range(epochs):
			run_dp(sess, envs, agent, params)
		if test_env is None:
			test_env = envs
		executions = execute_algo(agent, params, sess, test_env, test_steps)
		write_trades(executions, tradesOutputFilename=test_file_name)


def train_DQN_DP_warmup(epochs, ob_file, params, test_steps, test_file_name='DQN-trades', envs=None, test_env=None):
	if envs is None:
		envs = Environment(ob_file,setup=False)
	S = params['S']
	agent = Q_Approx(params) 
	init = tf.initialize_all_variables()
	with tf.Session() as sess:
		sess.run(init)
		if params['backup'] == 'sampling':
			sess.run(agent.updateTargetOperation)
		for i in range(epochs):
			params['S'] = params['w_S']
			run_dp(sess, envs, agent, params)
			params['S'] = S
			run_sampling_DQN(sess, envs, agent, params)
		if test_env is None:
			test_env = envs
		executions = execute_algo(agent, params, sess, test_env, test_steps)
		write_trades(executions, tradesOutputFilename=test_file_name)


if __name__ == "__main__":
	
	file = '../data/10_GOOG.csv'
	params = {
		'backup': 'sampling',
		'network': 'CNN',
		'advantage': True,
		'replay': True,
		'replay_size': 10,
		'replays': 10,
		'window': 10,
		'ob_size': 10,
		'hidden_size': 10, 
		'depth': 2, 
		'actions': 11, 
		'batch': 10,
		'continuous': True,
		'stateful': True,
		'length': 1,
		'H': 10000, 
		'V': 10000,
		'T': 10,
		'I': 10,
		'T': 10,
		'L': 10,
		'w_S': 1,
		'S': 1
	}
	layers = {
		'A-conv1': {
			'type': 'conv',
			'size': 2,
			'stride': 1,
			'num': 20
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
			'num': 30
		}
	}
	params['layers'] = layers

	train_DQN_DP(1, file, params, 100000)
