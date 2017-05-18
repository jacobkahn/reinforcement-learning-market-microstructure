from environment import *
from agent import *
from q_learners import *
from collections import defaultdict
from random import shuffle
import tensorflow as tf 
import numpy as np
import csv


class Params:
	def __init__(self, window, ob_size, hidden_size, depth, actions, batch):
		self.window = window
		self.ob_size = ob_size
		self.hidden_size = hidden_size
		self.hidden_depth = depth
		self.actions = actions
		self.batch = batch

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
		if b == 'replay_buffer' or b == 'sampling':
			self.Q = self.create_network(a, params, 'Q')
			self.Q_target = self.create_network(a, params, 'Q_target')
			self.buff = []
			self.updateTargetOperation = self.Q_target.copy_Q_Op(self.Q)
		elif b == 'dueling net':
			self.Q = self.create_network(a, params, 'Q')
			self.V = self.create_network(a, params, 'V')
		elif b == 'doubleQ':
			self.Q_1 = self.create_network(a, params, 'Q_1')
			self.Q_2 = self.create_network(a, params, 'Q_2')
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

	def backup(self, sess, inp, targ):
		b = self.params['backup']
		if b == 'sampling':
			print 'hi'
		elif b == 'replay_buffer':
			self.Q = self.create_network(a, params, 'Q')
			self.Q_target = self.create_network(a, params, 'Q_target')
			self.updateTargetOperation = Q_target.copy_Q_Op(Q)
		elif b == 'dueling net':
			self.Q = self.create_network(a, params, 'Q')
			self.V = self.create_network(a, params, 'V')
		elif b == 'doubleQ':
			self.Q_1 = self.create_network(a, params, 'Q_1')
			self.Q_2 = self.create_network(a, params, 'Q_2')


	def choose_backup_networks(self):
		b = self.params['backup']
		if b == 'replay_buffer' or b == 'sampling':
			self.target = self.Q_target
			self.fit = self.Q
		elif b == 'dueling net':
			print 'hi'
		elif b == 'doubleQ':
			Q_tables = [self.Q_1, self.Q_2]	
			shuffle(Q_tables)
			self.target = Q_tables[0]
			self.fit = Q_tables[1]


	def batch_ready(self):
		return ((self.input_batch != None) and (self.targ_batch != None))

	def update_networks(self, sess):
		Q = self.fit
		inp = self.input_batch
		targ = self.targ_batch
		q_vals, loss, min_score = sess.run(
				(Q.predictions, Q.loss, Q.min_score), 
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
		self.counter +=1 

		i = state['inv']
		t = state['t']
		ts = state['ts']
		if self.counter % 100 == 0 and b == 'sampling':
			sess.run(self.updateTargetOperation)
		
		backup = 0
		curr_action = action
		tgt_price = env.mid_spread(ts + time_unit * (self.params['T']- t))
		leftover = i

		states = []
		rewards = []
		trades_cost = 0 

		if reset:
			if params['stateful']:
				env.get_timesteps(ts, ts + time_unit*length+ 1, self.params['I'], self.params['V'])
			left = (1.0 * leftover / self.params['V']) if continuous else int(round(leftover / vol_unit))
			s = create_input_window_stateless(env, ts, window, ob_size, t, left)
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
			num_shares = leftover
			spent, leftover = env.limit_order(0, limit_price, num_shares)
			if t >= T:
				spent += leftover * actions[-2]
				argmin = 0
				leftover = 0
			else:
				left = (1.0 * leftover / self.params['V']) if continuous else int(round(leftover / vol_unit))
				if stateful:
					next_book_vec = create_input_window_stateful(env, window, ob_size, t + 1, left, time_unit)
				else: 
					next_book_vec = create_input_window_stateless(env, ts + time_unit, window, ob_size, t + 1, left)
				if b == 'doubleQ':
					action = sess.run((use_Q.min_action), feed_dict={
											use_Q.input_place_holder: next_book_vec
								  })
					next_scores = sess.run((self.target.predictions), feed_dict={
											use_Q.input_place_holder: next_book_vec
								  })
					argmin = next_scores[0][action]
				else: 
					next_scores, argmin, action = sess.run((use_Q.predictions, use_Q.min_score, use_Q.min_action), feed_dict={
											use_Q.input_place_holder: next_book_vec
								  })
				states.append(next_book_vec)
			# update state and record value of transaction
			t += 1
			ts = ts + time_unit
			diff = old_leftover - leftover
			price_paid = tgt_price if diff == 0 else spent / diff
			t_cost =  (float(price_paid) - tgt_price)/tgt_price * 100
			backup += t_cost * diff if i!=0 else 0
			rewards.append([t_cost, diff])
		trades_cost = 0 if (i - leftover) == 0 else (1.0*backup/(i - leftover))
		backup += leftover * argmin
		backup = 0 if i == 0 else (1.0* backup)/ (i)
		rewards.append([argmin, leftover])
		import pdb
		return backup, states, rewards, trades_cost

	def predict(self, sess, state):
		if self.params['backup'] != 'doubleQ':
			scores, argmin, action = sess.run((self.target.predictions, self.target.min_score, self.target.min_action), feed_dict={
											self.target.input_place_holder: state})
		else: 
			scores, argmin, action = sess.run((self.target.predictions, self.target.min_score, self.target.min_action), feed_dict={
											self.target.input_place_holder: state})

			scores_1, argmin, action = sess.run((self.fit.predictions, self.fit.min_score, self.fit.min_action), feed_dict={
											self.fit.input_place_holder: state})
			scores = (scores + scores_1) / 2
			argmin = np.amin(scores)
			action = np.argmin(np.squeeze(scores))
		return scores, argmin, action

	def submit_to_batch(self, inp, target):
		self.inp_buff.append(inp)
		self.targ_buff.append(target)
		if self.params['replay'] == True:
			self.replay.append((inp, target))
			if len(self.replay) > self.params['replay_size']:
				self.replay.pop(0)
			for idx in range(self.params['replays'] - 1):
				if len(self.inp_buff) == self.params['batch']:
					self.input_batch = np.concatenate(self.inp_buff, axis=0)
					self.targ_batch = np.concatenate(self.targ_buff, axis=0)
					self.inp_buff = []
					self.targ_buff = []
				inp, target = random.choice(self.replay)
				self.inp_buff.append(inp)
				self.targ_buff.append(target)
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
				return True
		return False

# to do: policy gradients
# class A3C:


def compute_pool_size(b, h, w, psize, stride, k):
	W_2 = (w - psize)/stride + 1
	H_2 = (h - psize)/stride + 1
	return [b, H_2, W_2, k]

def compute_output_size(b, h, w, fsize, stride, padding, k):
	W_2 = (w - fsize + 2 * padding)/ stride + 1
	H_2 = (h - fsize + 2 * padding)/ stride + 1
	return [b, H_2, W_2, k]


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
	for i in range(0, time_unit - window):
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
	decisions = steps / time_unit

	for ts in range(0, decisions+1):
		# update the state of the algorithm based on the current book and timestep
		rounded_unit = int(volume / vol_unit)
		t_left =  ts % (T + 1)
		# regenerate orderbook simulation for the next time horizon of decisions

		if ts % (T+1) == 0:
			env.get_timesteps(ts*time_unit, ts*time_unit+T*time_unit+window+1, T, V)
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
			reward = [t_left, rounded_unit, spread, volume_misbalance, min_action, 'no trade ', 0]
		else:
			price_paid = paid / (volume - leftover)
			basis_p = (float(price_paid) - perfect_price)/perfect_price * 100
			reward = [t_left, rounded_unit, spread, volume_misbalance, immediate_cost, signed_vol, min_action, basis_p, volume - leftover]
			print str(perfect_price) + ' ' + str(price_paid)
		executions.append(reward)
		volume = leftover
	return executions

def write_trades(executions, tradesOutputFilename="DQN"):
	trade_file = open(tradesOutputFilename + '-trades.csv', 'wb')
	# write trades executed
	w = csv.writer(trade_file)
	executions.insert(0, ['Time Left', 'Rounded Units Left', 'Bid Ask Spread', 'Volume Misbalance', 'Immediate Cost', 'Signed Transcation Volume' ,'Action', 'Reward', 'Volume'])
	w.writerows(executions)

def run_sampling_DQN(sess, env, agent, params):
	ob_size = params['ob_size']
	window = params['window']
	H = params['H']
	V = params['V']
	I = params['I']	
	T = params['T']
	L = params['L']
	S = params['S']
	length = params['length']
	e = 0.5
	losses = []
	vol_unit = V / I
	time_unit = H / T
	state = {}
	agent.choose_backup_networks()
	order_books = len(env.books)
	averages = []
	costs = []
	for ts in range(11, S+11):
		e = e / ts * 100
		sample = random.randint(0, order_books - (H + 1))
		i = V
		t = 0
		curr_state = create_input_window_stateless(env, sample, window, ob_size, t, i)	
		env.get_timesteps(sample, sample + time_unit * T + 1, I, V)
		while t <= T:
			scores, argmin, action = agent.predict(sess, curr_state)
			if np.random.rand(1) < e:
				a = np.array([random.randint(0, params['L'])])[0]
			else:
				a = action
			state['inv'] = i
			state['t'] = t
			state['ts'] = sample 
			import pdb
			backup, states, rewards, cost = agent.calculate_target(sess, env, state, a, length, reset=False)
			costs.append(cost)
			scores[0][a] = backup
			inp = states[0]
			targ = scores.reshape(1, L + 1)
			agent.submit_to_batch(inp, targ)
			t = t + length
			i = rewards[-1][1]
			curr_state = states[-1]
			if agent.batch_ready():
				averages.append(np.mean(costs))
				#print np.mean(costs)
				costs = []
				b_in = agent.input_batch
				b_targ = agent.targ_batch
				q_vals, loss, min_score = agent.update_networks(sess)
				agent.choose_backup_networks()
				losses.append([q_vals, loss, min_score, b_in, b_targ])
				print_stuff(agent, q_vals, loss, b_in, b_targ)
				if len(averages) == 100:
					#print 'average reward of last 100 batches: {}'.format(np.mean(averages))
					averages = []
		#if ts % 1000 == 0:
			#print ts
	print 'Epoch Over'

def run_dp(sess, env, agent, params):
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
	averages = []
	costs = []
	vol_unit = V / I
	time_unit = H / T
	agent.choose_backup_networks()
	order_books = len(env.books)
	for t in range(T+1)[::-1]:
		for i in range(I+1):
			for ts in range(1, S+1):
				sample = random.randint(0, order_books - (H + 1))
				backup = np.zeros(shape=[L+1])
				argmins = np.zeros(shape=[L+1])
				t_costs = []
				for a in range(L+1):
					state = {}
					state['inv'] = i * vol_unit
					state['t'] = t
					state['ts'] = sample 
					backup[a], states, t_cost, cost = agent.calculate_target(sess, env, state, a, length)
					t_costs.append(t_cost)
				inp = states[0]
				targ = backup.reshape(1, L + 1)
				agent.submit_to_batch(inp, targ)
				if agent.batch_ready():
					averages.append(np.mean(costs))
					#print np.mean(costs)
					costs = []
					#print '{},{}'.format(t, i)
					b_in = agent.input_batch
					b_targ = agent.targ_batch
					q_vals, loss, min_score = agent.update_networks(sess)
					agent.choose_backup_networks()
					losses.append([q_vals, loss, min_score, b_in, b_targ])
					print_stuff(q_vals, loss, b_in, b_targ)
					if len(averages) == 100:
						#xprint 'average reward of last 100 batches: {}'.format(np.mean(averages))
						averages = []
	print 'Epoch Over'

def print_stuff(agent, q_vals, loss, inputs, targets):
	#print 'inputs'
	#print inputs[0][0][-2:]
	#print 'Q'
	#print q_vals[-1]
	#print 'targets'
	#print targets[-1]
	#print 'loss'
	print np.mean(loss)


def train_DQN_DP(epochs, ob_file, params, test_steps, env=None):
	if env is None:
		env = Environment(ob_file,setup=False)
	layers = {
		'conv1': {
			'type': 'conv',
			'size': 2,
			'stride': 1,	
			'num': 10
		},
		'pool1': {
			'type': 'pool',
			'stride': 2,
			'size': 5,
			'pool_type': 'max'
		},
		'conv2': {
			'type': 'conv',
			'size': 3,
			'stride': 2,	
			'num': 10
		}
	}
	params['layers'] = layers

	agent = Q_Approx(params) 
	init = tf.initialize_all_variables()
	with tf.Session() as sess:
		sess.run(init)
		if params['backup'] == 'sampling':
			sess.run(agent.updateTargetOperation)
		for i in range(epochs):
			run_sampling_DQN(sess, env, agent, params)
		executions = execute_algo(agent, params, sess, env, test_steps)
		write_trades(executions)


	

	
if __name__ == "__main__":
	params = {
		'backup': 'sampling',
		'network': 'CNN',
		'advantage': True,
		'replay': True,
		'replay_size': 1000,
		'replays': 20,
		'window': 100,
		'ob_size': 10,
		'hidden_size': 10, 
		'depth': 2, 
		'actions': 11, 
		'batch': 100,
		'continuous': True,
		'stateful': True,
		'length': 11,
		'H': 10000, 
		'V': 100,
		'T': 10,
		'I': 10,
		'T': 10,
		'L': 10,
		'S': 2000
	}
	train_DQN_DP(3, '../data/10_GOOG.csv', params, 100000)

	