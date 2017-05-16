from environment import *
from agent import *
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

class Filters:
	def __init__(self, filters):
		self.f = filters




class Q_Approx:
	def __init__(self, params):
		self.params = params
		b = params['backup']
		a = params['network']
		self.inp_buff = []
		self.targ_buff = []
		self.batchs = []
		self.input_batch = None
		self.targ_batch = None
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
		q_vals, loss, min_score, gradients = sess.run(
				(Q.predictions, Q.loss, Q.min_score, Q.gvs), 
				feed_dict={Q.input_place_holder: inp, Q.target_values: targ})
		self.input_batch = None
		self.targ_batch = None
		return q_vals, loss, min_score, gradients		

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
		if ts % 100 == 0 and b == 'sampling':
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
					argmin = next_scores[action]
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
		return backup, states, rewards, trades_cost

	def predict(self, sess, state):
		scores, argmin, action = sess.run((self.target.predictions, self.target.min_score, self.target.min_action), feed_dict={
											self.target.input_place_holder: state})
		return scores, argmin, action

	def submit_to_batch(self, inp, target):
		self.inp_buff.append(inp)
		self.targ_buff.append(target)
		if self.params['backup'] == 'replay_buffer':
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


def compute_pool_size(h, w, psize, stride, k):
	W_2 = (w - psize)/stride + 1
	H_2 = (h - psize)/stride + 1
	return [W_2, H_2, key]

def compute_outputsize(h, w, fsize, stride, padding, k):
	W_2 = (w - fsize + 2 * padding)/ stride + 1
	H_2 = (h - fsize + 2 * padding)/ stride + 1
	return [W_2, H_2, k]

class Q_CNN: 

	def __init__(self, params, name):
		self.name = name
		self.params = Params(params['window'], params['ob_size'], params['hidden_size'], params['depth'], params['actions'], params['batch'])
		self.layers = params['layers']
		self.build_model_graph()

	def build_model_graph(self):
		self.filter_tensors = {}
		self.bias_tensors = {}
		self.conv_layer_out
		# lots to decisions
		with tf.variable_scope(self.name) as self.scope:
			self.input_place_holder = tf.placeholder(tf.float32, shape=(self.params.batch, self.params.window, self.params.ob_size * 4 + 2, 1), name='input')
			curr_dimension = [self.params.batch, self.params.window, self.params.ob_size * 4 + 2, 1]
			for name, layer_params in self.layers.items():
				if layer['type'] == 'conv':
					n = 'conv_{}_filter_size_{}_stride_{}_num_{}'.format(name, layer_params['size'], layer_params['stride'], layer_params['num'])
					s = [layer_params['size'], layer_params['size'], 1, layer_params['num']]
					o_s = compute_outputsize(self.params.window, self.params.ob_size * 4 + 2,layer_params['size'], layer_params['stride'], 0, layer_params['num'])
					strides = [1, layer_params['stride'], layer_params['stride'], 1]
					self.filter_tensors[name] = tf.Variable(tf.truncated_normal(s, stddev=0.1), name=n)
					self.bias_tensors[name] = tf.Variable(tf.truncated_normal(shape=[params['num']], stddev=0.1), name=n + '_bias')
					conv_output = tf.nn.conv2d(self.input_place_holder, self.filter_tensors[name], strides, "VALID")
					#conv_bias = tf.nn.bias_add(conv_output, )
				#if layer['type'] == ['pool']:
				#for conv in conv_layer_out:
				#	out = tf.nn.max_pool(conv, [self.params.batch, p['size'], p['size'],1], [1, p['stride'], p['stride'], 1], 'VALID')
				#	pool_out.append(out)
				#self.final_layer = tf.squeeze(tf.concatenate(pool_out, 3)) 

	def add_training_objective(self):
		self.target_values = tf.placeholder(tf.float32, shape=[self.params.batch, self.params.actions], name='target')
		self.batch_losses = tf.reduce_sum(tf.sqrt(tf.squared_difference(self.predictions, self.target_values)), axis=1)
		self.loss = tf.reduce_sum(self.batch_losses, axis=0) + tf.nn.l2_loss(self.U) + tf.nn.l2_loss(self.b_2)
		self.trainer = tf.train.AdamOptimizer(learning_rate=0.0001)
		self.updateWeights = self.trainer.minimize(self.loss)

	def copy_Q_Op(self, Q):
		current_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope.name)
		target_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=Q.scope.name)
		op_holder =[]
		for var, target_val in zip(sorted(current_variables, key=lambda v: v.name),
                               sorted(target_variables, key=lambda v: v.name)):
			op_holder.append(var.assign(target_val))
		copy_operation = tf.group(*op_holder)
		return copy_operation


class Q_RNN: 

	def __init__(self, params, name):
		self.name = name
		self.params = Params(params['window'], params['ob_size'], params['hidden_size'], params['depth'], params['actions'], params['batch'])
		self.build_model_graph()	
		self.add_training_objective()

	def build_model_graph(self): 
		with tf.variable_scope(self.name) as self.scope:
			self.input_place_holder = tf.placeholder(tf.float32, shape=(None, self.params.window, self.params.ob_size * 4 + 2), name='input')
			self.forward_cell_layers = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.LSTMCell(self.params.hidden_size) for i in range(self.params.hidden_depth)])
			self.rnn_output, self.final_rnn_state = tf.nn.dynamic_rnn(self.forward_cell_layers, self.input_place_holder, dtype=tf.float32)
			self.outs = tf.squeeze(tf.slice(self.rnn_output, [0, self.params.window - 1, 0], [-1, 1, self.params.hidden_size]), axis=1)
			self.U = tf.get_variable('U', shape=[self.params.hidden_size, self.params.actions])
			self.b_2 = tf.get_variable('b2', shape=[self.params.actions])
			self.predictions = tf.cast((tf.matmul(self.outs, self.U) + self.b_2), 'float32') 
			self.min_score = tf.reduce_min(self.predictions, reduction_indices=[1])
			self.min_action = tf.argmin(tf.squeeze(self.predictions), axis=0, name="arg_min")

	def add_training_objective(self):
		self.target_values = tf.placeholder(tf.float32, shape=[self.params.batch, self.params.actions], name='target')
		self.batch_losses = tf.reduce_sum(tf.squared_difference(self.predictions, self.target_values), axis=1)
		self.loss = tf.reduce_sum(self.batch_losses, axis=0)
		self.trainer = tf.train.AdamOptimizer(learning_rate=0.0001)
		self.gvs, self.variables = zip(*self.trainer.compute_gradients(self.loss))
		self.clipped_gradients, _ = tf.clip_by_global_norm(self.gvs, 5.0)
		self.updateWeights = self.trainer.apply_gradients(zip(self.clipped_gradients, self.variables))

	def copy_Q_Op(self, Q):
		current_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope.name)
		target_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=Q.scope.name)
		op_holder =[]
		for var, target_val in zip(sorted(current_variables, key=lambda v: v.name),
                               sorted(target_variables, key=lambda v: v.name)):
			op_holder.append(var.assign(target_val))
		copy_operation = tf.group(*op_holder)
		return copy_operation
	
	def greedy_action(self, session, book_vec):
		min_action = session.run((self.min_action), feed_dict={ self.input_place_holder: book_vec})
		return min_action


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



def execute_algo(agent, session, env, H, V, I, T, steps, S):
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
		sample = random.randint(0, order_books - (H + 1))
		i = V
		t = 0
		curr_state = create_input_window_stateless(env, sample, window, ob_size, t, i)	
		env.get_timesteps(sample, sample + time_unit * T + 1, I, V)
		while t < T:
			scores, argmin, action = agent.predict(sess, curr_state)
			if np.random.rand(1) < e:
				a = np.array([random.randint(0, params['L'])])[0]
			else:
				a = action
			state['inv'] = i
			state['t'] = t
			state['ts'] = sample 
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
				averages.append(np.means(costs))
				#print np.mean(costs)
				costs = []
				b_in = agent.input_batch
				b_targ = agent.targ_batch
				q_vals, loss, min_score, gradients = agent.update_networks(sess)
				agent.choose_backup_networks()
				losses.append([q_vals, loss, min_score, gradients, b_in, b_targ])
				#print_stuff(q_vals, loss, b_in, b_targ)
				if len(averages) == 10:
					print 'average reward of last 10 batches: {}'.format(np.mean(averages))
					averages = []
		if ts % 1000 == 0:
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
					backup[a], states, t_cost = agent.calculate_target(sess, env, state, a, length)
					t_costs.append(t_cost)
				inp = states[0]
				targ = backup.reshape(1, L + 1)
				agent.submit_to_batch(inp, targ)
				if agent.batch_ready():
					print '{},{}'.format(t, i)
					b_in = agent.input_batch
					b_targ = agent.targ_batch
					q_vals, loss, min_score, gradients = agent.update_networks(sess)
					agent.choose_backup_networks()
					losses.append([q_vals, loss, min_score, gradients, b_in, b_targ])
					print_stuff(q_vals, loss, b_in, b_targ)
				if ts % 1000 == 0:
					print ts
	print 'Epoch Over'

def print_stuff(q_vals, loss, inputs, targets):
	#print 'inputs'
	#print inputs[0][0][-2:]
	#print 'Q'
	#print q_vals
	#print 'targets'
	#print targets
	#print 'loss'
	print np.mean(loss)


def train_DQN_DP(epochs, ob_file, params, test_steps, env=None):
	if env is None:
		env = Environment(ob_file,setup=False)
	filters = {
		'filt1': {
			'size': 2,
			'stride': 2,	
			'num': 10
		},
		'pool': {
			'stride': 4,
			'size': 5
		}
	}
	agent = Q_Approx(params) 
	init = tf.initialize_all_variables()
	with tf.Session() as sess:
		sess.run(init)
		if params['backup'] == 'sampling':
			sess.run(agent.updateTargetOperation)
		for i in range(epochs):
			run_sampling_DQN(sess, env, agent, params)
		executions = execute_algo(sess, env, agent, test_steps, params['S'])
		write_trades(executions)


	

	
if __name__ == "__main__":
	params = {
		'backup': 'sampling',
		'network': 'RNN',
		'window': 10,
		'ob_size': 10,
		'hidden_size': 40, 
		'depth': 3, 
		'actions': 11, 
		'batch': 1000,
		'continuous': True,
		'stateful': True,
		'rollout': 1,
		'length': 1,
		'H': 1000, 
		'V': 1000,
		'T': 10,
		'I': 10,
		'T': 10,
		'L': 10,
		'S': 10000
	}
	train_DQN_DP(1, '../data/10_GOOG.csv', params, 100000)

	