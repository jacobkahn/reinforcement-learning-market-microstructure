from environment import *
from agent import *
from collections import defaultdict
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

def compute_pool_size(h, w, psize, stride, k):
	W_2 = (w - psize)/stride + 1
	H_2 = (h - psize)/stride + 1
	return [W_2, H_2, key]

def compute_outputsize(h, w, fsize, stride, padding, k):
	W_2 = (w - fsize + 2 * padding)/ stride + 1
	H_2 = (h - fsize + 2 * padding)/ stride + 1
	return [W_2, H_2, k]


class Q_CNN: 

	def __init__(self, params, name, filters, target=False):
		self.name = name
		self.params = params
		self.f = filters
		self.pool = filters['pool']
		self.build_model_graph()
		self.add_training_objective()

	def build_model_graph(self):
		self.filter_tensors = {}
		self.bias_tensors = {}
		self.conv_layer_out = []
		self.pool_out = []
		p = self.pool
		with tf.variable_scope(self.name) as self.scope:
			self.input_place_holder = tf.placeholder(tf.float32, shape=(self.params.batch, self.params.window, self.params.ob_size * 4 + 2, 1), name='input')
			for fil, params in self.f.items():
				n = 'filter_size_{}_stride_{}_num_{}'.format(params['size'], params['stride'], params['num'])
				s = [params['size'], params['size'], 1, params['num']]
				o_s = compute_outputsize(self.params.window, self.params.ob_size * 4 + 2,params['size'], params['stride'], 0, params['num'])
				self.filter_tensors[name] = tf.Variable(tf.truncated_normal(s, stddev=0.1), name=n)
				self.bias_tensors[name] = tf.Variable(tf.truncated_normal(shape=[params['num']], stddev=0.1), name=n + '_bias')
				conv_output = tf.nn.conv2d(self.input_place_holder, self.filter_tensors[name], params, "VALID")
				h = tf.nn.relu(tf.nn.bias_add(conv_output, bias), name=n+'relu')
				conv_layer_out.append(h)
			for conv in conv_layer_out:
				out = tf.nn.max_pool(conv, [self.params.batch, p['size'], p['size'],1], [1, p['stride'], p['stride'], 1], 'VALID')
				pool_out.append(out)
			self.final_layer = tf.squeeze(tf.concatenate(pool_out, 3)) 

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

	def __init__(self, params, name, target=False):
		self.name = name
		self.params = params
		self.build_model_graph()
		if not target:
			self.add_training_objective()

	def build_model_graph(self): 
		with tf.variable_scope(self.name) as self.scope:
			self.input_place_holder = tf.placeholder(tf.float32, shape=(self.params.batch, self.params.window, self.params.ob_size * 4 + 2), name='input')
			self.forward_cell_layers = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.LSTMCell(self.params.hidden_size) for i in range(self.params.hidden_depth)])
			self.rnn_output, self.final_rnn_state = tf.nn.dynamic_rnn(self.forward_cell_layers, self.input_place_holder, \
								sequence_length=[self.params.window]*self.params.batch, dtype=tf.float32)
			self.outs = tf.squeeze(tf.slice(self.rnn_output, [0, self.params.window - 1, 0], [self.params.batch, 1, self.params.hidden_size]), axis=1)
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


def create_input_window_train(env, ts, window, batch, ob_size, t, i):
	if(ts < window - 1):
		return np.zeros(shape=[batch, window, ob_size * 4 + 2])
	else:
		vecs = []
		for idx in range(ts - window + 1, ts + 1):
			book_vec = env.get_book(ts).vectorize_book(ob_size, t, i).reshape(1,1,ob_size * 4 + 2)
			vecs.append(book_vec)
		window_vec = np.concatenate(vecs, axis=1)
		return window_vec


def execute_algo(Q, session, env, H, V, I, T, steps, S):
	divs = 10
	env.get_timesteps(0, S+1, I, V)
	spreads, misbalances, imm_costs, signed_vols = create_variable_divs(divs, env)
	window = Q.params.window
	ob_size = Q.params.ob_size
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
		min_action = Q.greedy_action(session, input_book)
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


def run_continuos_epoch(sess, env, Q, Q_target, updateTargetOperation, H, V, I, T, L, S):
	ob_size = Q.params.ob_size
	window = Q.params.window
	losses = []
	vol_unit = V / I
	time_unit = H / T
	for t in range(T+1)[::-1]:
		for i in range(I+1):
			for ts in range(1, S+1):
				curr_book = env.get_book(ts)
				tgt_price = env.mid_spread(ts)
				actions = sorted(curr_book.a.keys())
				actions.append(0)
				backup = np.zeros(shape=[L+1])
				argmins = np.zeros(shape=[L+1])
				t_costs = np.zeros(shape=[L+1])
				for a in range(L+1):
					curr_book = env.get_book(ts)
					limit_price = float("inf") if t == T else actions[a]
					spent, leftover = env.limit_order(0, limit_price, vol_unit * i)
					if t == T:
						spent += leftover * actions[-2]
						argmin = 0
						leftover = 0
					else: 
						left = (1.0 * leftover / H)
						next_book_vec = create_input_window_train(env, ts + time_unit, window, 1, ob_size, t + 1, left)
						next_scores, argmin = sess.run((Q_target.predictions, Q_target.min_score), feed_dict={
														Q_target.input_place_holder: next_book_vec
													})
					diff = vol_unit * i - leftover
					price_paid = tgt_price if diff == 0 else spent / diff
					t_cost =  (float(price_paid) - tgt_price)/tgt_price * 100
					backup[a] = (t_cost * diff + argmin * leftover)/(vol_unit * i) if i!=0 else 0
					if np.isinf(argmin).any() or np.isinf(argmin).any():
						import pdb
						pbd.set_trace()
					argmins[a] = argmin
					t_costs[a] = t_cost
				targ = backup.reshape(1, 11)
				book_vec = create_input_window_train(env, ts, window, 1, ob_size, t, (1.0 * i * vol_unit) / H)
				q_vals, loss, min_score, gradients = sess.run((Q.predictions, Q.loss, Q.min_score, Q.gvs), feed_dict={Q.input_place_holder: book_vec, Q.target_values: targ})	
				gradient_nan = False
				for g in gradients:
					if np.isnan(g).any() or np.isinf(g).any():
						gradient_nan = True
						break
				if gradient_nan or np.isnan(q_vals).any():
					import pdb
					pdb.set_trace()
				q_vals, loss, min_score, gradients, _ = sess.run((Q.predictions, Q.loss, Q.min_score, Q.clipped_gradients, Q.updateWeights), feed_dict={Q.input_place_holder: book_vec, Q.target_values: targ})	
				if ts % 100 == 0:
					sess.run(updateTargetOperation)
					print ts
				if ts % 100 == 0:
					#print 'input'
					#print book_vec
					print str(t) + ',' + str(i)	
					print 'Q'
					print q_vals
					print 'targ'
					print targ
					print 'arg min'
					print argmins
					print 't cost'
					print t_costs	
					print np.mean(losses)
				losses.append(loss)
	print np.mean(losses)


def run_advantage_epoch(sess, env, Q, Q_target, updateTargetOperation, H, V, I, T, L, S):
	print 'hi'

def run_epoch(sess, env, Q, Q_target, updateTargetOperation, H, V, I, T, L, S):
	ob_size = Q.params.ob_size
	window = Q.params.window
	losses = []
	vol_unit = V / I
	time_unit = H / T
	for t in range(T+1)[::-1]:
		for i in range(I+1):
			for ts in range(1, S+1):
				curr_book = env.get_book(ts)
				tgt_price = env.mid_spread(ts)
				actions = sorted(curr_book.a.keys())
				actions.append(0)
				backup = np.zeros(shape=[L+1])
				argmins = np.zeros(shape=[L+1])
				t_costs = np.zeros(shape=[L+1])
				for a in range(L+1):
					curr_book = env.get_book(ts)
					limit_price = float("inf") if t == T else actions[a]
					spent, leftover = env.limit_order(0, limit_price, vol_unit * i)
					if t == T:
						spent += leftover * actions[-2]
						argmin = 0
						leftover = 0
					else: 
						rounded_unit = int(round(1.0 * leftover / vol_unit))
						next_book_vec = create_input_window_train(env, ts + time_unit, window, 1, ob_size, t + 1, rounded_unit)
						next_scores, argmin = sess.run((Q_target.predictions, Q_target.min_score), feed_dict={
														Q_target.input_place_holder: next_book_vec
													})
					diff = vol_unit * i - leftover
					price_paid = tgt_price if diff == 0 else spent / diff
					t_cost =  (float(price_paid) - tgt_price)/tgt_price * 100
					backup[a] = (t_cost * diff + argmin * leftover)/(vol_unit * i) if i!=0 else 0
					if np.isinf(argmin).any() or np.isinf(argmin).any():
						import pdb
						pbd.set_trace()
					argmins[a] = argmin
					t_costs[a] = t_cost
				targ = backup.reshape(1, 11)
				book_vec = create_input_window_train(env, ts, window, 1, ob_size, t, i)
				q_vals, loss, min_score, gradients = sess.run((Q.predictions, Q.loss, Q.min_score, Q.gvs), feed_dict={Q.input_place_holder: book_vec, Q.target_values: targ})	
				gradient_nan = False
				for g in gradients:
					if np.isnan(g).any() or np.isinf(g).any():
						gradient_nan = True
						break
				if gradient_nan or np.isnan(q_vals).any():
					import pdb
					pdb.set_trace()
				q_vals, loss, min_score, gradients, _ = sess.run((Q.predictions, Q.loss, Q.min_score, Q.clipped_gradients, Q.updateWeights), feed_dict={Q.input_place_holder: book_vec, Q.target_values: targ})	
				if ts % 100 == 0:
					sess.run(updateTargetOperation)
					print ts
				if ts % 100 == 0:
					#print 'input'
					#print book_vec
					print str(t) + ',' + str(i)	
					print 'Q'
					print q_vals
					print 'targ'
					print targ
					print 'arg min'
					print argmins
					print 't cost'
					print t_costs	
					print np.mean(losses)
				losses.append(loss)
	print np.mean(losses)

def train_DQN(epochs, ob_file, H, V, I, T, L, S, test_steps, params, env=None, debug=False):
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
	Q = Q_RNN(params, 'original')
	Q_target = Q_RNN(params, 'target', target=True)
	updateTargetOperation = Q_target.copy_Q_Op(Q)
	init = tf.initialize_all_variables()
	with tf.Session() as sess:
		sess.run(init)
		sess.run(updateTargetOperation)
		for i in range(epochs):
			run_continuos_epoch(sess, env, Q, Q_target, updateTargetOperation, H, V, I, T, L, S)
		executions = execute_algo(Q, sess, env, H, V, I, T, test_steps, S)
		write_trades(executions)

def write_function(function, T, L,functionFilename='deepQ'):
	table_file = open(functionFilename + '.csv', 'wb')
	tw = csv.writer(table_file)
	table_rows = []	
	table_rows.append(['Time Left', 'Rounded Units Left', 'Action', 'Expected Payout'])
	if type(function) is list:
		table_rows.append(function[0].coef_)
		table_rows.append(function[1].coef_)
		table_rows.append(function[0].intercept_)
		table_rows.append(function[1].intercept_)
	else:
		table_rows.append(function.coef_)
		table_rows.append(function.intercept_)
	tw.writerows(table_rows)


	

	
if __name__ == "__main__":
	params = Params(10, 10, 5, 1, 11, 1)
	train_DQN(1, '../data/10_GOOG.csv', 1000, 1000, 10, 10, 10, 10, 100000, params)

	