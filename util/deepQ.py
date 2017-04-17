from environment import *
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


class Q_RNN: 

	def __init__(self, params, name, target=False):
		self.name = name
		self.params = params
		self.build_model_graph()
		self.add_training_objective()

	def build_model_graph(self): 
		with tf.variable_scope(self.name) as self.scope:
			self.input_place_holder = tf.placeholder(tf.float32, shape=(self.params.batch, self.params.window, self.params.ob_size * 4), name='input')
			self.forward_cell_layers = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.LSTMCell(self.params.hidden_size) for i in range(self.params.hidden_depth)])
			self.rnn_output, self.final_rnn_state = tf.nn.dynamic_rnn(self.forward_cell_layers, self.input_place_holder, \
								sequence_length=[self.params.window]*self.params.batch, dtype=tf.float32)
			self.outs = tf.squeeze(tf.slice(self.rnn_output, [0, self.params.window - 1, 0], [self.params.batch, 1, self.params.hidden_size]), axis=1)
			self.U = tf.get_variable('U', shape=[self.params.hidden_size, self.params.actions])
			self.b_2 = tf.get_variable('b2', shape=[self.params.actions])
			self.predictions = tf.cast((tf.matmul(self.outs, self.U) + self.b_2), 'float32')
			self.min_score = tf.reduce_min(self.predictions, reduction_indices=[1])


	def add_training_objective(self):
		self.target_values = tf.placeholder(tf.float32, shape=[self.params.batch, self.params.actions], name='target')
		self.batch_losses = tf.reduce_sum(tf.sqrt(tf.squared_difference(self.predictions, self.target_values)), axis=1)
		self.loss = tf.reduce_sum(self.batch_losses, axis=0) + tf.nn.l2_loss(self.U) + tf.nn.l2_loss(self.b_2)
		self.trainer = tf.train.AdamOptimizer(learning_rate=0.1)
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


def create_input_window(env, ts, window, batch, ob_size):
	if(ts < window - 1):
		return np.zeros(shape=[batch, window, ob_size * 4])
	else:
		vecs = []
		for i in range(ts - window + 1, ts + 1):
			book_vec = env.get_book(ts).vectorize_book(10).reshape(1,1,ob_size * 4)
			vecs.append(book_vec)
		window_vec = np.concatenate(vecs, axis=1)
		return window_vec


def run_epoch(sess, env, Q, Q_target, updateTargetOperation,L, V):
	losses = []
	for ts in range(15000):
		curr_book = env.get_book(ts)
		tgt_price = env.mid_spread(ts)
		actions = sorted(curr_book.a.keys())
		actions.append(0)
		rewards = np.zeros(shape=[L+1])
		for a in range(L+1):
			curr_book = env.get_book(ts)
			spent, leftover = env.limit_order(0, actions[a], V)
			diff = V - leftover
			price_paid =  tgt_price if diff == 0 else spent / diff
			t_cost =  (float(price_paid) - tgt_price)/tgt_price * 100
			rewards[a] = t_cost
		book_vec = create_input_window(env, ts, 10, 1, 10)
		target_scores, min_score = sess.run((Q_target.predictions, Q_target.min_score), feed_dict={
			Q_target.input_place_holder: book_vec
		})
		min_action = np.argmin(target_scores, axis=1)
		true_val = target_scores + min_score
		true_val.reshape(1, 11)
		q_vals, loss, min_score, _ = sess.run((Q.predictions, Q.loss, Q.min_score, Q.updateWeights), feed_dict={
			Q.input_place_holder: book_vec,
			Q.target_values: true_val
		})
		if ts % 100 == 0:
			sess.run(updateTargetOperation)
		losses.append(loss)
	print true_val
	print q_vals
	print np.mean(losses)

def train_DQN(epochs, ob_file, H, V, I, T, L, debug=False):
	env = Environment(ob_file,setup=False)
	params = Params(10, 10, 5, 1, L + 1, 1)
	Q = Q_RNN(params, 'original')
	Q_target = Q_RNN(params, 'target', target=True)
	updateTargetOperation = Q_target.copy_Q_Op(Q)
	init = tf.initialize_all_variables()
	with tf.Session() as sess:
		sess.run(init)
		for i in range(10):
			run_epoch(sess, env, Q, Q_target, updateTargetOperation, L, V)




	

	
if __name__ == "__main__":
	train_DQN(30, '../data/10_GOOG.csv', 1000, 1000, 10, 10, 10, debug=False)

	