from deepQ import *


class Q_CNN: 

	def __init__(self, params, name):
		self.name = name
		self.advantage = params['advantage']
		self.params = Params(params['window'], params['ob_size'], params['hidden_size'], params['depth'], params['actions'], params['batch'])
		self.layers = params['layers']
		self.build_model_graph()
		self.add_training_objective()

	def build_model_graph(self):
		self.filter_tensors = {}
		self.bias_tensors = {}
		# lots to decisions
		with tf.variable_scope(self.name) as self.scope:
			self.input_place_holder = tf.placeholder(tf.float32, shape=(None, self.params.window, self.params.ob_size * 4 + 2), name='input')
			curr_dimension = [tf.shape(self.input_place_holder)[0], self.params.window, self.params.ob_size * 4 + 2, 1]
			curr_layer = tf.reshape(self.input_place_holder, curr_dimension)
			for name, layer_params in sorted(self.layers.items()):
				print curr_dimension
				print curr_layer 
				if layer_params['type'] == 'conv':
					n = 'conv_{}_filter_size_{}_stride_{}_num_{}'.format(name, layer_params['size'], layer_params['stride'], layer_params['num'])
					s = [layer_params['size'], layer_params['size'], curr_dimension[3], layer_params['num']]
					strides = [1, layer_params['stride'], layer_params['stride'], 1]
					self.filter_tensors[name] = tf.Variable(tf.truncated_normal(s, stddev=0.0001), name=n)
					self.bias_tensors[name] = tf.Variable(tf.truncated_normal(shape=[layer_params['num']], stddev=0.1), name=n + '_bias')
					conv_output = tf.nn.conv2d(curr_layer, self.filter_tensors[name], strides, "VALID")
					conv_bias = tf.nn.bias_add(conv_output, self.bias_tensors[name])
					curr_layer = tf.nn.relu(conv_bias)
					curr_dimension = compute_output_size(curr_dimension[0], curr_dimension[1], curr_dimension[2],layer_params['size'], layer_params['stride'], 0, layer_params['num'])
				if layer_params['type'] == 'pool':
					if layer_params['pool_type'] == 'max':
						s = [1, layer_params['size'], layer_params['size'], 1]
						stride = [1, layer_params['stride'], layer_params['stride'], 1]
						x = tf.nn.max_pool(curr_layer, s, stride, 'VALID')
						curr_layer = x
						curr_dimension = compute_pool_size(curr_dimension[0], curr_dimension[1], curr_dimension[2],layer_params['size'], layer_params['stride'], curr_dimension[3])
					if layer_params['pool_type'] == 'avg':
						s = [1, layer_params['size'], layer_params['size'], 1]
						stride = [1, layer_params['stride'], layer_params['stride'], 1]
						x = tf.nn.avg_pool(curr_layer, s, stride, 'VALID')
						curr_layer = x
						curr_dimension = compute_pool_size(curr_dimension[0], curr_dimension[1], curr_dimension[2],layer_params['size'], layer_params['stride'], curr_dimension[3])
				if layer_params['type'] == 'fc':
					print 'hi'
			print curr_dimension
			print curr_layer
			if not self.advantage:
				final_s = [curr_dimension[1], curr_dimension[2], curr_dimension[3],self.params.actions]
				strides = [1,1,1,1]
				projection = tf.Variable(tf.truncated_normal(final_s, stddev=0.1), name="final_projection")
				bias = tf.Variable(tf.truncated_normal([self.params.actions], stddev=0.1), name="final_projection")
				self.outs = tf.nn.conv2d(curr_layer, projection, strides, 'VALID') + bias
				self.predictions = tf.squeeze(self.outs, squeeze_dims=[1, 2])
			else:
				self.advantage_stream, self.value_stream = tf.split(curr_layer, 2, 3)
				final_s_a = [curr_dimension[1], curr_dimension[2], curr_dimension[3]/2, self.params.actions]
				final_s_v = [curr_dimension[1], curr_dimension[2], curr_dimension[3]/2, 1]
				strides = [1,1,1,1]
				self.projection_a = tf.Variable(tf.truncated_normal(final_s_a, stddev=0.01), name="final_projection")
				self.projection_v = tf.Variable(tf.truncated_normal(final_s_v, stddev=0.01), name="final_projection")
				self.A = tf.squeeze(tf.nn.conv2d(self.advantage_stream, self.projection_a, strides, 'VALID'), squeeze_dims=[1,2])
				self.V = tf.squeeze(tf.nn.conv2d(self.value_stream, self.projection_v, strides, 'VALID'), squeeze_dims=[1,2])
				self.predictions = self.V + tf.subtract(self.A, tf.reduce_mean(self.A, axis=1, keep_dims=True))
			self.min_score = tf.reduce_min(self.predictions, axis=[1])
			self.min_action = tf.argmin(tf.squeeze(self.predictions), axis=0, name="arg_min")



	def add_training_objective(self):
		self.target_values = tf.placeholder(tf.float32, shape=[self.params.batch, self.params.actions], name='target')
		self.batch_losses = tf.reduce_sum(tf.squared_difference(self.predictions, self.target_values), axis=1)
		self.loss = tf.reduce_sum(self.batch_losses, axis=0)
		self.trainer = tf.train.AdamOptimizer(learning_rate=0.0001)
		self.gvs, self.variables = zip(*self.trainer.compute_gradients(self.loss))
		self.clipped_gradients, _ = tf.clip_by_global_norm(self.gvs, 20.0)
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


class Q_RNN: 

	def __init__(self, params, name):
		self.name = name
		self.params = Params(params['window'], params['ob_size'], params['hidden_size'], params['depth'], params['actions'], params['batch'])
		self.advantage = params['advantage']
		self.build_model_graph()	
		self.add_training_objective()

	def build_model_graph(self): 
		with tf.variable_scope(self.name) as self.scope:
			self.input_place_holder = tf.placeholder(tf.float32, shape=(None, self.params.window, self.params.ob_size * 4 + 2), name='input')
			self.forward_cell_layers = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.LSTMCell(self.params.hidden_size) for i in range(self.params.hidden_depth)])
			self.rnn_output, self.final_rnn_state = tf.nn.dynamic_rnn(self.forward_cell_layers, self.input_place_holder, dtype=tf.float32)
			self.outs = tf.squeeze(tf.slice(self.rnn_output, [0, self.params.window - 1, 0], [-1, 1, self.params.hidden_size]), axis=1)
			
			if not self.advantage:
				self.U = tf.get_variable('U', shape=[self.params.hidden_size, self.params.actions])
				self.b_2 = tf.get_variable('b2', shape=[self.params.actions])
				self.predictions = tf.cast((tf.matmul(self.outs, self.U) + self.b_2), 'float32') 
			else:
				self.advantage_stream, self.value_stream = tf.split(self.outs, 2, 1)
				self.U_a = tf.get_variable('U_a', shape=[self.params.hidden_size//2, self.params.actions])
				self.U_v = tf.get_variable('U_v', shape=[self.params.hidden_size//2, 1])
				self.A =  tf.cast(tf.matmul(self.advantage_stream, self.U_a), 'float32') 
				self.V = tf.cast(tf.matmul(self.value_stream, self.U_v), 'float32') 
				self.predictions = self.V + tf.subtract(self.A, tf.reduce_mean(self.A, axis=1, keep_dims=True))
			self.min_score = tf.reduce_min(self.predictions, reduction_indices=[1])
			self.min_action = tf.argmin(tf.squeeze(self.predictions), axis=0, name="arg_min")

	def add_training_objective(self):
		self.target_values = tf.placeholder(tf.float32, shape=[self.params.batch, self.params.actions], name='target')
		self.batch_losses = tf.reduce_sum(tf.squared_difference(self.predictions, self.target_values), axis=1)
		self.loss = tf.reduce_sum(self.batch_losses, axis=0)
		self.trainer = tf.train.AdamOptimizer(learning_rate=0.0001)
		self.gvs, self.variables = zip(*self.trainer.compute_gradients(self.loss))
		self.clipped_gradients, _ = tf.clip_by_global_norm(self.gvs, 20.0)
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