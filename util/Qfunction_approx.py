from environment import *
import random # for double q learning
import numpy as np 
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model

class Q_Function:

	def __init__(self, T, L, backup):
		self.backup = backup		
		self.T = T
		self.L = L
		self.pre_process = PolynomialFeatures(degree=2, include_bias=False)
		if self.backup['name'] == 'sampling':
			self.Q = linear_model.SGDRegressor(loss='huber', penalty='l2', learning_rate='invscaling', eta0=0.1, power_t=0.25, warm_start=False)
		elif self.backup['name'] == 'doubleQ':
			self.Q_1 = linear_model.SGDRegressor(loss='huber', penalty='l2', learning_rate='invscaling', eta0=0.1, power_t=0.25, warm_start=False)
			self.Q_2 = linear_model.SGDRegressor(loss='huber', penalty='l2', learning_rate='invscaling', eta0=0.1, power_t=0.25, warm_start=False)
		elif self.backup['name'] == 'replay buffer':
			self.Q = linear_model.SGDRegressor(loss='huber', penalty='l2', learning_rate='invscaling', eta0=0.1, power_t=0.25, warm_start=False)
			self.buff = []
		else:
			print "Illegal Backup Type"

	def train_example(self, func, x, y):
		x += 2
		func.partial_fit(self.pre_process.fit_transform(x), y)
		print self.backup['name'] + ' ' + str(x) + ' ' + str(y)

	def predict(self, func, x):
		x += 2
		return func.predict(self.pre_process.fit_transform(x))

	def update_table_buy(self, t, i, vol_unit, spread, volume_misbalance, im_cost, signed_vol, action, actions, env, tgt):
		# create keys to index into table
		in_a = -1 if action == self.L else action
		num_key = str(action)+ ',n'
		key = str(t) + "," + str(i) + "," + str(spread) + "," +str(volume_misbalance) + ',' + str(im_cost) + "," +str(signed_vol)

		if self.backup['name'] == "sampling":
			# determine limit price this action specifies and submit it to orderbook
			limit_price = float("inf") if t == self.T else actions[action]
			spent, leftover = env.limit_order(0, limit_price, vol_unit * i)
			# if at last step and leftovers, assume get the rest of the shares at the worst price in the book
			if t == self.T:
				if leftover	>= 0:
					spent += leftover * actions[-2]
					leftover = 0
				arg_min = 0
			else:
				rounded_unit = int(round(1.0 * leftover / vol_unit))
				next_key = str(t + 1) + "," + str(rounded_unit) + "," + str(spread) + "," +str(volume_misbalance) + ',' + str(im_cost) + "," +str(signed_vol)
				_, arg_min = self.arg_min(next_key)
			# update key entry
			diff = vol_unit * i - leftover
			price_paid =  tgt if diff == 0 else spent / diff
			t_cost =  (float(price_paid) - tgt)/tgt * 100
			weighted_reward = (t_cost * diff + arg_min * leftover)/(vol_unit * i) if i!=0 else 0
			s = [int(n) for n in key.split(',')]
			s.append(int(in_a))
			self.train_example(self.Q, np.array(s, ndmin=2), np.array((weighted_reward), ndmin=1))

		elif self.backup['name'] == "replay buffer":
			# pull replay information		
			replays = self.backup['replays']
			size = self.backup['buff_size']
			# determine limit price this action specifies and submit it to orderbook
			limit_price = float("inf") if t == self.T else actions[action]
			spent, leftover = env.limit_order(0, limit_price, vol_unit * i)
			# if at last step and leftovers, assume get the rest of the shares at the worst price in the book - no need to replay terminal states
			if t == self.T:
				if leftover	>= 0:
					spent += leftover * actions[-2]
					leftover = 0
					diff = vol_unit * i - leftover
					price_paid =  tgt if diff == 0 else spent / diff
					t_cost =  (float(price_paid) - tgt)/tgt * 100
					s = [int(n) for n in key.split(',')]
					s.append(int(in_a))
					self.train_example(self.Q, np.array(s, ndmin=2), np.array(t_cost, ndmin=1))
			else:
				rounded_unit = int(round(1.0 * leftover / vol_unit))
				next_key = str(t + 1) + "," + str(rounded_unit) + "," + str(spread) + "," +str(volume_misbalance) + ',' + str(im_cost) + "," +str(signed_vol)
				_, arg_min = self.arg_min(next_key)

				# update the table with this key 
				diff = vol_unit * i - leftover
				price_paid =  tgt if diff == 0 else spent / diff
				t_cost =  (float(price_paid) - tgt)/tgt * 100
				weighted_reward = (t_cost * diff + arg_min * leftover)/(vol_unit * i) if i!=0 else 0
				s = [int(n) for n in key.split(',')]
				s.append(int(in_a))
				self.train_example(self.Q, np.array(s, ndmin=2), np.array((weighted_reward), ndmin=1))
				# add this SARSA transition to the buffer
				self.buff.append((key, action, next_key, t_cost, diff, leftover))
				if len(self.buff) > size:
					self.buff.pop(0)
				# update table by sampling from buffer of transitions
				for r in range(0, replays):
					s,a,s_n, t_c, d, l = random.choice(self.buff)
					_, a_m = self.arg_min(s_n)
					s = [int(n) for n in s_n.split(',')]
					s.append(int(in_a))
					w_r = (t_c*d + a_m * l)/(d + l) if d + l != 0 else 0
					self.train_example(self.Q, np.array(s, ndmin=2), np.array((w_r), ndmin=1))

		elif self.backup['name'] == "doubleQ":
			# set values appropriately for new keys
			# double Q, so flip a coin to determine which table to update (Q1 or Q2)
			Q_table_number = random.randint(1, 2) - 1
			use_Q = [self.Q_1, self.Q_2][Q_table_number]
			# determine limit price this action specifies and submit it to orderbook
			limit_price = float("inf") if t == self.T else actions[action]
			spent, leftover = env.limit_order(0, limit_price, vol_unit * i)
			# if at last step and leftovers, assume get the rest of the shares at the worst price in the book - no need to replay terminal states
			if t == self.T:
				if leftover	>= 0:
					spent += leftover * actions[-2]
				arg_min = 0
			else:
				rounded_unit = int(round(1.0 * leftover / vol_unit))
				next_key = str(t + 1) + "," + str(rounded_unit)+ "," + str(spread) + "," +str(volume_misbalance) + ','+ str(im_cost) + "," +str(signed_vol)
				arg_min = float("inf")
				# use the opposite table to the current one (flip lookup)
				next_state_table = [self.Q_2, self.Q_1][Q_table_number]
				_, arg_min = self.arg_min(next_key, func=next_state_table)

			# update key
			diff = vol_unit * i - leftover
			price_paid =  tgt if diff == 0 else spent / diff
			t_cost =  (float(price_paid) - tgt)/tgt * 100
			weighted_reward = (t_cost * diff + arg_min * leftover)/(vol_unit * i) if i!=0 else 0
			s = [int(n) for n in key.split(',')]
			s.append(int(in_a))
			self.train_example(use_Q, np.array(s, ndmin=2), np.array((weighted_reward), ndmin=1))

		else:
			print "Illegal backup"


	def arg_min(self, key, func=None):
		if func is None:
			func = self.Q
		min_action = -1
		min_val = float("inf")
		for action in range(self.L+1):
			in_a = -1 if action == self.L else action
			x = [int(n) for n in key.split(',')]
			x.append(int(in_a))
			value = self.predict(func, np.array(x,ndmin=2))
			if value < min_val:
					min_val = value
					min_action = action
		return min_action, min_val


	def greedy_action(self, t_left, rounded_unit, spread, volume_misbalance, im_cost, signed_vol, ts):
		min_action = -1
		min_val = float("inf")
		key = str(t_left) + ',' + str(rounded_unit) + "," + str(spread) + "," +str(volume_misbalance) + ',' + str(im_cost) + "," +str(signed_vol)
		for action in range(self.L+1):
			in_a = -1 if action == self.L else action
			x = [int(n) for n in key.split(',')]
			x.append(int(in_a))
			if self.backup['name'] == 'doubleQ':
				value = (self.predict(self.Q_1, np.array(x,ndmin=2)) + self.predict(self.Q_2, np.array(x,ndmin=2)))/2
			else:	
				value = self.predict(self.Q, np.array(x,ndmin=2))
			if value < min_val:
					min_val = value
					min_action = action
		return min_action, min_val


