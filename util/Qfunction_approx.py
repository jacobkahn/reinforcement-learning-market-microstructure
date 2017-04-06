from environment import *
import random # for double q learning
import numpy as np 
from sklearn import linear_model

class Q_Function:

	def __init__(self, T, L, backup, function):
		self.backup = backup
		self.T = T
		self.L = L
		if self.backup['name'] == 'sampling':
			self.Q = linear_model.SGDRegressor(loss='huber', penalty='l2', seed=0, learning_rate='invscaling', eta0=0.1, power_t=0.25, warm_start=False)
		elif self.backup['name'] == 'doubleQ':
			self.Q_1 = linear_model.SGDRegressor(loss='huber', penalty='l2', seed=0, learning_rate='invscaling', eta0=0.1, power_t=0.25, warm_start=False)
			self.Q_2 = linear_model.SGDRegressor(loss='huber', penalty='l2', seed=0, learning_rate='invscaling', eta0=0.1, power_t=0.25, warm_start=False)
		elif self.backup['name'] == 'replay buffer':
			self.Q = linear_model.SGDRegressor(loss='huber', penalty='l2', seed=0, learning_rate='invscaling', eta0=0.1, power_t=0.25, warm_start=False)
			self.buff = []
		else:
			print "Illegal Backup Type"

	def update_table_buy(self, t, i, vol_unit, spread, volume_misbalance, action, actions, env, tgt):
		# create keys to index into table
		num_key = str(action)+ ',n'
		key = str(t) + "," + str(i)+ "," +str(spread) + "," +str(volume_misbalance)
		self.update_table_keys(key, num_key)

		if self.backup['name'] == "sampling":
			# determine limit price this action specifies and submit it to orderbook
			limit_price = float("inf") if t == self.T else actions[action]
			spent, leftover = env.limit_order(0, limit_price, vol_unit * i)
			# if at last step and leftovers, assume get the rest of the shares at the worst price in the book
			if t == self.T:
				if leftover	>= 0:
					spent += leftover * actions[-2]
				arg_min = 0
			else:
				rounded_unit = int(round(1.0 * leftover / vol_unit))
				next_key = str(t + 1) + "," + str(rounded_unit)+ "," + str(spread) + "," +str(volume_misbalance)
				_, arg_min = self.arg_min(next_key)
			# update key entry
			price_paid = spent / (vol_unit * i - leftover)
			t_cost =  (float(price_paid) - tgt)/tgt * 100
			s = [int(n) for n in key.split(',')]
			s.append(int(a))
			self.Q.partial_fit(np.array(s, ndim=2), t_cost + arg_min)
			self.Q[key][num_key] += 1

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
					price_paid = spent / (vol_unit * i)
					t_cost =  (float(price_paid) - tgt)/tgt * 100
					s = [int(n) for n in key.split(',')]
					s.append(int(a))
					self.Q.partial_fit(np.array(s, ndim=2), t_cost)
			else:
				rounded_unit = int(round(1.0 * leftover / vol_unit))
				next_key = str(t + 1) + "," + str(rounded_unit)+ "," + str(spread) + "," +str(volume_misbalance)
				_, arg_min = self.arg_min(next_key)

				# update the table with this key 
				price_paid = spent / (vol_unit * i - leftover)
				t_cost =  (float(price_paid) - tgt)/tgt * 100
				s = [int(n) for n in key.split(',')]
				s.append(int(a))
				self.Q.partial_fit(np.array(s, ndim=2), t_cost + arg_min)
				# add this SARSA transition to the buffer
				self.buff.append((key, action, next_key, t_cost))
				if len(self.buff) > size:
					self.buff.pop(0)
				# update table by sampling from buffer of transitions
				for r in range(0, replays):
					s,a,s_n, r = random.choice(self.buff)
					c_key =  str(a)+ ',n'
					_, a_m = self.arg_min(s_n)
					s = [int(n) for n in s_n.split(',')]
					s.append(int(a))
					self.Q.partial_fit(np.array(s, ndim=2), t_cost + arg_min)

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
				next_key = str(t + 1) + "," + str(rounded_unit)+ "," + str(spread) + "," +str(volume_misbalance)
				arg_min = float("inf")
				# use the opposite table to the current one (flip lookup)
				next_state_table = [self.Q_2, self.Q_1][Q_table_number]
				# if the other table doesn't have a next state, then use the reverted table next state
				if not next_key in next_state_table:
					next_state_table = use_Q
				_, arg_min = self.arg_min(next_key, func=next_state_table)

			# update key
			price_paid = spent / (vol_unit * i - leftover)
			t_cost =  (float(price_paid) - tgt)/tgt * 100
			s = [int(n) for n in key.split(',')]
			s.append(int(a))
			use_Q.partial_fit(np.array(s, ndim=2), t_cost + arg_min)

		else:
			print "Illegal backup"


	def arg_min(self, key, func=self.Q):
		min_action = -1
		min_val = float("inf")
		if key in table:
			for key, value in table[key].items():
				action, _ = value.split(',')
				x = [int(n) for n in key.split(',')]
				x.append(int(a))
				value = func.predict(np.array(x,ndim=2))
				if value < min_val:
						min_val = value
						min_action = action
		else:
			# if we haven't seen this state, give most aggressive order - 0
			return 0
		return min_action, min_val


	def greedy_action(self, t_left, rounded_unit, spread, volume_misbalance, ts):
		func = self.curr_Q if self.backup['name'] == 'doubleQ' else self.Q
		min_action = -1
		min_val = float("inf")
		key = str(t_left) + ',' + str(rounded_unit) + ',' + str(spread) + ',' + str(volume_misbalance)
		if key in table:
			for key, value in table[key].items():
				action, _ = value.split(',')
				x = [int(n) for n in key.split(',')]
				x.append(int(a))
				if self.backup['name'] == 'doubleQ':
					value = (self.Q_1.predict(np.array(x,ndim=2)) + self.Q_2.predict(np.array(x,ndim=2)))/2
				else:	
					value = self.Q.predict(np.array(x,ndim=2))
				if value < min_val:
						min_val = value
						min_action = action
		else:
			# if we haven't seen this state, give most aggressive order - 0
			return 0
		return min_action, min_val


