from environment import *
import random # for double q learning
import numpy as np 
from sklearn import linear_model

class Q:

	def __init__(self, T, L, backup):
		self.backup = backup
		self.T = T
		self.L = L
		if self.backup['name'] == 'sampling':
			self.Q = {}
		elif self.backup['name'] == 'doubleQ':
			self.Q_1 = {}
			self.Q_2 = {}
			self.curr_Q = {} # aggregate average table, updated dynamically
		elif self.backup['name'] == 'replay buffer':
			self.Q = {}
			self.buff = []
		else:
			print "Illegal Backup Type"

	def update_table_keys(self, key, num_key, use_Q):
		if key not in use_Q:
			use_Q[key] = {}
		if num_key not in use_Q[key]:
			use_Q[key][num_key] = 0
		if action not in use_Q[key]:
			use_Q[key][action] = 0
		n = use_Q[key][num_key]
		return n


	def update_table_buy(self, t, i, vol_unit, spread, volume_misbalance, action, actions, env, tgt):
		# create keys to index into table
		num_key = str(action)+ ',n'
		key = str(t) + "," + str(i)+ "," +str(spread) + "," +str(volume_misbalance)
		self.update_table_keys(key, num_key)

		if self.backup['name'] == "sampling":
			# set values appropriately for new keys
			n = self.update_table_keys(key, num_key, self.Q)
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

			self.Q[key][action] = float(n)/(n+1)*self.Q[key][action] + float(1)/(n+1)*(t_cost + arg_min)
			self.Q[key][num_key] += 1

		elif self.backup['name'] == "replay buffer":
			# pull replay information
			replays = self.backup['replays']
			size = self.backup['buff_size']
			# update table for new keys
			n = self.update_table_keys(key, num_key, self.Q)
			# determine limit price this action specifies and submit it to orderbook
			limit_price = float("inf") if t == self.T else actions[action]
			spent, leftover = env.limit_order(0, limit_price, vol_unit * i)
			# if at last step and leftovers, assume get the rest of the shares at the worst price in the book - no need to replay terminal states
			if t == self.T:
				if leftover	>= 0:
					spent += leftover * actions[-2]
					price_paid = spent / (vol_unit * i)
					t_cost =  (float(price_paid) - tgt)/tgt * 100
					self.Q[key][action] = float(n)/(n+1)*self.Q[key][action] + float(1)/(n+1)*(t_cost)
					self.Q[key][num_key] += 1
			else:
				rounded_unit = int(round(1.0 * leftover / vol_unit))
				next_key = str(t + 1) + "," + str(rounded_unit)+ "," + str(spread) + "," +str(volume_misbalance)
				_, arg_min = self.arg_min(next_key)

				# update the table with this key - we do this to make sure every key ends up in table at least once
				price_paid = spent / (vol_unit * i - leftover)
				t_cost =  (float(price_paid) - tgt)/tgt * 100
				self.Q[key][action] = float(n)/(n+1)*self.Q[key][action] + float(1)/(n+1)*(t_cost + arg_min)
				self.Q[key][num_key] += 1
				# add this SARSA transition to the buffer
				self.buff.append((key, action, next_key, t_cost))
				if len(self.buff) > size:
					self.buff.pop(0)
				# update table by sampling from buffer of transitions
				for r in range(0, replays):
					s,a,s_n, r = random.choice(self.buff)
					c_key =  str(a)+ ',n'
					num = self.Q[s][c_key]
					_, a_m = self.arg_min(s_n)
					self.Q[s][a] = float(num)/(num+1)*self.Q[s][a] + float(1)/(num+1)*(r + a_m)
					self.Q[s][c_key] += 1

		elif self.backup['name'] == "doubleQ":
			# set values appropriately for new keys
			# double Q, so flip a coin to determine which table to update (Q1 or Q2)
			Q_table_number = random.randint(1, 2) - 1
			use_Q = [self.Q_1, self.Q_2][Q_table_number]
			n = self.update_table_keys(key, num_key, use_Q)
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
				_, arg_min = self.arg_min(next_key, next_state_table)

			# update key
			price_paid = spent / (vol_unit * i - leftover)
			t_cost =  (float(price_paid) - tgt)/tgt * 100
			use_Q[key][action] = float(n) / (n+1) * use_Q[key][action] + float(1) / (n + 1) * (t_cost + arg_min)
			use_Q[key][num_key] += 1

			# update the average curr_Q table
			# make sure keys are in place
			if key not in self.curr_Q:
				self.curr_Q[key] = {}
			if num_key not in self.curr_Q[key]:
				self.curr_Q[key][num_key] = 0
			if action not in self.curr_Q[key]:
				self.curr_Q[key][action] = 0
			# set the num_key to the total number of times we see the action
			Q_1_num_key = 0
			Q_2_num_key = 0
			if key in self.Q_1 and num_key in self.Q_1[key]:
				Q_1_num_key = self.Q_1[key][num_key]
			elif key in self.Q_2 and num_key in self.Q_2[key]:
				Q_2_num_key = self.Q_2[key][num_key]
			self.curr_Q[key][num_key] = Q_1_num_key + Q_2_num_key
			# ensure the key and action are present in both tables
			if key not in self.Q_1 or action not in self.Q_1[key]:
				self.curr_Q[key][action] = self.Q_2[key][action]
			elif key not in self.Q_2 or action not in self.Q_2[key]:
				self.curr_Q[key][action] = self.Q_1[key][action]
			else:
				self.curr_Q[key][action] = (self.Q_1[key][action] + self.Q_2[key][action]) / 2
		else:
			print "Illegal backup"


	def arg_min(self, key,table=self.Q):
		min_action = -1
		min_val = float("inf")
		if key in table:
			for action, value in table[key].items():
				if type(action) != str:
					if value < min_val:
						min_val = value
						min_action = action
		else:
			# if we haven't seen this state, give most aggressive order - 0
			return 0
		return min_action, min_val


	def greedy_action(self, t_left, rounded_unit, spread, volume_misbalance, ts):
		table = self.curr_Q if self.backup['name'] == 'doubleQ' else self.Q
		min_action = -1
		min_val = float("inf")
		key = str(t_left) + ',' + str(rounded_unit) + ',' + str(spread) + ',' + str(volume_misbalance)
		print key + ' ' + str(ts)
		if key in table:
			for action, value in table[key].items():
				if type(action) != str:
					if value < min_val:
						min_val = value
						min_action = action
		else:
			# if we haven't seen this state, give most aggressive order - 0
			return 0
		return min_action, min_val
