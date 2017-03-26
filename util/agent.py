from environment import *
from collections import defaultdict
import random
import csv
import sys


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
		elif self.backup['name'] == 'replay buffer':
			self.Q = {}
			self.buff = []
		else:
			print "Illegal Backup Type"	

	def update_table_buy(self, t, i, vol_unit, spread, volume_misbalance, action, actions, env):
		if self.backup['name'] == "sampling": 
			# create keys to index into table
			num_key = str(action)+ ',n'
			key = str(t) + "," + str(i)+ "," +str(spread) + "," +str(volume_misbalance) 
			# set values appropriately for new keys
			if key not in self.Q:
				self.Q[key] = {}
			if num_key not in self.Q[key]:
				self.Q[key][num_key] = 0
			if action not in self.Q[key]:
				self.Q[key][action] = 0	
			n = self.Q[key][num_key]
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
				arg_min = float("inf")
				next_state = self.Q[next_key]
				for k,v in next_state.items():
					if type(k) != str: 
						arg_min = v if arg_min > v else arg_min
			# update key entry
			self.Q[key][action] = float(n)/(n+1)*self.Q[key][action] + float(1)/(n+1)*(spent + arg_min)
			self.Q[key][num_key] += 1
		elif self.backup['name'] == "replay buffer":
			# pull replay information
			replays = self.backup['replays']
			size = self.backup['buff_size']
			# create keys to index into table
			num_key = str(action)+ ',n'
			key = str(t) + "," + str(i)+ "," + str(spread) + "," +str(volume_misbalance) 
			# set values appropriately for new keys
			if key not in self.Q:
				self.Q[key] = {}
			if num_key not in self.Q[key]:
				self.Q[key][num_key] = 0
			if action not in self.Q[key]:
				self.Q[key][action] = 0	
			n = self.Q[key][num_key]
			# determine limit price this action specifies and submit it to orderbook
			limit_price = float("inf") if t == self.T else actions[action]
			spent, leftover = env.limit_order(0, limit_price, vol_unit * i)
			# if at last step and leftovers, assume get the rest of the shares at the worst price in the book - no need to replay terminal states
			if t == self.T:
				if leftover	>= 0:
					spent += leftover * actions[-2]
				self.Q[key][action] = float(n)/(n+1)*self.Q[key][action] + float(1)/(n+1)*(spent)
			else: 
				rounded_unit = int(round(1.0 * leftover / vol_unit))
				next_key = str(t + 1) + "," + str(rounded_unit)+ "," + str(spread) + "," +str(volume_misbalance)
				arg_min = float("inf")
				next_state = self.Q[next_key]
				for k,v in next_state.items():
					if type(k) != str: 
						arg_min = v if arg_min > v else arg_min
				# update the table with this key - we do this to make sure every key ends up in table at least once		
				self.Q[key][action] = float(n)/(n+1)*self.Q[key][action] + float(1)/(n+1)*(spent + arg_min)
				self.Q[key][num_key] += 1
				# add this SARSA transition to the buffer
				self.buff.append((key, action, next_key, spent))
				if len(self.buff) > size:
					self.buff.pop(0)
				# update table by sampling from buffer of transitions
				for r in range(0, replays):
					s,a,s_n, r = random.choice(self.buff)
					c_key =  str(a)+ ',n'
					num = self.Q[s][c_key]
					n_s = self.Q[s_n]
					a_m = float("inf")
					for k,v in n_s.items():
						if type(k) != str: 
							a_m = v if a_m > v else a_m
					self.Q[s][a] = float(num)/(num+1)*self.Q[s][a] + float(1)/(num+1)*(r + a_m)
					self.Q[s][c_key] += 1
		else:
			print "Illegal backup"

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
		return min_action

''' 
Algorithm from Kearns 2006 Paper
Uses market variables for bid-ask spread and volume misbalance
Arguments:
	H: Horizon for trading - number of orderbook steps we get to trade total
	V: Number of shares to trade
	I: Number of units to consider inventory in
	T: Number of decision points(units of time)
	L: Number of actions to try(how many levels into the book we should consider)
	S: Number of orderbooks to train Q function on
	divs: Number of intervals to discretize spreads and misbalances 
'''
def dp_algo(ob_file, H, V, I, T, L, S=10000, divs=10):
	backup = { 'name': 'replay buffer', 
				'buff_size': 100,
				'replays': 20
				}
	table = Q(T, L, backup)
	env = Environment(ob_file, setup=False)
	all_books = len(env.books)
	steps = H / T
	# state variables

	t = T
	inv = V
	vol_unit = V/I
	volume_misbalance = 0

	env.get_timesteps(0, S+1)	
	spreads, misbalances = create_variable_divs(divs, env)
	
	# loop for the DP algorithm
	for t in range(0, T+1)[::-1]:
		print t
		for ts in range(0, S): 
			if ts % 1000 == 0:
				print ts
			curr_book = env.get_book(ts)
			spread = compute_bid_ask_spread(curr_book, spreads)
			volume_misbalance = compute_volume_misbalance(curr_book, misbalances, env)	
			actions = sorted(curr_book.a.keys())
			actions.append(0)
			for i in range(0, I + 1):
				for action in range(0, L+1):
					# regenerate the order book so that we don't have the effects of the last action
					curr_book = env.get_book(ts)
					table.update_table_buy(t, i, vol_unit, spread, volume_misbalance, action, actions, env)
	executions = execute_algo(table, env, H, V, I, T, 10000, spreads, misbalances)
	write_model_files(table.Q, executions, T, L)
	import pdb
	pdb.set_trace()


def create_variable_divs(divs, env):
	spreads = []
	misbalances = []
	if divs > 1:
		spread_diff = (env.max_spread - env.min_spread) * 1.0 / (divs)
		misbalance_diff = (env.max_misbalance - env.min_misbalance) * 1.0 / (divs)
		for i in range(1, divs):
			spreads.append(env.min_spread + i * spread_diff)
			misbalances.append(env.min_misbalance + i * misbalance_diff)
	spreads.sort()
	misbalances.sort()
	return spreads, misbalances

def compute_bid_ask_spread(curr_book, spreads):
	spread = min(curr_book.a.keys()) - max(curr_book.b.keys())
	if len(spreads) == 0 or spread < spreads[0]:
		return 0
	for i in range(len(spreads) - 1):
		if spread >= spreads[i] and spread < spreads[i+1]:
			return (i + 1)
	return len(spreads)

def compute_volume_misbalance(curr_book, misbalances, env):
	m = env.misbalance(curr_book)
	if len(misbalances) == 0 or m < misbalances[0]:
		return 0
	for i in range(len(misbalances) - 1):
		if m >= misbalances[i] and misbalances < misbalances[i+1]:
			return (i + 1)
	return len(misbalances)

def execute_algo(table, env, H, V, I, T, steps, spreads, misbalances):
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
		# regenerate orderbook simulation for the next time horizon of decisions
		if ts % (T+1) == 0:
			env.get_timesteps(ts*time_unit, ts*time_unit+T*time_unit+1)
			volume = V

		# update the state of the algorithm based on the current book and timestep
		rounded_unit = int(volume / vol_unit)
		t_left =  ts % (T + 1)
		curr_book = env.get_next_state()
		perfect_price = env.mid_spread(ts - ts % (T + 1))
		spread = compute_bid_ask_spread(curr_book, spreads)
		volume_misbalance = compute_volume_misbalance(curr_book, misbalances, env)	
		actions = sorted(curr_book.a.keys())
		actions.append(0)

		# compute and execute the next action using the table

		min_action = table.greedy_action(t_left, rounded_unit, spread, volume_misbalance, ts)
		paid, leftover = env.limit_order(0, actions[min_action], volume)


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
			reward = [t_left, rounded_unit, spread, volume_misbalance, min_action, basis_p, volume - leftover]

		executions.append(reward)
		volume = leftover

		# simulate market till next decision point - no need to simulate after last decision point
		if ts % T != 0:
			for i in range(0, time_unit - 1):
				env.get_next_state()	

	return executions


def write_model_files(table, executions, T, L): 
	table_file = open("table1.csv", 'wb')
	trade_file = open("trades1.csv", 'wb')
		# write trades executed
	w = csv.writer(trade_file)	
	executions.insert(0, ['Time Left', 'Rounded Units Left', 'Bid Ask Spread', 'Volume Misbalance', 'Action', 'Reward', 'Volume'])
	w.writerows(executions)
	# write table
	tw = csv.writer(table_file)
	table_rows = []
	table_rows.append(['Time Left', 'Rounded Units Left', 'Bid Ask Spread', 'Volume Misbalance', 'Action', 'Expected Payout'])
	for key in table:
		for action, payoff in table[key].items():
			if type(action) != str:
				t_left, rounded_unit, spread, volume_misbalance = key.split(",")
				table_rows.append([t_left, rounded_unit, spread, volume_misbalance, action, payoff])
	tw.writerows(table_rows)

	
			
if __name__ == "__main__":
	dp_algo("../data/10_GOOG.csv", 1000, 1000, 10, 10, 10)
