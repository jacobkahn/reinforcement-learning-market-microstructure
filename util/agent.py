from environment import *
from collections import defaultdict
import csv



def dp_algo(ob_file, H, V, I, T, L, S=1000, divs=5):
	table = {}
	env = Environment(ob_file, setup=False)
	all_books = len(env.books)
	steps = H / T
	# state variables
	t = T
	inv = V
	vol_unit = V/I
	volume_misbalance = 0
	# loop for the DP algorithm
	env.get_timesteps(0, S+1)	
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

	for t in range(0, T+1)[::-1]:
		print t
		# regenerate the order books
		for ts in range(0, S): 
			if ts % 1000 == 0:
				print ts
			curr_book = env.get_book(ts)
			spread = compute_bid_ask_spread(curr_book, spreads)
			volume_misbalance = compute_volume_misbalance(curr_book, misbalances, env)	
			actions = sorted(curr_book.a.keys())
			actions.append(0)
			for i in range(0, I + 1):
				for action in range(0, L):
					curr_book = env.get_book(ts)
					key = str(t) + "," + str(i)+ "," +str(spread) + "," +str(volume_misbalance) 
					if key not in table:
						table[key] = {}
					if t is T:
						if actions[action] == 0:
							break
						spent, leftover = env.limit_order(0, 9999999999, V/I * i)
						if leftover	>= 0:
							spent+= leftover * actions[-2]
						num_key = str(action)+ ',n'
						if num_key not in table[key]:
							table[key][num_key] = 0
						n = table[key][num_key]
						if action not in table[key]:
							table[key][action] = spent
						else:
							table[key][action] = float(n)/(n+1)*table[key][action] + float(1)/(n+1)*(spent)
						table[key][num_key] += 1
					else: 
						spent, leftover = env.limit_order(0, actions[action], V/I * i)
						rounded_unit = int(round(1.0 * leftover / vol_unit))
						next_key = str(t + 1) + "," + str(rounded_unit)+ "," + str(spread) + "," +str(volume_misbalance)
						arg_min = 99999999999999
						num_key = str(action)+ ',n'
						if num_key not in table[key]:
							table[key][num_key] = 0
						n = table[key][num_key]
						next_state = table[next_key]
						for k,v in next_state.items():
							if type(k) != str: 
								arg_min = v if arg_min > v else arg_min
						if action not in table[key]:
							table[key][action] = 0
						table[key][action] = float(n)/(n+1)*table[key][action] + float(1)/(n+1)*(spent + arg_min)
						table[key][num_key] += 1
	rwds, volumes, actions = execute_algo(table, env, H, T, V, I, 1000, spreads, misbalances)
	# write_model_files(table, rwds, volumes, actions, T)
	import pdb
	pdb.set_trace()

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

def execute_algo(table, env, H, T, V, I, steps, spreads, misbalances):
	rewards = []
	volume = V
	vol_unit = V/I
	time_unit = H/T
	decisions = steps / time_unit
	volumes = []
	ds = []
	for ts in range(0, decisions+1): 
		if ts % (T+1) == 0:
			env.get_timesteps(ts*time_unit, ts*time_unit+T*time_unit+1)
			volume = V
		rounded_unit = int(volume / vol_unit)
		t_left =  ts % (T + 1)
		perfect_price = env.mid_spread(ts - ts % (T + 1))
		curr_book = env.get_next_state()

		spread = compute_bid_ask_spread(curr_book, spreads)
		volume_misbalance = compute_volume_misbalance(curr_book, misbalances, env)	

		key = str(t_left) + ',' + str(rounded_unit) + ',' + str(spread) + ',' + str(volume_misbalance)

		actions = sorted(curr_book.a.keys())
		actions.append(0)
		print key + ' ' + str(ts)
		min_action = -1
		min_val = 999999999999
		for action, value in table[key].items():
			if type(action) != str:
				if value < min_val:
					min_val = value
					min_action = action
		ds.append(min_action)
		print min_action
		print 'initial ' + str(volume)
		paid, leftover = env.limit_order(0, actions[min_action], volume)
		if t_left == T:
			finish, clear = env.limit_order(0, 999999999999, leftover)
			paid += clear * actions[-2] + paid
			leftover = 0
		if min_action == len(actions) - 1 or (paid == 0 and leftover == volume):
			volumes.append(0)
			rewards.append('no trade ' + key)
			print 'leftover ' + str(volume)
			continue
		if volume != leftover:
			price_paid = paid / (volume - leftover)
			basis_p = (float(price_paid) - perfect_price)/perfect_price * 100
			reward = (basis_p, volume - leftover, key)
			rewards.append(reward)
			volumes.append(volume - leftover)
		volume = leftover
		print 'leftover ' + str(volume)
		if ts % T != 0:
			for i in range(0, time_unit - 1):
				env.get_next_state()	
	return rewards, volumes, ds


def write_model_files(table, rewards, volumes, decisions, T): 
	table_file = open("table.csv", 'w+')
	reward_
	w = csv.writer(out_file)
	to_write = []
	for t in range(0, T+1):
		to_write.append(rewards[t::T+1])
	w.writerows(to_write)






					
			
if __name__ == "__main__":
	dp_algo("../data/GOOG_2012-06-21_34200000_57600000_orderbook_10.csv", 100, 1000, 4, 4, 11)
