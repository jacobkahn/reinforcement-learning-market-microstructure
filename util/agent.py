from environment import *
from collections import defaultdict
import csv

def dp_algo(ob_file, H, V, I, T, L):
	table = {}
	env = Environment(ob_file, setup=False)
	all_books = len(env.books)
	steps = H / T
	# state variables
	t = T
	inv = V
	vol_unit = V/I
	market_cost = 999999
	volume_misbalance = 0
	# loop for the DP algorithm
	for t in range(0, T+1)[::-1]:
		print t
		# regenerate the order books
		for ts in range(0, 10000): 
			curr_book = env.get_book(ts)
			# market_cost = compute_mc(curr_book)
			# volume_misbalance = compute_vm(curr_book)
			market_cost = 17
			volume_misbalance = 17		
			actions = sorted(curr_book.a.keys())
			actions.append(0)
			for i in range(0, I + 1):
				for action in range(0, L):
					curr_book = env.get_book(ts)
					key = str(t) + "," + str(i)+ "," +str(market_cost) + "," +str(volume_misbalance) 
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
						rounded_unit = int(leftover / vol_unit)
						# market_cost = compute_mc(curr_book)
						# volume_misbalance = compute_vm(curr_book)
						market_cost = 17
						volume_misbalance = 17
						next_key = str(t + 1) + "," + str(rounded_unit)+ "," +str(market_cost) + "," +str(volume_misbalance)
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
	rwds = execute_algo(table, env, H, T, V, I, 30000)
	out_file = open("dp_summary1.csv", 'w+')
	w = csv.writer(out_file)
	to_write = []
	for t in range(0, T+1):
		to_write.append(rwds[t::11])
	w.writerows(to_write)

def execute_algo(table, env, H, T, V, I, steps):
	rewards = []
	volume = V
	vol_unit = V/I
	time_unit = H/T
	decisions = steps / time_unit
	for ts in range(0, decisions+1): 
		if ts % (T+1) == 0:
			env.get_timesteps(ts, ts+T*time_unit+1)
			volume = V
		rounded_unit = int(volume / vol_unit)
		t_left =  ts % (T + 1)
		perfect_price = env.mid_spread(ts - ts % (T + 1))
		key = str(t_left) + ',' + str(rounded_unit) + ',' +'17,17'
		curr_book = env.get_next_state()
		actions = sorted(curr_book.a.keys())
		actions.append(0)
		print key + ' ' + str(ts)
		print curr_book.a
		min_action = -1
		min_val = 999999999999
		for action, value in table[key].items():
			if type(action) != str:
				if value < min_val:
					min_val = value
					min_action = action
		print min_action
		print 'initial ' + str(volume)
		paid, leftover = env.limit_order(0, actions[min_action], volume)
		if min_action == len(actions) - 1 or (paid == 0 and leftover == volume):
			rewards.append("no trade")
			print 'leftover ' + str(volume)
			continue
		if t_left == T:
			paid += leftover * actions[-2]
			leftover = 0
		if volume != leftover:
			price_paid = paid / (volume - leftover)
			basis_p = (float(price_paid) - perfect_price)/perfect_price * 100
			reward = (basis_p, volume - leftover)
			rewards.append(basis_p)
		volume = leftover
		print 'leftover ' + str(volume)
		for i in range(0, time_unit - 1):
			env.get_next_state()	
	return rewards






					
			
if __name__ == "__main__":
	dp_algo("../LOBSTER_SampleFile_MSFT_2012-06-21_1/MSFT_2012-06-21_34200000_57600000_orderbook_1.csv", 100, 10000, 10, 10, 2)
