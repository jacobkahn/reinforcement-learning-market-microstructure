from environment import *
from collections import defaultdict

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
		# regenerate the order books
		for ts in range(0, 100): 
			curr_book = env.get_book(ts)
			# market_cost = compute_mc(curr_book)
			# volume_misbalance = compute_vm(curr_book)
			market_cost = 17
			volume_misbalance = 17		
			actions = sorted(curr_book.a.keys())
			actions.append(0)
			perfect_price = env.mid_spread(max(ts-t, 0))
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
	execute_algo(table, env, T, V, I, 20)
	import pdb
	pdb.set_trace()

def execute_algo(table, env, T, V, I, steps):
	reward = 0
	volume = V
	vol_unit = V/I
	for ts in range(0, steps+1):
		if ts % (T+1) == 0:
			env.get_timesteps(ts, ts+T+1)
			volume = V
		rounded_unit = int(volume / vol_unit)
		t_left = T - ts % (T + 1)
		key = str(t_left) + ',' + str(rounded_unit) + ',' +'17,17'
		curr_book = env.get_next_state()
		print key + ' ' + str(ts)
		print curr_book.b
		min_action = -1
		min_val = -1
		for action, value in table[key]:
			if type(k) != str:
				if min_val < value:
					





					
			
if __name__ == "__main__":
	dp_algo("../LOBSTER_SampleFile_MSFT_2012-06-21_1/MSFT_2012-06-21_34200000_57600000_orderbook_1.csv", 10, 10000, 10, 10, 2)
