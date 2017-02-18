from environment import *
from collections import defaultdict

def dp_algo(ob_file, H, V, I, T, L):
	c = {}
	env = Environment(ob_file, setup=False)
	steps = H / T
	vol_unit = V / I
	# state variables
	t = T
	inv = V
	market_cost = 999999
	volume_misbalance = 0
	# loop for the DP algorithm
	for t in range(0, T+1)[::-1]:
		# regenerate the order books
		env.get_timesteps(t, T+1)
		curr_book = env.get_next_state()
		import	pdb
		pdb.set_trace()
		#market_cost = compute_mc(curr_book)
		#volume_misbalance = compute_vm(curr_book)
		market_cost = 17
		volume_misbalance = 17		
		actions = sorted(curr_book.b.keys())
		for i in range(0, I + 1):
			for a in range(0, L + 1):
				curr_units = int(V/vol_unit) * i
				key = str(t) + "," + str(curr_units)+ "," +str(market_cost) + "," +str(volume_misbalance) 
				if t is T:
					spent, leftover = env.limit_order(1, 9999999, vol_unit * i)
					if leftover	>= 0:
						spent += leftover * actions[-1]
					c[key] = {}
					c[key][a] = spent
					print c
					break


					
			
if __name__ == "__main__":
	dp_algo("../LOBSTER_SampleFile_MSFT_2012-06-21_1/MSFT_2012-06-21_34200000_57600000_orderbook_1.csv", 10, 10000, 10, 10, 5)
