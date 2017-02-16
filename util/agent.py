from environment import *
from collections import defaultdict

def dp_algo(ob_file, H, V, I, T):
	c = {}
	env = Environment(ob_file, setup=False)
	steps = H / T
	# state variables
	t = T
	inv = V
	market_cost = 999999
	volume_misbalance = 0
	# loop for the DP algorithm
	for i in range(0, T)[::-1]:
		# regenerate the order books
		for a in range(0, I):
			env.get_timesteps(i, T)
			for j in range(i, T):
				curr_book = env.get_next_state()
				market_cost = compute_mc(curr_book)
				volume_misbalance = compute_vm(curr_book)		
				key = str(t) + "," +str(inv)+ "," +str(market_cost) + "," +str(volume_misbalance)  
				if j == T:
					if key not in c:
						c[key] = {}




			



			
if __name__ == "__main__":
	dp_algo("../LOBSTER_SampleFile_MSFT_2012-06-21_1/MSFT_2012-06-21_34200000_57600000_orderbook_1.csv", 10, 10000, 10, 10	)
