from environment import *
from collections import defaultdict
from Q import *
from Qfunction_approx import *
import csv # for reading
import sys
import multiprocess # for multithreading

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

def dp_algo(ob_file, H, V, I, T, L, backup, S, divs, test_steps, func_approx=None, envs=None, test_env=None):
	print 'HI'
	if func_approx is not None:
		table = Q_Function(T, L, backup)
	else:
		table = Q(T,L, backup)
	if envs is None:
		envs = Environment(ob_file, setup=False)
	steps = H / T
	# number of timesteps in between decisions
	time_unit = H/T

	# state variables

	t = T
	inv = V
	vol_unit = V/I
	volume_misbalance = 0


	if isinstance(envs, dict):
		days = envs.keys()
		dividers = {}
		for day in days:
			env = envs[day]
			vols = env.get_timesteps(0, S+1, I, V)
			spreads, misbalances, imm_costs, signed_vols = create_variable_divs(divs, env)
			dividers[day] = (vols, spreads, misbalances, imm_costs, signed_vols)
	else:
		env = envs
		vols = env.get_timesteps(0, S+1, I, V)
		spreads, misbalances, imm_costs, signed_vols = create_variable_divs(divs, env)

	
	# loop for the DP algorithm
	for t in range(0, T+1)[::-1]:
		print t
		for ts in range(0, S):
			if isinstance(envs, dict):
				days = envs.keys()
				day = random.choice(days)
				env = envs[day]
				vols, spreads, misbalances, imm_costs, signed_vols = dividers[day]
			else:
				env = envs
			if ts % 1000 == 0:
				print ts
			tgt_price = env.mid_spread(ts + time_unit * (T- t))
			curr_book = env.get_book(ts)
			spread = compute_bid_ask_spread(curr_book, spreads)
			volume_misbalance = compute_volume_misbalance(curr_book, misbalances, env)
			signed_vol = compute_signed_vol(vols[ts], signed_vols)
			actions = sorted(curr_book.a.keys())
			actions.append(0)
			for i in range(0, I + 1):
				for action in range(0, L+1):
					# regenerate the order book so that we don't have the effects of the last action
					curr_book = env.get_book(ts)
					immediate_cost = compute_imm_cost(curr_book, i*vol_unit, imm_costs)
					table.update_table_buy(t, i, vol_unit, spread, volume_misbalance, immediate_cost, signed_vol, action, actions, env, tgt_price)
	if test_env is None:
		test_env = envs
	executions = execute_algo(table, test_env, H, V, I, T, test_steps, spreads, misbalances, imm_costs, signed_vols)
	process_output(table, func_approx, executions, T, L)


def process_output(table, func, executions, T, L):
	"""
	Process output for each run and write to file
	"""
	if table.backup['name'] == 'sampling' or table.backup['name'] == 'replay buffer':
		table_to_write = table.Q
	elif table.backup['name'] == 'doubleQ':
		if func is None:
			table_to_write = table.curr_Q
		else:
			table_to_write = []
			table_to_write.append(table.Q_1)
			table_to_write.append(table.Q_2)
	else:
		print 'agent.dp_algo - invalid backup method'

	tradesOutputFilename = ''
	if not func is None:
		# linear approx model
		tradesOutputFilename += 'linear-'

	tradesOutputFilename += table.backup['name']
	write_trades(executions, tradesOutputFilename=tradesOutputFilename)

	if func is None:
		write_table_files(table_to_write, T, L, tableOutputFilename=table.backup['name'])
	else:
		write_function(table_to_write, T, L, 'linear model',functionFilename=table.backup['name'])

def create_variable_divs(divs, env):
	spreads = []
	misbalances = []
	imm_costs = []
	signed_vols	= []
	if divs > 1:
		spread_diff = (env.max_spread - env.min_spread) * 1.0 / (divs)
		misbalance_diff = (env.max_misbalance - env.min_misbalance) * 1.0 / (divs)
		imm_cost_diff = (env.max_imm_cost - env.min_imm_cost) * 1.0 / (divs)
		signed_vols_diff = (env.max_signed_vol - env.min_signed_vol) * 1.0 / (divs)
		for i in range(1, divs):
			spreads.append(env.min_spread + i * spread_diff)
			misbalances.append(env.min_misbalance + i * misbalance_diff)
			imm_costs.append(env.min_imm_cost + i * imm_cost_diff)
			signed_vols.append(env.min_signed_vol + i * signed_vols_diff)
	spreads.sort()
	misbalances.sort()
	imm_costs.sort()
	signed_vols.sort()
	return spreads, misbalances, imm_costs, signed_vols

def compute_signed_vol(vol, signed_vols):
	if len(signed_vols) == 0 or vol < signed_vols[0]:
		return 0
	for i in range(len(signed_vols) - 1):
		if vol >= signed_vols[i] and vol < signed_vols[i+1]:
			return (i + 1)
	return len(signed_vols)

def compute_imm_cost(curr_book, inv, im_costs):
	if inv == 0:
		return 0
	im_cost = curr_book.immediate_cost_buy(inv)
	if len(im_costs) == 0 or im_cost < im_costs[0]:
		return 0
	for i in range(len(im_costs) - 1):
		if im_cost >= im_costs[i] and im_cost < im_costs[i+1]:
			return (i + 1)
	return len(im_costs)

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
		if m >= misbalances[i] and  m < misbalances[i+1]:
			return (i + 1)
	return len(misbalances)

def execute_algo(table, env, H, V, I, T, steps, spreads, misbalances, imm_costs, signed_vols):
	# remaining volume and list of trades made by algorithm
	executions = []
	volume = V
	# number of shares per unit of inventory
	vol_unit = V/I
	# number of timesteps in between decisions
	time_unit = H/T
	# number of decisions possible during test steps set
	decisions = steps / time_unit - 1
	for x in range(10):
		offset = random.randint(0, time_unit - 1)
		for ts in range(x, decisions+1):
			# regenerate orderbook simulation for the next time horizon of decisions
			if ts % (T+1) == 0:
				env.get_timesteps(ts*time_unit + offset, ts*time_unit+T*time_unit+ offset + 1, T, V)
				volume = V
			# update the state of the algorithm based on the current book and timestep
			rounded_unit = int(volume / vol_unit)
			t_left =  ts % (T + 1)
			curr_book = env.get_next_state()
			# ideal price is mid-spread end of the period
			perfect_price = env.mid_spread(ts*time_unit + time_unit * (T- t_left))

			spread = compute_bid_ask_spread(curr_book, spreads)
			volume_misbalance = compute_volume_misbalance(curr_book, misbalances, env)
			immediate_cost = compute_imm_cost(curr_book, volume, imm_costs)
			signed_vol = compute_signed_vol(env.running_vol, signed_vols)


			actions = sorted(curr_book.a.keys())
			actions.append(0)
			# compute and execute the next action using the table
			min_action, _ = table.greedy_action(t_left, rounded_unit, spread, volume_misbalance, immediate_cost, signed_vol, ts)
			paid, leftover = env.limit_order(0, actions[min_action], volume)
			print min_action
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
				reward = [t_left, rounded_unit, spread, volume_misbalance, immediate_cost, signed_vol, min_action, basis_p, volume - leftover]
				print str(perfect_price) + ' ' + str(price_paid)
			executions.append(reward)
			volume = leftover
			# simulate market till next decision point - no need to simulate after last decision point
			if ts % T != 0:
				for i in range(0, time_unit - 1):
					env.get_next_state()
	return executions


def write_trades(executions, tradesOutputFilename="run"):
	trade_file = open(tradesOutputFilename + '-trades.csv', 'wb')
	# write trades executed
	w = csv.writer(trade_file)
	executions.insert(0, ['Time Left', 'Rounded Units Left', 'Bid Ask Spread', 'Volume Misbalance', 'Immediate Cost', 'Signed Transcation Volume' ,'Action', 'Reward', 'Volume'])
	w.writerows(executions)

def write_function(function, T, L, model,  functionFilename='run'):
	table_file = open(functionFilename + '-' + model + '.csv', 'wb')
	tw = csv.writer(table_file)
	table_rows = []
	table_rows.append(['Time Left', 'Rounded Units Left', 'Bid Ask Spread', 'Volume Misbalance', 'Immediate Cost', 'Signed Transcation Volume','Action'])
	if type(function) is list:
		table_rows.append(function[0].coef_)
		table_rows.append(function[1].coef_)
		table_rows.append(function[0].intercept_)
		table_rows.append(function[1].intercept_)
	else:
		table_rows.append(function.coef_)
		table_rows.append(function.intercept_)
	tw.writerows(table_rows)

def write_table_files(table, T, L, tableOutputFilename="run"):
	table_file = open(tableOutputFilename + '-tables.csv', 'wb')
	# write table
	tw = csv.writer(table_file)
	table_rows = []
	table_rows.append(['Time Left', 'Rounded Units Left', 'Bid Ask Spread', 'Volume Misbalance', 'Immediate Cost', 'Signed Transcation Volume', 'Action', 'Expected Payout'])
	for key in table:
		for action, payoff in table[key].items():
			if type(action) != str:
				t_left, rounded_unit, spread, volume_misbalance, im_cost, signed_vol = key.split(",")
				table_rows.append([t_left, rounded_unit, spread, volume_misbalance,  im_cost, signed_vol, action, payoff])
	tw.writerows(table_rows)


"""
We here run three backup methods based on how dp tables are updated:
- sampling (simple update)
- double q learning
- replay buffer
"""
if __name__ == "__main__":
	# define method params
	doubleQbackup = {
		'name': 'doubleQ'
	}
	samplingBackup = {
		'name': 'sampling'
	}
	replayBufferBackup = { 'name': 'replay buffer',
					'buff_size': 50,
					'replays': 5
	}
	# tables
	doubleQProcess = multiprocess.Process(target=dp_algo, args=("../data/10_GOOG.csv", 1000, 1000, 10, 10, 10, doubleQbackup, 100000))
	samplingProcess = multiprocess.Process(target=dp_algo, args=("../data/10_GOOG.csv", 1000, 1000, 10, 10, 10, samplingBackup, 100000))
	replayBufferProcess = multiprocess.Process(target=dp_algo, args=("../data/10_GOOG.csv", 1000, 1000, 10, 10, 10, replayBufferBackup, 100000))
	# start
	#doubleQProcess.start()
	samplingProcess.start()
	#replayBufferProcess.start()

	# function approx
	#func_doubleQProcess = multiprocess.Process(target=dp_algo, args=("../data/10_GOOG.csv", 1000, 1000, 10, 10, 10, doubleQbackup, "linear", 100000))
	#func_samplingProcess = multiprocess.Process(target=dp_algo, args=("../data/10_GOOG.csv", 1000, 1000, 10, 10, 10, samplingBackup, "linear", 100000))
	#func_replayBufferProcess = multiprocess.Process(target=dp_algo, args=("../data/10_GOOG.csv", 1000, 1000, 10, 10, 10, replayBufferBackup, "linear", 100000))
	# start
	#func_doubleQProcess.start()
	#func_samplingProcess.start()
	#func_replayBufferProcess.start()
