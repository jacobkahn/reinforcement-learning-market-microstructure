from environment_1_27 import *
import numpy as np
import csv
import pdb

if __name__ == "__main__":
	print ("hi")
	file = 'ETHUSD07_22.csv'
	builder = EpisodeBuilder(file)
	ep = builder.new_episode(0.2, 50, 50, 300, 10)
	print (ep.curr_decision_pt)
	print (ep.get_state().get_time())
	ep.trade(Action(100, Level(231, 0.1), "buy"))
	print (ep.curr_decision_pt)
	print (ep.get_state().get_time())
