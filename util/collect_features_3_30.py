from environment_1_27 import *
import numpy as np
import csv
import pdb
import pandas as pd
#import matplotlib.pyplot as plt
import sklearn
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import scipy
from scipy.stats import zscore

if __name__ == "__main__":
	print ("hi")
	file = 'gemini-snapshot.csv'
	builder = EpisodeBuilder(file)
	ep = builder.new_episode(200, 50, 50, 580663, 580613) # (self, start_vol, window_size, start_idx, end_idx, num_decision_pts)
	spreads = []
	imbalance = []
	imm_imbalance = []
	avolchange = []
	bvolchange = []
	smartpr = []
#200, 50, 50, 580663, 580613
	print("collecting features")
	for i in range(ep.start_idx, ep.end_idx):
		ep.trade(i, 250, 0, "buy")
		lookback = ep.get_state().get_lookback()
		spreads_window = []
		imbalance_window = []
		best_imbalance_window = []
		avolchange_window = []
		bvolchange_window = []
		imm_imbalance_window = []
		smartpr_window = []
		avol = 0
		bvol = 0
		prevavg = 0
		for book in lookback:
			a = book.get_asks()
			b = book.get_bids()
			spreads_window.append(a[0].get_price() - b[0].get_price())
			imbalance_window.append((book.vol_bids() - book.vol_asks())/(book.vol_bids() + book.vol_asks()))
			imm_imbalance_window.append(a[0].get_volume() - b[0].get_volume())
			#for alev, blev in zip(a, b):
			#	avg += 1/book.vol_bids()/book.vol_bids(inv = True)*blev.get_price() + 1/book.vol_asks()/book.vol_asks(inv = True)*alev.get_price()
			avg = 1/a[0].get_volume()/(1/a[0].get_volume()+1/b[0].get_volume())*a[0].get_price() + 1/b[0].get_volume()/(1/a[0].get_volume()+1/b[0].get_volume())*b[0].get_price()
			smartpr_window.append(avg - prevavg)
			avolchange_window.append(book.vol_asks() - avol)
			bvolchange_window.append(book.vol_bids() - bvol)
			prevavg = avg
			avol = book.vol_asks()
			bvol = book.vol_bids()

		spreads.append(np.mean(spreads_window))
		imbalance.append(np.mean(imbalance_window))
		imm_imbalance.append(np.mean(imm_imbalance_window))
		smartpr.append(np.mean(smartpr_window))
		avolchange.append(np.mean(avolchange_window[1:]))
		bvolchange.append(np.mean(bvolchange_window[1:]))
	spreads_m = np.mean(spreads)
	spreads_sd = np.std(spreads)
	imbalance_m = np.mean(imbalance)
	imbalance_sd = np.std(imbalance)
	spreads_norm = [(s-spreads_m)/spreads_sd for s in spreads]
	imbalance_norm = [(i-imbalance_m)/imbalance_m for i in imbalance]

	print("collecting returns")
	vol_per_trade = .1
	returns = []
	bcost = ep.trade(0, 1000, vol_per_trade, "buy")
	for i in range(1, ep.num_decision_pts):
		scost = ep.trade(i, 100, vol_per_trade, "sell")
		returns.append(scost - bcost)
		bcost = ep.trade(i, 1000, vol_per_trade, "buy")
	scost = ep.trade(ep.end_idx, 1000, vol_per_trade, "sell")
	returns.append(scost - bcost)

	df = pd.DataFrame.from_items([('returns', returns), ('spreads', spreads), ('imbalance', imbalance), ('imm_imbalance', imm_imbalance), ('smartpr', smartpr), ('avolchange', avolchange), ('bvolchange', bvolchange)])
	df = df.apply(zscore)
	print(df.head)
	df_test = df[:-29000]
	df_train = df[-29000:]
	X_train = df_train.drop('returns', axis = 1)
	X_test = df_test.drop('returns', axis = 1)

	lm = linear_model.LinearRegression().fit(X_train, df_train.returns)
	print(lm.coef_)

	pred = lm.predict(X_train)
	print(r2_score(df_train.returns, pred))


	# plt.scatter(spreads_norm, returns)
	# plt.show()
	#
	# plt.scatter(imbalance_norm, returns)
	# plt.show()

	#sci-kit learn - bid-ask spread, vol misbalance - run a simulation where calculate bid-ask spreads for lookback windows
	# plot features and returns from trianing set
	#if make more than threshold, then make trade
	#in addition to forward return, predict price change over term structure (predict 1, 2, 3, etc. min the future)
