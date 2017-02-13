import numpy as np
import csv


class Environment:

	def __init__(self, orderbook_file, setup=True):
		file_stream = open(orderbook_file, 'r')
		books = csv.reader(file, ',')
		self.books = [book for book in books]
		if setup:
			self.get_timesteps(0, len(books))

	# generates the correct environment from timesteps start to end-1
	def get_timesteps(self, start, end):
		if start < 0 or end > len(self.books):
			print "Timesteps out of bounds!"
			return
		self.current_timestep = 0
		self.time_steps = []
		for i in range(start, end):
			book = self.books[i]
			ask_prices = book[0::4]
			ask_volumes = book[1::4]
			bid_prices = book[2::4]
			bid_volumes = book[3::4]
			if len(self.time_steps) > 0:
				ob = self.time_steps[-1].diff(ask_prices, ask_volumes, bid_prices, bid_volumes)
				self.time_steps.append(ob)
			else:
				ob = OrderBook(ask_prices, ask_volumes, bid_prices, bid_volumes)
				self.time_steps.append(ob)

	# returns orderbook of next state: after the first orderbook this only provides diffs
	def get_next_state(self);
		if self.current_timestep >= len(self.time_steps):
			print "Simulation Over"
			return OrderBook([],[],[],[])
		else:
			ret = self.time_steps[self.current_timestep]
			self.current_timestep += 1
			return ret

	# returns total price paid or received and volume left
	def limit_order(self, side, price, volume):
		curr_book = self.time_steps[self.current_timestep]
		# 0 is buy, 1 is sell
		total = 0
		if side == 0:
			for p, v in sorted(curr_book.a.items()):
				if volume == 0:
					return (total, 0)
				else:
					if p <= price:
						# returns number left after you clear orderbook at this price
						left = curr_book.order(side, p, volume)
						volume = left
					else:
						return (total, volume)
		if side == 1:
			for p, v in sorted(curr_book.b.items())[::-1]:
				if volume == 0:
					return (total, 0)
				else:
					if p >= price:
						# returns number left after you clear orderbook at this price
						left = curr_book.order(side, p, volume)
						volume = left
					else:
						return (total, volume)

class OrderBook:

	def __init__(self, asks, ask_vols, bids, bid_vols):
		self.a = {}
		for i in range(len(asks)):
			a[asks[i]] = ask_vols[i]
		self.b = {}
		for i in range(len(asks)) :
			a[bids[i]] = bid_vols[i]

	'''
	Assumes this is orderbook at step t, takes info for step t+1,
	creates a new order book with the additional or fewer shares now offered
	at existing price levels as well as quantities at new prices levels. 
	Generated BEFORE orders are submitted by the agent: these are the 
	actual changes from the market. 
	'''
	def diff(self, asks, ask_vols, bids, bid_vols):
		new_a = []
		new_av = []
		new_b = []
		new_bv = []
		for i in range(len(asks)):
			if asks[i] in self.a:
				new_a.append(asks[i])
				new_av.append(ask_vols[i] - self.a[asks[i]])
			else:
				new_a.append(asks[i])
				new_av.append(ask_vols[i])
			if bids[i] in self.b:
				new_b.append(bids[i])
				new_bv.append(bid_vols[i] - self.b[bids[i]])
			else:
				new_b.append(bids[i])
				new_bv.append(bid_vols[i])
		return OrderBook(new_a, new _av, new_b, new_bv)

	'''
	Takes difference orderbook and applies it to this one.
	Allows agent's orders to be processed without losing the actual
	changes in the market between the order books -- UNDER CONSTRUCTION
	'''
	def apply_diff(self, ob_next):
		a = {}
		for price, volume in ob_next.a:
			a[price] = 

	def clean_book(self):
		# clean up gone price levels
		for price, vol in self.a:
			if vol == 0:
				del self.a[price]
		for price, vol in self.b:
			if vol == 0:
				del self.b[price]


	def order(self, side, price, volume):
		# add proper error handling eventually
		# 0 is buy, 1 is sell
		if side == 0:
			if price in self.a: 
				ret = max(0, volume - self.a[price])
				self.a[price] = max(self.a[price] - volume, 0)
				self.clean_book()
				return ret
			return -1
		elif side == 1:
			if price in self.b: 
				ret = max(0, volume - self.b[price])
				self.b[price] = max(self.b[price] - volume, 0)
				self.clean_book()
				return ret
			return -1
		else:
			print "Invalid side code - OrderBook.order"
			return -2


