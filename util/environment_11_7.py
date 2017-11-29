import numpy as np
import csv


def num(l):
	return [int(x) for x in l]


class EpisodeBuilder:

# Using raw data an EpisodeBuilder is constructed, which contains a copy/source of truth of all raw data

	def __init__(self, raw_data, bid_price_col = 2, bid_vol_col = 3, ask_price_col = 0, ask_vol_col = 1, time_col, total_col = 4):
		file_stream = open(raw_data, 'r')
		books = csv.reader(file_stream, delimiter=',')
		self.raw_books = [book for book in raw_data]
		self.raw_books = self.raw_books[1:]
		self.full_library = []
		for i in raw_books:
			 book = raw_books[i]
			 bid_levels = Level(num(book[bid_price_col::total_col]), num(book[bid_vol_col::total_col]))
			 ask_levels = Level(num(book[ask_price_col::total_col]), num(book[ask_vol_col::total_col]))
			 new_book = Orderbook(num(book[time_col::total_col]), bid_levels, ask_levels)
			 self.full_library.append(new_book)

	#instantiate an episode
	def new_episode(start_time, start_vol, window_size, start_time, end_time, ep_library, num_decision_pts):
		# copy in subset of full_library within the start and end times
		lib = [book for book in full_library if book.get_time <= end_time & book.get_time >= start_time - window_size]
		return Episode(start_time, start_vol, window_size, start_time, end_time, lib, num_decision_pts)


class Episode:
	def __init__(self, start_vol, window_size, start_time, end_time, ep_library, num_decision_pts):
		self.num_decision_pts = num_decision_pts
		self.start_time, self.curr_time = start_time
		self.end_time = end_time
		self.curr_decision_pt = 0
		self.ep_library = ep_library
		self.curr_vol = start_vol
		self.init_state = State([book for book in ep_library if book.get_time <= start_time & book.get_time >= start_time - window_size], end_time - start_time, self.curr_vol)
		self.curr_state = self.init_state
		self.diffs = get_time_steps(ep_library)

	def get_state(self):
		return self.curr_state

	def get_time_steps(ob_vec):
		#what is signed volume?
		time_steps = []
		for book in ob_vec:
			if len(time_steps) > 0:
				ob = curr_book.diff(book.get_asks(), book.get_bids())
				time_steps.append(ob)
				curr_book.apply_diff(time_steps[-1])
			else:
				curr_book = book
				time_steps.append(book)
		return time_steps


	def trade(self, act):
		#should this return any cost info?
		# apply the action
		self.curr_vol = self.curr_state.limit_order(act)
		if self.curr_time >= self.end_time:
			self.curr_vol = 0
		else:
#time_steps contain diff info?
			for book in self.ep_library if book.get_time() >= self.curr_time:
				book.apply_diff([diff for diff in self.diffs if diff.get_time() == book.get_time()])

		# update the state with the diffed books and new lookback window
		self.curr_vol -= act.get_volume
		self.curr_time += (self.end_time - self.start_time)/self.num_decision_pts
		self.curr_decision_pt += 1
		self.curr_state = State([book for book in ep_library if book.get_time < self.curr_time & book.get_time >= self.curr_time - window_size],
								self.end_time - self.curr_time, self.curr_vol)

class State:

	def __init__(self, lookback_window, time_remaining, inventory_remaining):
		self.lookback_window = lookback_window
		self.time_remaining = time_remaining
		self.inventory_remaining = inventory_remaining

	def get_lookback(self):
		return self.lookback_window
	def get_time(self):
		return self.time_remaining
	def get_inventory(self):
		return self.inventory_remaining
	def get_last_price(self):
		return self.look_backwindow[-1].get_price()

	def limit_order(self, action):
		curr_book = self.lookback_window[-1]
		total = 0
		price = action.get_price()
		volume = action.get_volume()
		if action.get_side() == "buy":
			for level in sorted(curr_book.get_asks(), key = get_price):
				if volume == 0:
					return (total, volume)
				else:
					if level.get_price() <= price:
						# returns number left after you clear orderbook at this price
						left = curr_book.order(side, action)
						total += (volume-left) * pr
						volume = left
					else:
						return (total, volume)
			return total, volume
		if action.get_side() == "sell":
			# should this be sorted in opp direction?
			for level in sorted(curr_book.get_bids(), key = get_price)[::-1]: # not sure what the -1 is for
				if volume == 0:
					return (total, level.get_volume())
				else:
					if level.get_price() >= price:
						# returns number left after you clear orderbook at this price
						left = curr_book.order(side, action)
						total += (volume-left) * pr
						volume = left
					else:
						return (total, volume)
				return total, volume

class Library:

	def __init__(self, books):
		self.books = books

class Orderbook:
	def __init__(self, time, bids, asks):
		self.bids = bids
		self.asks = asks
		self.time = time

	def get_bids(self):
		return self.bids
	def get_asks(self):
		return self.asks
	def get_time(self):
		return self.time

	def order(self, side, act):
		if side == "buy":
			for level in self.asks:
				if act.get_price() == level.get_price():
					ret = max(0, act.get_volume() - level.get_price())
					level.set_price(max(level.get_price() - volume, 0))
				return ret
			return -1
		elif side == "sell":
			for level in self.bids:
				if act.get_price() == level.get_price():
					ret = max(0, act.get_volume - level.get_price())
					level.set_price(max(level.get_price() - volume, 0))
				return ret
			return -1
		else:
			print "Invalid side code - OrderBook.order"
			return -2



	def clean_book(self):
		for level in self.bids:
			if level.get_volume == 0:
				del self.bids[level]
		for level in self.asks:
			if level.get_volume == 0:
				del self.asks[level]

	def diff(self, asks, bids,):
		net_vol = 0
		new_a = []
		new_b = []
		for level_next in asks:
			missing = True
			for level_this in self.get_asks():
				if level_next.get_price() == level_this.get_price():
					new_a.append(Level(level_next.get_price(), level_next.get_volume - level_this.get_volume)
					missing = False
			if missing:
				new_a.append(level_next)
		for level_next in bids:
			missing = True
			for level_this in self.get_bids():
				if level_next.get_price() == level_this.get_price():
					new_b.append(Level(level_next.get_price(), level_next.get_volume - level_this.get_volume)
					#net_vol	+= level_next.get_volume - level_this.get_volume
					missing = False
			if missing:
				new_b.append(level_next)
		return OrderBook(self.time, new_b, new_a)

	def apply_diff(self, ob_next):
		new_a = []
		new_b = []
		for level_next in ob_next.get_asks():
			missing = True
			for level_this in self.get_asks():
				if level_next.get_price() == level_this.get_price():
					v = level_this.get_price() + level_next.get_price()
					missing = False
			 	if missing:
					v = level_this.get_price()
				new_a.append() = Level(max(v, 0), level_this.get_volume())
		for level_next in ob_next.get_bids():
			missing = True
			for level_this in self.get_bids():
				if level_next.get_price() == level_this.get_price():
					v = level_this.get_price() + level_next.get_price()
					missing = False
			 	if missing:
					v = level_this.get_price()
				new_b.append() = Level(max(v, 0), level_this.get_volume())
		self.asks = new_a
		self.bids = new_b

class Action:

	_sides = ["buy", "sell"] #whoops use 0 and 1??

	def __init__(self, time, level, side):
		self.time = time
		self.volume = volume
		if side in _sides
			self.side = side
		# prob not idiomatic

	def get_side(self):
		return self.side
	def get_volume(self):
		return self.volume

class Level:

	def __init__(self, price, volume):
		self.price = price
		self.level = volume
	def get_price(self):
		return self.price
	def get_volume(self):
		return self.volume
	def set_price(self, price):
		self.price = price
