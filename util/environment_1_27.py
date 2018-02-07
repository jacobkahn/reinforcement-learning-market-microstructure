import numpy as np
import csv


def num(l):
	return [float(x) for x in l]


class EpisodeBuilder:

# Using raw data an EpisodeBuilder is constructed, which contains a copy/source of truth of all raw data

	def __init__(self, raw_data, bid_price_col = 2, bid_vol_col = 3, ask_price_col = 0, ask_vol_col = 1, total_col = 4):
		file_stream = open(raw_data, 'r')
		books = csv.reader(file_stream, delimiter=',')
		self.raw_books = [book for book in books if len(book) > 0]
		self.raw_books = self.raw_books[1:]
		self.full_library = []
		for r, book in enumerate(self.raw_books):
			time = book[-1]
			book = book[:-1]
			bid_levels = [Level(price, vol) for price, vol in zip(num(book[bid_price_col::total_col]), num(book[bid_vol_col::total_col]))]
			ask_levels = [Level(price, vol) for price, vol in zip(num(book[ask_price_col::total_col]), num(book[ask_vol_col::total_col]))]
			new_book = Orderbook(r, time, bid_levels, ask_levels)
			self.full_library.append(new_book)

	#instantiate an episode
	def new_episode(self, start_vol, window_size, start_idx, end_idx, num_decision_pts):
		# copy in subset of full_library within the start and end times
		lib = [book for book in self.full_library if book.idx <= end_idx and book.idx >= start_idx - window_size]
		return Episode(start_vol, window_size, start_idx, end_idx, lib, num_decision_pts)


class Episode:
	def __init__(self, start_vol, window_size, start_idx, end_idx, ep_library, num_decision_pts):
		self.num_decision_pts = num_decision_pts
		self.start_idx, self.curr_idx = start_idx, start_idx
		self.end_idx = end_idx
		self.curr_decision_pt = 0
		self.ep_library = ep_library
		self.curr_vol = start_vol

		self.init_state = State([book for book in ep_library if book.idx <= start_idx and book.idx >= start_idx - window_size], num_decision_pts, self.curr_vol)
		self.curr_state = self.init_state
		self.diffs = {book.get_idx():book for book in self.get_time_steps(ep_library)}

	def get_state(self):
		return self.curr_state

	def get_time_steps(self, ob_vec):
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
		if self.curr_idx >= self.end_idx:
			self.curr_vol = 0
		else:
			for book in self.ep_library:
				# why would some idx not be in the diffs
				if book.get_idx() > self.curr_idx and book.get_idx() in self.diffs.keys():
					book.apply_diff(self.diffs[book.get_idx()])


		# update the state with the diffed books and new lookback window
		print (self.curr_vol)
		print (act.get_volume())
		self.curr_vol = self.curr_vol - act.get_volume()
		self.curr_idx += (self.end_idx - self.start_idx)/self.num_decision_pts
		self.curr_decision_pt += 1
		self.curr_state = State([book for book in self.ep_library if book.get_idx < self.curr_idx and book.get_idx >= self.curr_idx - window_size],
								self.end_idx - self.curr_idx, self.curr_vol)

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
		total = 0 #not returned right now
		price = action.get_price()
		volume = action.get_volume()
		side = action.get_side()
		if action.get_side() == "buy":
			for level in sorted(curr_book.get_asks(), key = lambda level: level.price):
				if volume == 0:
					return volume
				else:
					if level.get_price() <= price:
						# returns number left after you clear orderbook at this price
						left, ex_price = curr_book.order(side, price, volume)
						total += (volume-left) * ex_price
						volume = left
					else:
						return volume
			return volume
		if action.get_side() == "sell":
			# should this be sorted in opp direction?
			for level in sorted(curr_book.get_bids(), key = lambda level: level.price)[::-1]: # not sure what the -1 is for
				if volume == 0:
					return volume
				else:
					if level.get_price() >= price:
						# returns number left after you clear orderbook at this price
						left = curr_book.order(side, action)
						total += (volume-left) * ex_price
						volume = left
					else:
						return volume
				return volume

class Library:

	def __init__(self, books):
		self.books = books

class Orderbook:
	def __init__(self, idx, time, bids, asks):
		self.idx = idx
		self.bids = bids
		self.asks = asks
		self.time = time

	def get_bids(self):
		return self.bids
	def get_asks(self):
		return self.asks
	def get_time(self):
		return self.time
	def get_idx(self):
		return self.idx

	def order(self, side, price, volume):
		if side == "buy":
			for level in self.asks:
				if price >= level.get_price():
					ret = max(0, volume - level.get_volume())
					level.set_volume(max(level.get_volume() - volume, 0))
					return ret, level.get_price()
			return -1
		elif side == "sell":
			for level in self.bids:
				if act.get_price() <= level.get_price():
					ret = max(0, volume - level.get_volume())
					level.set_volume(max(level.get_price() - volume, 0))
					return ret, level.get_price()
			return -1
		else:
			print ("Invalid side code - OrderBook.order")
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
					new_a.append(Level(level_next.get_price(), level_next.get_volume() - level_this.get_volume()))
					missing = False
			if missing:
				new_a.append(level_next)
		for level_next in bids:
			missing = True
			for level_this in self.get_bids():
				if level_next.get_price() == level_this.get_price():
					new_b.append(Level(level_next.get_price(), level_next.get_volume() - level_this.get_volume()))
					missing = False
			if missing:
				new_b.append(level_next)
		return Orderbook(self.idx, self.time, new_b, new_a)

	def apply_diff(self, ob_next):
		new_a = [] #would it be faster to copy here
		new_b = []
		for level_next in self.get_asks():
			for level_this in ob_next.get_asks():
				vol = level_next.get_volume()
				if level_next.get_price() == level_this.get_price():
					vol = max(level_this.get_volume() + level_next.get_volume(), 0)
					break
			print(vol)
			new_a.append(Level(level_next.get_price(), vol))

		for level_next in self.get_bids():
			for level_this in ob_next.get_bids():
				vol = level_next.get_volume()
				if level_next.get_price() == level_this.get_price():
					vol = max(level_this.get_volume() + level_next.get_volume(), 0)
					break
			new_b.append(Level(level_next.get_price(), vol))


class Action:

	#_sides = ["buy", "sell"]

	def __init__(self, idx, level, side):
		self.idx = idx
		self.volume = level.get_volume()
		self.price = level.get_price()
		#if side in _sides:
		self.side = side

	def get_side(self):
		return self.side
	def get_volume(self):
		return self.volume
	def get_price(self):
		return self.price

class Level:

	def __init__(self, price, volume):
		self.price = price
		self.volume = volume
	def get_price(self):
		return self.price
	def get_volume(self):
		return self.volume
	def set_volume(self, volume):
		self.volume = volume
