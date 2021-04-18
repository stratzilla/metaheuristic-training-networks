#!/usr/bin/env python3

from random import uniform, gauss
from sys import argv, exit
from math import exp
import network_shared as net
import network_io_plot as io
import network_params as par

class Bat:
	"""Bat class.
	Containerizes a position, velocity.

	Attributes:
		pos : the position in n-space.
		vel : the velocity of the bat.
		fit : the fitness of the bat.
		loudness : the bat echolocation loudness.
		pulse_rate : the rate of echolocation.
		maximum_pulse_rate : the highest value pulse_can can be.
	"""

	def __init__(self, pos):
		"""Bat constructor."""
		self.pos, self.vel = pos, [0.00 for _ in range(len(pos))]
		self.loudness = uniform(1, 2) # loudness is some random value 1..2
		self.max_pulse_rate = uniform(0, 1) # max pulse rate varies per bat
		self.pulse_rate = 0 # initially pulse rate is 0 and climbs to max
		# find fitness at instantiation
		network = net.initialize_network(self.pos, FEATURES, \
			HIDDEN_SIZE, CLASSES)
		self.fit = net.mse(network, CLASSES, TRAIN, activation_function)

	def set_pos(self, pos):
		"""Position mutator method."""
		self.pos = pos

	def set_vel(self, vel):
		"""Velocity mutator method."""
		self.vel = vel

	def set_fit(self, fit):
		"""Fitness mutator method."""
		self.fit = fit

	def set_loudness(self, loudness):
		"""Loudness mutator method."""
		self.loudness = loudness

	def set_pulse_rate(self, pulse_rate):
		"""Pulse rate mutator method."""
		self.pulse_rate = pulse_rate

	def get_pos(self):
		"""Position accessor method."""
		return self.pos

	def get_vel(self):
		"""Velocity accessor method."""
		return self.vel

	def get_fit(self):
		"""Fitness accessor method."""
		return self.fit

	def get_loudness(self):
		"""Loudness accessor method."""
		return self.loudness

	def get_pulse_rate(self):
		"""Pulse rate accessor method."""
		return self.pulse_rate

	def get_max_pulse_rate(self):
		"""Max pulse rate accessor method."""
		return self.max_pulse_rate

	def __lt__(self, other):
		"""Less-than operator overload."""
		return self.fit < other.fit

	def __getitem__(self, key):
		"""List index operator overload."""
		return self.pos[key]

	def __len__(self):
		"""List length operator overload."""
		return len(self.pos)

def bat_algorithm(dim, epochs, pop_size, axis_range, alf, gam, bnd, qmin, qmax):
	"""Differential evolution training function.
	Main driver for the BA optimization of network weights.

	Parameters:
		dim : the dimensionality of network.
		epochs : how many generations to run.
		pop_size : the population size.
		alf : loudness decreasing rate.
		gam : pulse rate increasing rate.
		bnd : boundary to clamp position.
		qmin : minimum frequency.
		qmax : maximum frequency.
	"""
	if not AUTO:
		print('Epoch, MSE, Train. Acc%, Test Acc%')
	# initialize the network as initially random
	population = net.initialize_population(Bat, pop_size, dim, axis_range)
	for e in range(1, epochs+1):
		population.sort() # sort the population by fitness
		MSE.append(population[0].get_fit()) # get fitness of best network
		# make network to get performance metrics
		network = net.initialize_network(population[0].get_pos(), \
			FEATURES, HIDDEN_SIZE, CLASSES)
		# training accuracy of network
		TRP.append(net.performance_measure(network, TRAIN, activation_function))
		# testing accuracy of network
		TEP.append(net.performance_measure(network, TEST, activation_function))
		step = float(e)/epochs # how many epochs have elapsed
		# move each bat in population
		population = move_bats(population, dim, qmin, qmax, alf, gam, bnd, step)
		io.out_console(AUTO, e, MSE, TRP, TEP)

def move_bats(population, dim, qmin, qmax, alf, gam, bnd, step):
	"""Bat movement function.

	Parameters:
		population : the population of bats to move.
		dim : the dimensionality of the problem.
		qmin : minimum value for frequency.
		qmax : maximum value for frequency.
		alf : loudness decreasing rate.
		gam : pulse rate increasing rate.
		bnd : boundary to clamp position.
		step : function of how many epochs have elapsed over max epochs.

	Returns:
		The bat population after movement.
	"""
	best_pos = population[0].get_pos() # population best position
	best_fit = population[0].get_fit() # population best fitness
	for bat in population:
		# new position and velocity is initially zero
		new_pos = [0.00 for _ in range(dim)]
		new_vel = [0.00 for _ in range(dim)]
		average_loudness = sum(a.get_loudness() for a in population)
		average_loudness /= len(population)
		#print(average_loudness)
		freq = uniform(qmin, qmax) # find a random frequency for bat
		pulse_chance = uniform(0, 1) # chance for bat to move closer to best
		for d in range(dim): # for each axis of position
			# calculate new velocity as a function of old and distance to best
			new_vel[d] = bat.get_vel()[d] + (bat[d] - best_pos[d]) * freq
			# if chance is in favor, make local solution around best
			if pulse_chance > bat.get_pulse_rate():
				# found by random walk around best solution
				new_pos[d] = best_pos[d] + (gauss(0, 1) * 0.001)
			else: # otherwise, new position function of old and velocity
				new_pos[d] = bat[d] + new_vel[d]
			# clamp position to remain within boundaries
			new_pos[d] = min(max(new_pos[d], -bnd), bnd)
		bat.set_vel(new_vel) # set the bat's new velocity
		new_bat_fit = Bat(new_pos).get_fit() # propose new solution
		# if better than the current bat
		if (new_bat_fit <= bat.get_fit()) and \
			(uniform(0, 1) < bat.get_loudness()):
			bat.set_pos(new_pos) # set bat position to the new position
			bat.set_fit(new_bat_fit) # update fitness as well
			bat.set_loudness(bat.get_loudness() * alf) # decay loudness
			bat.set_pulse_rate(bat.get_max_pulse_rate() * \
				(1 - exp(-gam * step))) # increase pulse rate
		if new_bat_fit <= best_fit: # update best bat if needed
			best_pos = new_pos
			best_fit = new_bat_fit
	return population

def activation_function(z):
	"""ReLU activation function.

	Parameters:
		z : summed output of neuron.

	Returns:
		The neuron activation based on the summed output.
	"""
	return z if z >= 0 else 0.01 * z

if __name__ == '__main__':
	# if executed from automation script
	if len(argv) == 3:
		AUTO = bool(int(argv[2]))
	else:
		AUTO = False
	MSE, TRP, TEP = [], [], [] # set up variables to store testing data
	# load data to train and test network on
	TRAIN, TEST = io.load_data(f'../data/{argv[1]}.csv', par.get_holdout())
	# network-specific parameters
	FEATURES = len(TRAIN[0][:-1]) # number of attributes of data
	CLASSES = len({c[-1] for c in TRAIN+TEST}) # distinct classifications
	HIDDEN_SIZE = par.get_hidden_size(argv[1])
	DIMENSIONS = (HIDDEN_SIZE * (FEATURES+1)) + (CLASSES * (HIDDEN_SIZE+1))
	EPOCHS, AXIS_RANGE = par.get_epochs(), par.get_rand_range()
	# ba-specific parameters
	POP_SIZE = par.get_ba_population_size()
	FREQ_MIN, FREQ_MAX, BOUND, ALPHA, GAMMA = par.get_ba_params(argv[1])
	# run the ba-nn
	bat_algorithm(DIMENSIONS, EPOCHS, POP_SIZE, AXIS_RANGE, ALPHA, GAMMA, \
		BOUND, FREQ_MIN, FREQ_MAX)
	if not AUTO:
		io.plot_data(EPOCHS, MSE, TRP, TEP)
	exit(0)
