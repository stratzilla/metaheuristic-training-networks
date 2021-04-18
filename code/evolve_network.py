#!/usr/bin/env python3

from random import sample, randint, uniform
from sys import argv, exit
import network_shared as net
import network_io_plot as io
import network_params as par

class Solution:
	"""Solution class.
	Containerizes a position within a solution/agent.

	Attributes:
		pos : the position of the agent.
		fit : the fitness of the agent.
	"""
	def __init__(self, pos):
		"""Solution constructor."""
		self.pos = pos
		# initialize solution as a network to check fitness, since fitness is
		# a function of feedforwarding training examples
		network = net.initialize_network(self.pos, FEATURES, \
			HIDDEN_SIZE, CLASSES)
		self.fit = net.mse(network, CLASSES, TRAIN, activation_function)

	def get_pos(self):
		"""Position accessor method."""
		return self.pos

	def get_fit(self):
		"""Fitness accessor method."""
		return self.fit

	def __lt__(self, other):
		"""Less-than operator overload."""
		return self.fit < other.fit

	def __le__(self, other):
		"""Less-than-or-equal operator overload."""
		return self.fit <= other.fit

	def __getitem__(self, key):
		"""List index operator overload."""
		return self.pos[key]

	def __len__(self):
		"""List length operator overload."""
		return len(self.pos)

def differential_evolution(dim, epochs, pop_size, axis_range, cr, dw):
	"""Differential evolution training function.
	Main driver for the DE optimization of network weights.

	Parameters:
		dim : the dimensionality of network.
		epochs : how many generations to run.
		pop_size : the population size.
		axis_range : the minimum and maximum values for a given axis.
		cr : crossover rate.
		dw : differential weight.
	"""
	if not AUTO:
		print('Epoch, MSE, Train. Acc%, Test Acc%')
	# initialize network as initially random
	population = net.initialize_population(Solution, pop_size, dim, axis_range)
	for e in range(1, epochs+1):
		population.sort() # sort population by fitness
		MSE.append(population[0].get_fit()) # get fitness of best network
		# change most fit solution to a network to test performance
		network = net.initialize_network(population[0].get_pos(), \
			FEATURES, HIDDEN_SIZE, CLASSES)
		# training accuracy of network
		TRP.append(net.performance_measure(network, TRAIN, activation_function))
		# testing accuracy of network
		TEP.append(net.performance_measure(network, TEST, activation_function))
		# evolve population based on differential evolution rules
		population = evolve(population, dim, cr, dw)
		io.out_console(AUTO, e, MSE, TRP, TEP)

def evolve(population, dim, cr, dw):
	"""Evolves population based on DE arithmetic.

	Parameters:
		population : the original population.
		dim : dimensionality of the problem.
		cr : crossover rate.
		dw : differential weight.

	Returns:
		A new population of updated positions using DE.
	"""
	new_population = []
	for i, base in enumerate(population): # find a base agent
		# and three other distinct agents
		[a, b, c] = sample(population[:i]+population[i+1:], 3)
		# random dimension to assuredly mutate
		random_idx = randint(0, dim)
		new_pos = [] # construct an empty position
		for j in range(dim): # then iteratively fill it
			# if crossover occurs based on crossover rate, OR the dimension is
			# the assured dimension to mutate as before
			if uniform(0, 1) < cr or j == random_idx:
				# find a position in the current axis based on the three agents
				# found before, as well as differential weight
				new_pos.append(a[j] + (dw * (b[j] - c[j])))
			else:
				new_pos.append(base[j]) # if no crossover, position is verbatim
		possible_update = Solution(new_pos) # turn position into an agent
		# if the position is equal or better than the base agent, update it
		if possible_update <= base:
			new_population.append(possible_update)
		else:
			new_population.append(base)
	return new_population

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
	DIMENSIONS = (HIDDEN_SIZE * (FEATURES+1)) +	(CLASSES * (HIDDEN_SIZE+1))
	EPOCHS, AXIS_RANGE = par.get_epochs(), par.get_rand_range()
	# de-specific parameters
	POP_SIZE = par.get_de_population_size()
	CROSS_RATE, DIFF_WEIGHT = par.get_de_params(argv[1])
	# run the de-nn
	differential_evolution(DIMENSIONS, EPOCHS, POP_SIZE, AXIS_RANGE, \
		CROSS_RATE, DIFF_WEIGHT)
	if not AUTO:
		io.plot_data(EPOCHS, MSE, TRP, TEP)
	exit(0)
