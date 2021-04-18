#!/usr/bin/env python3

from random import uniform
from sys import argv, exit
import network_shared as net
import network_io_plot as io
import network_params as par

class Particle:
	"""Particle class.
	Containzerizes a position, velocity.

	Attributes:
		pos : the position in n-space.
		best_pos : the best position this particle has had.
		fit : the fitness of the particle.
		best_fit : the best fitness this particle has had.
		vel : the velocity in n-space.
	"""

	def __init__(self, pos):
		"""Particle constructor."""
		# initialize position and velocity
		self.pos, self.vel = pos, [0.00 for _ in range(len(pos))]
		# find fitness at instantiation
		network = net.initialize_network(self.pos, FEATURES, \
			HIDDEN_SIZE, CLASSES)
		self.fit = net.mse(network, CLASSES, TRAIN, activation_function)
		# best so far is just initial
		self.best_pos, self.best_fit = self.pos, self.fit

	def set_pos(self, pos):
		"""Position mutator method."""
		self.pos = pos
		if not any(p < -BOUND for p in pos)\
		and not any(p > BOUND for p in pos):
			# get fitness of new position
			network = net.initialize_network(self.pos, FEATURES, \
				HIDDEN_SIZE, CLASSES)
			fitness = net.mse(network, CLASSES, TRAIN, activation_function)
			# if better
			if fitness < self.best_fit:
				self.fit = fitness
				# update best fitness
				self.best_fit = self.fit
				# update best position
				self.best_pos = self.pos

	def set_vel(self, vel):
		"""Velocity mutator method."""
		self.vel = vel

	def get_pos(self):
		"""Position accessor method."""
		return self.pos

	def get_vel(self):
		"""Velocity accessor method."""
		return self.vel

	def get_best_pos(self):
		"""Best position accessor method."""
		return self.best_pos

	def get_fit(self):
		"""Fitness accessor method."""
		return self.fit

	def __getitem__(self, key):
		"""List index operator overload."""
		return self.pos[key]
	
	def __lt__(self, other):
		"""Less-than operator overload."""
		return self.fit < other.fit

def particle_swarm_optimization(dim, epochs, swarm_size, axis_range, w, c1, c2):
	"""Particle Network training function.
	Main driver for the PSO optimization of network weights.

	Parameters:
		dim : dimensionality of the problem.
		epochs : how many iterations.
		swarm_size : how big a swarm is.
		axis_range : the minimum and maximum value an axis may be.
		w : inertial coefficient (omega).
		c1 : cognitive coefficient (c_1).
		c2 : social coefficient (c_2).
	"""
	if not AUTO:
		print('Epoch, MSE, Train. Acc%, Test Acc%')
	# initialize swarm of solutions
	swarm = net.initialize_population(Particle, swarm_size, dim, axis_range)
	for e in range(1, epochs+1):
		swarm.sort() # sort swarm by fitness
		MSE.append(swarm[0].get_fit()) # get error of network using swarm best
		# network to get performance metrics on
		network = net.initialize_network(swarm[0].get_pos(), FEATURES, \
			HIDDEN_SIZE, CLASSES)
		# get classification error of network for training and test
		TRP.append(net.performance_measure(network, TRAIN, activation_function))
		TEP.append(net.performance_measure(network, TEST, activation_function))
		# reposition particles based on PSO params
		move_particles(swarm, swarm[0], dim, w, c1, c2)
		io.out_console(AUTO, e, MSE, TRP, TEP)

def move_particles(swarm, best, dim, ine_c, cog_c, soc_c):
	"""Particle movement function.

	Parameters:
		swarm : the swarm to move.
		best : the swarm best.
		dim : dimensionality of each particle.
		ine_c : inertial coefficient.
		cog_c : cognitive coefficient.
		soc_c : social coefficient.
	"""
	for particle in swarm: # for each particle
		# new position and velocity is initially zero
		new_pos = [0 for _ in range(dim)]
		new_vel = [0 for _ in range(dim)]
		for d in range(dim): # for each axis
			# this is split for readability but the update is based
			# on an addition of a weight, cognitive, and social term
			weight = ine_c * particle.get_vel()[d]
			cognitive = cog_c * uniform(0.00, 1.00)
			cognitive *= (particle.get_best_pos()[d] - particle.get_pos()[d])
			social = soc_c * uniform(0.00, 1.00)
			social *= (best[d] - particle.get_pos()[d])
			# new velocity is simply weight + cognitive + social
			new_vel[d] = weight + cognitive + social
			# new position is just old position + velocity
			new_pos[d] = particle.get_pos()[d] + new_vel[d]
		# update particle with new position and velocity
		particle.set_pos(new_pos)
		particle.set_vel(new_vel)

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
	# pso-specific parameters
	SWARM_SIZE = par.get_swarm_size()
	W, C_1, C_2, BOUND = par.get_pso_params(argv[1])
	# run the pso-nn
	particle_swarm_optimization(DIMENSIONS, EPOCHS, SWARM_SIZE, AXIS_RANGE, \
		W, C_1, C_2)
	if not AUTO:
		io.plot_data(EPOCHS, MSE, TRP, TEP)
	exit(0)
