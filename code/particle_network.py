#!/usr/bin/env python3

import random
from sys import argv, exit
import network_shared as shr
import network_io_plot as io
import network_params as net

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

	def __init__(self, pos, vel):
		"""Particle constructor."""
		# initialize position and velocity as params
		self.pos, self.vel = pos, vel
		# find fitness at instantiation
		network = initialize_network(self.pos)
		self.fit = shr.mse(network, CLASSES, TRAIN, activation_function)
		# best so far is just initial
		self.best_pos, self.best_fit = self.pos, self.fit

	def set_pos(self, pos):
		"""Position mutator method."""
		self.pos = pos
		if not any(p < -BOUND for p in pos)\
		and not any(p > BOUND for p in pos):
			# get fitness of new position
			network = initialize_network(self.pos)
			fitness = shr.mse(network, CLASSES, TRAIN, activation_function)
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

def pso(dim, epochs, swarm_size, ic, cc, sc):
	"""Particle Network training function
	Main driver for PSO algorithm

	Parameters:
		dim : dimensionality of the problem.
		epochs : how many iterations.
		swarm_size : how big a swarm is.
		ic : inertial coefficient (omega).
		cc : cognitive coefficient (c_1).
		sc : social coefficient (c_2).
	"""
	if not AUTO:
		print('Epoch, MSE, Train. Acc%, Test Acc%')
	swarm = initialize_swarm(swarm_size, dim) # init swarm
	for e in range(1, epochs+1):
		# get swarm best fitness and position
		swarm_best = get_swarm_best(swarm)
		MSE.append(swarm_best[0]) # get error of network using swarm best
		# network to get performance metrics on
		network = initialize_network(swarm_best[1])
		# get classification error of network for training and test
		TRP.append(shr.performance_measure(network, TRAIN, activation_function))
		TEP.append(shr.performance_measure(network, TEST, activation_function))
		# reposition particles based on PSO params
		move_particles(swarm, dim, ic, cc, sc)
		io.out_console(AUTO, e, MSE, TRP, TEP)

def move_particles(swarm, dim, ic, cc, sc):
	"""Particle movement function.

	Parameters:
		swarm : the swarm to move.
	"""
	# get swarm bests
	swarm_best = get_swarm_best(swarm)
	for particle in swarm: # for each particle
		# new position and velocity is initially zero
		new_pos = [0 for _ in range(dim)]
		new_vel = [0 for _ in range(dim)]
		for d in range(dim): # for each axis
			# this is split for readability but the update is based
			# on an addition of a weight, cognitive, and social term
			weight = ic * particle.get_vel()[d]
			cognitive = cc * random.uniform(0.00, 1.00)
			cognitive *= (particle.get_best_pos()[d] - particle.get_pos()[d])
			social = sc * random.uniform(0.00, 1.00)
			social *= (swarm_best[1][d] - particle.get_pos()[d])
			# new velocity is simply weight + cognitive + social
			new_vel[d] = weight + cognitive + social
			# new position is just old position + velocity
			new_pos[d] = particle.get_pos()[d] + new_vel[d]
		# update particle with new position and velocity
		particle.set_pos(new_pos)
		particle.set_vel(new_vel)

def initialize_swarm(size, dim):
	"""Swarm initialization function.

	Parameters:
		size : the size of our swarm.
		dim : the dimensionality of the problem.

	Returns:
		A random swarm of that many Particles.
	"""
	swarm = [] # swarm stored as list
	for _ in range(size): # for the size of the swarm
		# position is random in every dimension
		position = [random.uniform(-0.50, 0.50) for _ in range(dim)]
		# velocity is initially zero in every dimension
		velocity = [0 for _ in range(dim)]
		# init a particle
		particle = Particle(position, velocity)
		swarm.append(particle) # add to swarm
	return swarm

def get_swarm_best(swarm):
	"""Finds the swarm best fitness and position.

	Parameters:
		swarm : the swarm to search.

	Returns:
		The swarm best fitness and swarm best position.
	"""
	# initially assume the first is the best
	best_fit = swarm[0].get_fit()
	best_pos = swarm[0].get_pos()
	for particle in swarm: # for each particle
		# if better fitness found
		if particle.get_fit() < best_fit:
			# update best fitness and position
			best_fit = particle.get_fit()
			best_pos = particle.get_pos()
	return best_fit, best_pos

def initialize_network(p):
	"""Neural network initializer.

	Parameters:
		p : the particle to encode into the network.

	Returns:
		The n-h-o neural network.
	"""
	n, h, o = FEATURES, HIDDEN_SIZE, CLASSES
	part = iter(p) # make iterator from p
	neural_network = [] # initially an empty list
	# there are (n * h) connections between input layer and hidden layer
	neural_network.append([[next(part) for i in range(n+1)] for j in range(h)])
	# there are (h * o) connections between hidden layer and output layer
	neural_network.append([[next(part) for i in range(h+1)] for j in range(o)])
	return neural_network

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
	MSE, TRP, TEP = [], [], []
	TRAIN, TEST = io.load_data(f'../data/{argv[1]}.csv')
	FEATURES = len(TRAIN[0][:-1])
	CLASSES = len(list(set([c[-1] for c in (TRAIN+TEST)])))
	HIDDEN_SIZE = net.get_hidden_size(argv[1])
	DIMENSIONS = (HIDDEN_SIZE * (FEATURES+1)) + \
		(CLASSES * (HIDDEN_SIZE+1))
	SWARM_SIZE = net.get_swarm_size()
	EPOCHS = net.get_epochs()
	W, C_1, C_2, BOUND = net.get_pso_params(argv[1])
	pso(DIMENSIONS, EPOCHS, SWARM_SIZE, W, C_1, C_2)
	if not AUTO:
		io.plot_data(EPOCHS, MSE, TRP, TEP)
	exit(0)
