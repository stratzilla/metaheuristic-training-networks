#!/usr/bin/env python3

from sys import argv, exit
import random
import network_shared as shr
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
		self.fit = mse(network)
		# best so far is just initial
		self.best_pos, self.best_fit = self.pos, self.fit

	def set_pos(self, pos):
		"""Position mutator method."""
		self.pos = pos
		if not any(p < -BOUND for p in pos)\
		and not any(p > BOUND for p in pos):
			# get fitness of new position
			network = initialize_network(self.pos)
			fitness = mse(network)
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
		# network from swarm best
		network = initialize_network(swarm_best[1])
		# get classification error of network for training and test
		TRP.append(performance_measure(network, TRAIN))
		TEP.append(performance_measure(network, TEST))
		# reposition particles based on PSO params
		move_particles(swarm, dim, ic, cc, sc)
		shr.helper(e, MSE, TRP, TEP, AUTO)

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

def feed_forward(network, example):
	"""Feedforward method. Feeds data forward through network.

	Parameters:
		network : the neural network.
		example : an example of data to feed forward.

	Returns:
		The output of the forward pass.
	"""
	layer_input, layer_output = example, []
	for layer in network:
		for neuron in layer:
			# sum the weight with inputs
			summ = summing_function(neuron, layer_input)
			# activate the sum, append output to outputs
			layer_output.append(activation_function(summ))
		# inputs become outputs of previous layer
		layer_input, layer_output = layer_output, []
	return layer_input # return the final output

def summing_function(weights, inputs):
	"""Sums the synapse weights with inputs and bias.

	Parameters:
		weights : synaptic weights.
		inputs : a vector of inputs.

	Returns:
		The aggregate of inputs times weights, plus bias.
	"""
	# bias is the final value in the weight vector
	bias = weights[-1]
	summ = 0.00 # to sum
	for i in range(len(weights)-1):
		# aggregate the weights with input values
		summ += (weights[i] * float(inputs[i]))
	return summ + bias

def activation_function(z):
	"""ReLU activation function.

	Parameters:
		z : summed output of neuron.

	Returns:
		The neuron activation based on the summed output.
	"""
	return z if z >= 0 else 0.01 * z

def performance_measure(network, data):
	"""Measures accuracy of the network using classification error.

	Parameters:
		network : the network to test.
		data : a set of data examples.

	Returns:
		A percentage of correct classifications.
	"""
	correct, total = 0, 0
	for example in data:
		# check to see if the network output matches target output
		if check_output(network, example) == float(example[-1]):
			correct += 1
		total += 1
	return 100*(correct / total)

def check_output(network, example):
	"""Compares network output to actual output.

	Parameters:
		network : the neural network.
		example : an example of data.

	Returns:
		The class the example belongs to (based on network guess).
	"""
	output = feed_forward(network, example)
	return output.index(max(output))

def sse(actual, target):
	"""Sum of Squared Error.

	Parameters:
		actual : network output.
		target : example target output.

	Returns:
		The sum of squared error of the network for an example.
	"""
	summ = 0.00
	for i in range(len(actual)):
		summ += (actual[i] - target[i])**2
	return summ

def mse(network):
	"""Mean Squared Error.

	Parameters:
		network : the neural network to test.
	"""
	training = TRAIN
	summ = 0.00
	# for each training example
	for example in training:
		# populate a target vector
		target = [0 for _ in range(CLASSES)]
		# denote correct classification
		target[int(example[-1])] = 1
		# get actual output by feeding example through network
		actual = feed_forward(network, example)
		# sum up the sum of squared error
		summ += sse(actual, target)
	# MSE is just sum(sse)/number of examples
	return summ / len(training)

if __name__ == '__main__':
	# if executed from automation script
	if len(argv) == 3:
		AUTO = bool(int(argv[2]))
	else:
		AUTO = False
	MSE, TRP, TEP = [], [], []
	TRAIN, TEST = shr.load_data(f'../data/{argv[1]}.csv')
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
		shr.plot_data(EPOCHS, MSE, TRP, TEP)
	exit(0)
