#!/usr/bin/env python3

from random import uniform
from sys import argv, exit
from math import exp
import network_shared as net
import network_io_plot as io
import network_params as par

def stochastic_gradient_descent(network, classes, training):
	"""Training function for neural network.
	Performs feed-forward, backpropagation, and update weight functions.

	Parameters:
		network : the neural network.
		classes : the number of classes for the data.
		training : data to train the network on.
	"""
	if not AUTO: # if normal execution
		print('Epoch, MSE, Train. Acc%, Test Acc%')
	for e in range(1, EPOCHS+1):
		# there is no temporal delta and therefore no momentum for the first
		# training example, so skip that step later for first example
		first_example = True
		total_error = 0.00
		for example in training:
			# skip if not first example; keep track of prior example delta
			temporal_delta = [neuron['d'] \
				for layer in network for neuron in layer] \
				if not first_example else None
			# create a list of possible outputs
			outputs = [0 for _ in range(classes)]
			outputs[int(example[-1])] = 1 # denote correct classification
			# get actual output from feed forward pass. Feeding forward will
			# also initialize network neuron outputs to be used in backprop
			actual = net.feed_forward(network, example, activation_function)
			total_error += net.sse(actual, outputs) # aggregate error
			# perform backpropagation to propagate error through network
			backpropagate(network, outputs)
			# update weights based on network params and neuron contents
			update_weights(network, example, temporal_delta)
			first_example = False # now we can consider momentum
		# append results for this epoch to global lists to make plots
		MSE.append(total_error/len(TRAIN))
		TRP.append(net.performance_measure(NETWORK, TRAIN, activation_function))
		TEP.append(net.performance_measure(NETWORK, TEST, activation_function))
		io.out_console(AUTO, e, MSE, TRP, TEP) # output to console

def backpropagate(network, example):
	"""Backpropagation function. Backpropagates error through network.

	Parameters:
		network : the neural network.
		example : a training example.
	"""
	for i in range(len(network)-1, -1, -1): # for each reverse order layer
		for j in range(len(network[i])): # for each neuron in the layer
			err = 0.00
			if i == len(network)-1: # if the output layer
				# error is a function of what the output is versus the target
				err = example[j] - network[i][j]['o']
			else: # if an inner layer
				summ = 0.00
				# error is the sum of neuron weights times their deltas
				for neuron in network[i+1]:
					summ += neuron['w'][j] * neuron['d']
				err = summ
			# delta is amount of correction
			network[i][j]['d'] = activation_derivative(network[i][j]['o']) * err

def update_weights(network, example, delta):
	"""Function to update network weights.

	Parameters:
		network : the neural network.
		example : a training example.
		delta : temporal delta.
	"""
	for i, _ in enumerate(network): # for each layer in network
		# we update the weights in order of layers 0..n
		# so we need either the example or the outputs of a layer
		if i != 0: # if not first layer
			# init t as neural outputs
			t = [neuron['o'] for neuron in network[i-1]]
		else: # if first layer
			t = example[:-1] # init t as the training example attributes
		# for each neuron in layer
		for d, neuron in enumerate(network[i]):
			for f, _ in enumerate(t): # for each feature or output of t
				# update weight based on learning rate term
				neuron['w'][f] += LEARNING_RATE * float(t[f]) * neuron['d']
				if delta is not None: # if there is a temporal delta
					# update weight based on momentum rate term
					neuron['w'][f] += MOMENTUM_RATE * delta[d]
				# also update neural bias
				neuron['w'][-1] += LEARNING_RATE * neuron['d']

def activation_function(z):
	"""Logistic Sigmoid function.

	Parameters:
		z : summed output of neuron.

	Returns:
		The neuron activation based on the summed output.
	"""
	return 1 / (1 + exp(-z))

def activation_derivative(z):
	"""Derivative of Logistic Sigmoid function.

	Parameters:
		z : summing output.

	Returns:
		The differential of the neural output.
	"""
	return z * (1 - z)

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
	# bp-specific parameters
	LEARNING_RATE, MOMENTUM_RATE = par.get_bp_params(argv[1])
	# network initialization
	WEIGHTS = [uniform(AXIS_RANGE[0], AXIS_RANGE[1]) for _ in range(DIMENSIONS)]
	NETWORK = net.initialize_network(WEIGHTS, FEATURES, HIDDEN_SIZE, CLASSES)
	# run the bp-nn
	stochastic_gradient_descent(NETWORK, CLASSES, TRAIN)
	if not AUTO:
		io.plot_data(EPOCHS, MSE, TRP, TEP)
	exit(0)
