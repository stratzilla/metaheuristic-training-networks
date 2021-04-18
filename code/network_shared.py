from random import uniform

def feed_forward(network, example, act):
	"""Feedforward method. Feeds data forward through network.

	Parameters:
		network : the neural network.
		example : an example of data to feed forward.
		act : activation function.

	Returns:
		The output of the forward pass.
	"""
	layer_input, layer_output = example, []
	for layer in network:
		for neuron in layer:
			# sum the weight with inputs
			summ = summing_function(neuron['w'], layer_input)
			# activate the sum, store output
			neuron['o'] = act(summ)
			# append output to outputs
			layer_output.append(neuron['o'])
		# inputs become outputs of previous layer
		layer_input, layer_output = layer_output, []
	return layer_input

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

def mse(network, classes, training, act):
	"""Mean Squared Error.

	Parameters:
		network : the neural network to test.
		classes : number of unique instance classes.
		training : training data to find MSE upon.
		act : activation function.
	"""
	summ = 0.00
	# for each training example
	for example in training:
		# populate a target vector
		target = [0 for _ in range(classes)]
		# denote correct classification
		target[int(example[-1])] = 1
		# get actual output by feeding example through network
		actual = feed_forward(network, example, act)
		# sum up the sum of squared error
		summ += sse(actual, target)
	# MSE is just sum(sse)/number of examples
	return summ / len(training)

def sse(actual, target):
	"""Sum of Squared Error.

	Parameters:
		actual : network output.
		target : example target output.

	Returns:
		The sum of squared error of the network for an example.
	"""
	summ = 0.00
	for i, _ in enumerate(actual):
		summ += (actual[i] - target[i])**2
	return summ

def performance_measure(network, data, act):
	"""Measures accuracy of the network using classification error.

	Parameters:
		network : the network to test.
		data : a set of data examples.
		act : the activation function.

	Returns:
		A percentage of correct classifications.
	"""
	correct, total = 0, 0
	for example in data:
		# check to see if the network output matches target output
		if check_output(network, example, act) == float(example[-1]):
			correct += 1
		total += 1
	return 100*(correct / total)

def check_output(network, example, act):
	"""Compares network output to actual output.

	Parameters:
		network : the neural network.
		example : an example of data.
		act : the activation function.

	Returns:
		The class the example belongs to (based on network guess).
	"""
	output = feed_forward(network, example, act)
	return output.index(max(output))

def initialize_population(agent_type, size, dim, axis_range):
	"""Initializes a random population.

	Parameters:
		agent_type : the type of agent to initialize a population of.
		size : the size of the population.
		dim : the dimensionality of the problem.
		axis_range : the minimum and maximum value an axis can be.

	Returns:
		A random population of that many agents.
	"""
	population = [] # population stored as a list
	for _ in range(size): # for the size of the population
		# randomly uniform position in all axes
		pos = [uniform(axis_range[0], axis_range[1]) for _ in range(dim)]
		agent = agent_type(pos) # create new agent from position
		population.append(agent) # append to population
	return population

def initialize_network(weights, n, h, o):
	"""Neural network initializer.
	The network will be structured as nested data structures, namely a list of
	lists of dicts. As the algorithm continues, not only the weights will be
	stored but also deltas, outputs, errors.

	Parameters:
		weights : the weights to initialize network as.
		n : the number of input neurons.
		h : the number of hidden neurons.
		o : the number of output neurons.

	Returns:
		An n-h-o neural network as a list of list of dicts.
	"""
	w = iter(weights)
	neural_network = [] # initially an empty list
	# there are (n * h) connections between input layer and hidden layer
	# a 'w' will denote weights
	neural_network.append([{'w':[next(w) for i in range(n+1)]} \
		for j in range(h)])
	# there are (h * o) connections between hidden layer and output layer
	neural_network.append([{'w':[next(w) for i in range(h+1)]} \
		for j in range(o)])
	return neural_network
