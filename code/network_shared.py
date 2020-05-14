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
			summ = summing_function(neuron, layer_input)
			# activate the sum, append output to outputs
			layer_output.append(act(summ))
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
	for i in range(len(actual)):
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