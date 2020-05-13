import matplotlib.pyplot as plt
import pandas as pd

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

def helper(auto, epoch, mse, trp, tep):
	"""Helper function.
	Outputs to console performance.
	
	Parameters:
		auto : flag for automation script.
		epoch : the epoch to output data for.
		mse : mean squared error over epochs.
		trp : training accuracy over epochs.
		tep : testing accuracy over epochs.
	"""
	if not auto:
		print(f'{epoch}, {mse[-1]:.4f}, {trp[-1]:.2f}, {tep[-1]:.2f}')
	else:
		print(f'{mse[-1]:.4f}')

def load_data(filename):
	"""Loads CSV for splitting into training and testing data.
	
	Parameters:
		filename : the filename of the file to load.
	
	Returns:
		Two lists, each corresponding to training and testing data.
	"""
	# load into pandas dataframe
	df = pd.read_csv(filename, header=None, dtype=float)
	df = df.sample(100)
	# normalize the data
	for features in range(len(df.columns)-1):
		df[features] = (df[features] - df[features].mean())/df[features].std()
	train = df.sample(frac=0.70).fillna(0.00) # get training portion
	test = df.drop(train.index).fillna(0.00) # remainder testing portion
	return train.values.tolist(), test.values.tolist()

def plot_data(epochs, mse, trp, tep):
	"""Plots data.
	Displays MSE, training accuracy, and testing accuracy over time.
	
	Parameters:
		epochs : the number of training epochs to plot over.
		mse : the mean squared error over epochs.
		trp : the training accuracy over epochs.
		tep : the testing accuracy over epochs.
	"""
	x = range(0, epochs)
	fig, ax2 = plt.subplots()
	ax2.set_xlabel('Epoch')
	ax2.set_ylabel('MSE', color='blue')
	line, = ax2.plot(x, mse, '-', c='blue', lw='1', label='MSE')
	ax1 = ax2.twinx()
	ax1.set_ylabel('Accuracy (%)', color='green')
	line2, = ax1.plot(x, trp, '-', c='green', lw='1', label='Training')
	line3, = ax1.plot(x, tep, ':', c='green', lw='1', label='Testing')
	fig.legend(loc='center')
	ax1.set_ylim(0, 101)
	plt.show()
	plt.clf()