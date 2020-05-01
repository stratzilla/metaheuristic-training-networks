#!/usr/bin/env python3

import pandas as pd
from math import ceil
import matplotlib.pyplot as plt
import random
from sys import argv, exit
import network_params as net

def helper(e):
	"""Helper function.
	Outputs to console performance.
	"""
	if not AUTO:
		err = MSE[-1]
		tr = TRP[-1]
		te = TEP[-1]
		print(f'{e}, {err:.4f}, {tr:.2f}, {te:.2f}')
	else:
		err = MSE[-1]
		print(f'{err:.4f}')

class Chromosome:
	"""Chromosome class.
	Containerizes genes for chromosome.
	
	Attributes:
		genes : the weights of the network.
		fit : the fitness of the chromosome.
	"""
	
	def __init__(self, genes, fit=None):
		"""Chromosome constructor without fitness."""
		# initialize weights from parameter
		self.genes = genes
		# if no argument passed as fitness
		# take fitness from genes argument
		# else init as fit argument
		if fit is None:
			network = initialize_network(self.genes)
			self.fit = mse(network)
		else:
			self.fit = fit
	
	def set_genes(self, genes):
		"""Genes mutator method."""
		self.genes = genes
		# when setting genes subsequent times
		# update the fitness
		network = initialize_network(self.genes)
		self.fit = mse(network)
	
	def get_genes(self):
		"""Genes accessor method."""
		return self.genes
	
	def get_fit(self):
		"""Fitness accessor method."""
		return self.fit

	def __lt__(self, other):
		"""Less-than operator overload."""
		return self.fit < other.fit
	
	def __getitem__(self, key):
		"""List index operator overload."""
		return self.genes[key]
	
	def __len__(self):
		"""List length operator overload."""
		return len(self.genes)

def genetic_network(el_p, to_p, dim, epochs, pop_size, cr, mr):
	"""Genetic Neural Network training function.
	
	Parameters:
		el_p : the proportion of elites
		to_p : the proportion of tournament
		dim : dimensionality of network.
		epochs : how many generations to run.
		pop_size : the population size.
		cr : crossover rate.
		mr : mutation rate.
	
	Returns:
		A trained neural network.
	"""
	if not AUTO:
		print('Epoch, MSE, Train. Acc%, Test Acc%')
	# initialize network as initially random
	population = initialize_population(pop_size, dim)
	for e in range(1, epochs+1):
		# sort the population by fitness
		population.sort()
		# get fitness of network
		MSE.append(population[0].get_fit())
		# training accuracy of network
		TRP.append(performance_measure(population[0].get_genes(), TRAIN))
		# testing accuracy of network
		TEP.append(performance_measure(population[0].get_genes(), TEST))
		mating_pool = [] # init mating pool
		# get elites from population
		elites = elite_selection(population, el_p)
		del population[:len(elites)] # remove elites
		# find tournament and winner
		t_winner = tournament_selection(population, to_p)
		# add tournament victor and elites to mating pool
		mating_pool.extend(elites)
		mating_pool.append(t_winner)
		# generate a new population based on mating pool
		population = evolve(mating_pool, elites, pop_size, cr, mr)
		mating_pool.clear() # clear mating pool for next gen
		helper(e)

def evolve(mating_pool, elites, pop_size, cr, mr):
	"""Evolves population based on genetic operators.
	
	Parameters:
		mating_pool : where to select parents from.
		elites : previously found elites.
		pop_size : the population size.
		cr : crossover rate.
		mr : mutation rate.
	
	Returns:
		A new population of offspring from mating pool.
	"""
	new_population = [] # store new population as list
	new_population += elites # add elites verbatim
	while len(new_population) < pop_size: # while population isn't at max size
		# get both parents indices
		p_a_idx = random.randrange(len(mating_pool))
		p_b_idx = random.randrange(len(mating_pool))
		# we don't mind parents having identical genes but we don't
		# want the parents to use the same index. Parent A can be
		# equal to Parent B, but Parent A cannot be Parent B
		if p_a_idx == p_b_idx:
			continue
		# get the parents from indices
		parent_a = mating_pool[p_a_idx]
		parent_b = mating_pool[p_b_idx]
		# find children using crossover
		child_a, child_b = crossover(parent_a, parent_b, cr)
		# mutate each child
		child_a = mutation(child_a, mr)
		child_b = mutation(child_b, mr)
		# add children to population
		new_population += [child_a, child_b]
	return new_population

def crossover(parent_a, parent_b, cr):
	"""Two-point crossover operator.
	
	Parameters:
		parent_a : the first parent.
		parent_b : the second parent.
		cr : the crossover chance.
	
	Returns:
		Two child chromosomes as a product of both parents.
	"""
	# only perform crossover based on the crossover rate
	if random.uniform(0.00, 1.00) >= cr:
		child_a = Chromosome(parent_a.get_genes(), parent_a.get_fit())
		child_b = Chromosome(parent_b.get_genes(), parent_b.get_fit())
		return child_a, child_b
	genes_a, genes_b = [], []
	# find pivot points at random 1..n-1
	pivot_a = random.randint(1, len(parent_a)-1)
	# second pivot is between pivot_a..n-1
	pivot_b = random.randint(pivot_a, len(parent_a)-1)
	for i in range(0, len(parent_a)):
		# before first pivot, use genes from one parent
		if i < pivot_a:
			genes_a.append(parent_a[i])
			genes_b.append(parent_b[i])
		# before second pivot, use genes from second parent
		elif i < pivot_b:
			genes_a.append(parent_b[i])
			genes_b.append(parent_a[i])
		# after second pivot, use genes from first parent again
		else:
			genes_a.append(parent_a[i])
			genes_b.append(parent_b[i])
	return Chromosome(genes_a), Chromosome(genes_b)

def mutation(child, mr):
	"""Mutation operator.
	
	Parameters:
		child : the chromosome to mutate.
		mr : the mutation chance.
	
	Returns:
		A mutated child.
	"""
	# the new genes to make
	genes = [gene for gene in child.get_genes()]
	for i in range(len(genes)):
		# only perform mutation based on the mutation rate
		if random.uniform(0.00, 1.00) <= mr:
			# update that axes with random position
			genes[i] = random.gauss(mu=genes[i], sigma=(BASE + child.get_fit()))
	# we don't need to update the fitness if the gene
	# hasn't changed, so only update genes if they've changed
	if genes != child.get_genes():
		child.set_genes(genes)
	return child

def initialize_population(size, dim):
	"""Initializes a random population.
	
	Parameters:
		size : the size of the population.
		dim : the dimensionality of the problem
	
	Returns:
		A random population of that many points.
	"""
	population = [] # population stored as a list
	for _ in range(size): # for the size of the population
		genes = [random.uniform(-0.50, 0.50) for _ in range(dim)] # random genes
		chromosome = Chromosome(genes) # create the chromosome
		population.append(chromosome) # add to population
	return population

def elite_selection(population, percent):
	"""Elite selection function.
	Stores elites to bring into the next generation and mating pool.
	
	Parameters:
		population : the population to take elites from.
		percent : the proportion of the population to consider elites.
	
	Returns:
		A list of elite solutions.
	"""
	elites = []
	# grab percent% best individuals
	for i in range(ceil(len(population)*percent)):
		elites.append(population[i]) # and append to elites
	return elites
	
def tournament_selection(population, percent):
	"""Tournament selection function.
	Creates a tournament of random individuals and returns the best.
	
	Parameters:
		population : the population to take tournament from.
		percent : the proportion of the population who enters the tournament.
	
	Returns:
		Best fit individual from tournament.
	"""
	tournament = []
	# grab percent% random individuals
	for i in range(ceil(len(population)*percent)):
		random_idx = random.randint(0, len(population)-1)
		tournament.append(population.pop(random_idx)) # append to tournament
	tournament.sort() # sort by fitness
	return tournament[0] # return best fit from tournament

def initialize_network(c):
	"""Neural network initializer.
	
	Parameters:
		c : the chromosome to encode into the network.
	
	Returns:
		The n-h-o neural network.
	"""
	n, h, o = FEATURES, HIDDEN_SIZE, CLASSES
	chr = iter(c) # make iterator from c
	neural_network = [] # initially an empty list
	# there are (n * h) connections between input layer and hidden layer
	neural_network.append([[next(chr) for i in range(n+1)] for j in range(h)])
	# there are (h * o) connections between hidden layer and output layer
	neural_network.append([[next(chr) for i in range(h+1)] for j in range(o)])
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

def performance_measure(chromosome, data):
	"""Measures accuracy of the network using classification error.
	
	Parameters:
		chromosome : the chromosome to test.
		data : a set of data examples.
	Returns:
		A percentage of correct classifications.
	"""
	network = initialize_network(chromosome)
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

def load_data(filename):
	"""Loads CSV for splitting into training and testing data.
	
	Parameters:
		filename : the filename of the file to load.
	
	Returns:
		Two lists, each corresponding to training and testing data.
	"""
	# load into pandas dataframe
	df = pd.read_csv(filename, header=None, dtype=float)
	# normalize the data
	for features in range(len(df.columns)-1):
		df[features] = (df[features] - df[features].mean())/df[features].std()
	train = df.sample(frac=0.70).fillna(0.00) # get training portion
	test = df.drop(train.index).fillna(0.00) # remainder testing portion
	return train.values.tolist(), test.values.tolist()

def plot_data():
	"""Plots data.
	Displays MSE, training accuracy, and testing accuracy over time.
	"""
	x = range(0, EPOCHS)
	fig, ax2 = plt.subplots()
	ax2.set_xlabel('Epoch')
	ax2.set_ylabel('MSE', color='blue')
	line, = ax2.plot(x, MSE, '-', c='blue', lw='1', label='MSE')
	ax1 = ax2.twinx()
	ax1.set_ylabel('Accuracy (%)', color='green')
	line2, = ax1.plot(x, TRP, '-', c='green', lw='1', label='Training')
	line3, = ax1.plot(x, TEP, ':', c='green', lw='1', label='Testing')
	fig.legend(loc='center')
	ax1.set_ylim(0, 101)
	plt.title(f'GA-NN ({argv[1]})')
	plt.show()
	plt.clf()

if __name__ == '__main__':
	# if executed from automation script
	if len(argv) == 3:
		AUTO = bool(int(argv[2]))
	else:
		AUTO = False
	MSE, TRP, TEP = [], [], []
	TRAIN, TEST = load_data(f'../data/{argv[1]}.csv')
	FEATURES = len(TRAIN[0][:-1])
	CLASSES = len(list(set([c[-1] for c in (TRAIN+TEST)])))
	HIDDEN_SIZE = net.get_hidden_size(argv[1])
	CHROMOSOME_SIZE = (HIDDEN_SIZE * (FEATURES+1)) + \
		(CLASSES * (HIDDEN_SIZE+1))
	POP_SIZE = net.get_population_size()
	CROSS_RATE, MUTAT_RATE, ELITE_PROPORTION, \
		TOURN_PROPORTION, BASE = net.get_ga_params(argv[1])
	EPOCHS = net.get_epochs()
	genetic_network(ELITE_PROPORTION, TOURN_PROPORTION, \
		CHROMOSOME_SIZE, EPOCHS, POP_SIZE, CROSS_RATE, MUTAT_RATE)
	if not AUTO:
		plot_data()
	exit(0)