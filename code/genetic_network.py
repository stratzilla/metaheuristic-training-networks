#!/usr/bin/env python3

import random
from sys import argv, exit
from math import ceil
import network_shared as shr
import network_io_plot as io
import network_params as net

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
			self.fit = shr.mse(network, CLASSES, TRAIN, activation_function)
		else:
			self.fit = fit

	def set_genes(self, genes):
		"""Genes mutator method."""
		self.genes = genes
		# when setting genes subsequent times
		# update the fitness
		network = initialize_network(self.genes)
		self.fit = shr.mse(network, CLASSES, TRAIN, activation_function)

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
		# make network to get performance metrics
		network = initialize_network(population[0].get_genes());
		# training accuracy of network
		TRP.append(shr.performance_measure(network, TRAIN, activation_function))
		# testing accuracy of network
		TEP.append(shr.performance_measure(network, TEST, activation_function))
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
		io.out_console(AUTO, e, MSE, TRP, TEP)

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
	CHROMOSOME_SIZE = (HIDDEN_SIZE * (FEATURES+1)) + \
		(CLASSES * (HIDDEN_SIZE+1))
	POP_SIZE = net.get_population_size()
	CROSS_RATE, MUTAT_RATE, ELITE_PROPORTION, \
		TOURN_PROPORTION, BASE = net.get_ga_params(argv[1])
	EPOCHS = net.get_epochs()
	genetic_network(ELITE_PROPORTION, TOURN_PROPORTION, \
		CHROMOSOME_SIZE, EPOCHS, POP_SIZE, CROSS_RATE, MUTAT_RATE)
	if not AUTO:
		io.plot_data(EPOCHS, MSE, TRP, TEP)
	exit(0)
