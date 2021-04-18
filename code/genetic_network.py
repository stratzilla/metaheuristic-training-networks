#!/usr/bin/env python3

from random import randint, randrange, gauss, uniform
from sys import argv, exit
from math import ceil
import network_shared as net
import network_io_plot as io
import network_params as par

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
			network = net.initialize_network(self.genes, FEATURES, \
				HIDDEN_SIZE, CLASSES)
			self.fit = net.mse(network, CLASSES, TRAIN, activation_function)
		else:
			self.fit = fit

	def set_genes(self, genes):
		"""Genes mutator method."""
		self.genes = genes
		# when setting genes subsequent times
		# update the fitness
		network = net.initialize_network(self.genes, FEATURES, \
			HIDDEN_SIZE, CLASSES)
		self.fit = net.mse(network, CLASSES, TRAIN, activation_function)

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

def genetic_algorithm(el_p, to_p, dim, epochs, pop_size, axis_range, c_r, m_r):
	"""Genetic Neural Network training function.
	Main driver for the GA optimization of network weights.

	Parameters:
		el_p : the proportion of elites
		to_p : the proportion of tournament
		dim : dimensionality of network.
		epochs : how many generations to run.
		pop_size : the population size.
		axis_range : the minimum and maximum value an axis may be.
		c_r : crossover rate.
		m_r : mutation rate.
	"""
	if not AUTO:
		print('Epoch, MSE, Train. Acc%, Test Acc%')
	# initialize network as initially random
	population = net.initialize_population(Chromosome, pop_size, dim, \
		axis_range)
	for e in range(1, epochs+1):
		population.sort() # sort the population by fitness
		MSE.append(population[0].get_fit()) # get fitness of best network
		# make network to get performance metrics
		network = net.initialize_network(population[0].get_genes(), \
			FEATURES, HIDDEN_SIZE, CLASSES)
		# training accuracy of network
		TRP.append(net.performance_measure(network, TRAIN, activation_function))
		# testing accuracy of network
		TEP.append(net.performance_measure(network, TEST, activation_function))
		mating_pool = [] # init mating pool
		elites = elite_selection(population, el_p) # get elites from population
		del population[:len(elites)] # remove elites
		# find tournament and winner
		t_winner = tournament_selection(population, to_p)
		# add tournament victor and elites to mating pool
		mating_pool.extend(elites)
		mating_pool.append(t_winner)
		# generate a new population based on mating pool
		population = evolve(mating_pool, elites, pop_size, c_r, m_r)
		mating_pool.clear() # clear mating pool for next gen
		io.out_console(AUTO, e, MSE, TRP, TEP)

def evolve(mating_pool, elites, pop_size, cro_r, mut_r):
	"""Evolves population based on genetic operators.

	Parameters:
		mating_pool : where to select parents from.
		elites : previously found elites.
		pop_size : the population size.
		cro_r : crossover rate.
		mut_r : mutation rate.

	Returns:
		A new population of offspring from mating pool.
	"""
	new_population = [] # store new population as list
	new_population += elites # add elites verbatim
	while len(new_population) < pop_size: # while population isn't at max size
		# get both parents indices
		p_a_idx = randrange(len(mating_pool))
		p_b_idx = randrange(len(mating_pool))
		# we don't mind parents having identical genes but we don't
		# want the parents to use the same index. Parent A can be
		# equal to Parent B, but Parent A cannot be Parent B
		if p_a_idx == p_b_idx:
			continue
		# get the parents from indices
		parent_a = mating_pool[p_a_idx]
		parent_b = mating_pool[p_b_idx]
		# find children using crossover
		child_a, child_b = crossover(parent_a, parent_b, cro_r)
		# mutate each child
		child_a = mutation(child_a, mut_r)
		child_b = mutation(child_b, mut_r)
		# add children to population
		new_population += [child_a, child_b]
	return new_population

def crossover(parent_a, parent_b, cro_r):
	"""Two-point crossover operator.

	Parameters:
		parent_a : the first parent.
		parent_b : the second parent.
		cro_r : the crossover chance.

	Returns:
		Two child chromosomes as a product of both parents.
	"""
	# only perform crossover based on the crossover rate
	if uniform(0.00, 1.00) >= cro_r:
		child_a = Chromosome(parent_a.get_genes(), parent_a.get_fit())
		child_b = Chromosome(parent_b.get_genes(), parent_b.get_fit())
		return child_a, child_b
	genes_a, genes_b = [], []
	# find pivot points at random 1..n-1
	pivot_a = randint(1, len(parent_a)-1)
	# second pivot is between pivot_a..n-1
	pivot_b = randint(pivot_a, len(parent_a)-1)
	for i, _ in enumerate(parent_a):
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

def mutation(child, mut_r):
	"""Mutation operator.

	Parameters:
		child : the chromosome to mutate.
		mut_r : the mutation chance.

	Returns:
		A mutated child.
	"""
	# the new genes to make
	genes = [gene for gene in child.get_genes()]
	for i, _ in enumerate(genes):
		# only perform mutation based on the mutation rate
		if uniform(0.00, 1.00) <= mut_r:
			# update that axes with random position
			genes[i] = gauss(mu=genes[i], sigma=(BASE + child.get_fit()))
	# we don't need to update the fitness if the gene
	# hasn't changed, so only update genes if they've changed
	if genes != child.get_genes():
		child.set_genes(genes)
	return child

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
	for _ in range(ceil(len(population)*percent)):
		random_idx = randint(0, len(population)-1)
		tournament.append(population.pop(random_idx)) # append to tournament
	tournament.sort() # sort by fitness
	return tournament[0] # return best fit from tournament

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
	# ga-specific parameters
	POP_SIZE = par.get_ga_population_size()
	CROSS_RATE, MUTAT_RATE, ELITE_PROPORTION, \
		TOURN_PROPORTION, BASE = par.get_ga_params(argv[1])
	# run the ga-nn
	genetic_algorithm(ELITE_PROPORTION, TOURN_PROPORTION, DIMENSIONS, EPOCHS, \
		POP_SIZE, AXIS_RANGE, CROSS_RATE, MUTAT_RATE)
	if not AUTO:
		io.plot_data(EPOCHS, MSE, TRP, TEP)
	exit(0)
