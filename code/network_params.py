"""
General Network Parameters
"""

def get_epochs():
	num_epochs = 100
	return num_epochs

def get_hidden_size(data_used):
	if data_used == 'iris':
		# 4-3-3 = 27 dimensions
		hidden_layer_size = 3
	elif data_used == 'penguins':
		# 6-4-3 = 43 dimensions
		hidden_layer_size = 4
	elif data_used == 'wheat':
		# 7-5-3 = 58 dimensions
		hidden_layer_size = 5
	elif data_used == 'wine':
		# 13-6-3 = 105 dimensions
		hidden_layer_size = 6
	elif data_used == 'breast':
		# 31-8-2 = 274 dimensions
		hidden_layer_size = 8
	elif data_used == 'ionosphere':
		# 34-10-2 = 372 dimensions
		hidden_layer_size = 10
	return hidden_layer_size

def get_holdout():
	training_portion = 0.70
	# testing portion will be adjusted as 1 - training_portion
	return training_portion

def get_rand_range():
	rand_min = -0.50
	rand_max = 0.50
	return [rand_min, rand_max]

"""
BP-NN Specific Parameters
"""

def get_bp_params(data_used):
	if data_used == 'iris':
		learning_rate = 0.100
		momentum_rate = 0.001
	elif data_used == 'penguins':
		learning_rate = 0.100
		momentum_rate = 0.001
	elif data_used == 'wheat':
		learning_rate = 0.100
		momentum_rate = 0.001
	elif data_used == 'wine':
		learning_rate = 0.100
		momentum_rate = 0.002
	elif data_used == 'breast':
		learning_rate = 0.100
		momentum_rate = 0.003
	elif data_used == 'ionosphere':
		learning_rate = 0.100
		momentum_rate = 0.002
	return learning_rate, momentum_rate

"""
GA-NN Specific Parameters
"""

def get_ga_population_size():
	population_size = 100
	return population_size

def get_ga_params(data_used):
	if data_used == 'iris':
		crossover_rate = 0.90
		mutation_rate = 0.03
		elite_proportion = 0.05
		tournament_proportion = 0.03
		base = 0.5
	elif data_used == 'penguins':
		crossover_rate = 0.90
		mutation_rate = 0.03
		elite_proportion = 0.05
		tournament_proportion = 0.03
		base = 0.5
	elif data_used == 'wheat':
		crossover_rate = 0.90
		mutation_rate = 0.04
		elite_proportion = 0.05
		tournament_proportion = 0.04
		base = 0.6
	elif data_used == 'wine':
		crossover_rate = 0.90
		mutation_rate = 0.05
		elite_proportion = 0.05
		tournament_proportion = 0.05
		base = 0.7
	elif data_used == 'breast':
		crossover_rate = 0.90
		mutation_rate = 0.05
		elite_proportion = 0.05
		tournament_proportion = 0.07
		base = 0.8
	elif data_used == 'ionosphere':
		crossover_rate = 0.90
		mutation_rate = 0.06
		elite_proportion = 0.05
		tournament_proportion = 0.09
		base = 0.9
	return crossover_rate, mutation_rate, \
		elite_proportion, tournament_proportion, base

"""
PSO-NN Specific Parameters
"""

def get_swarm_size():
	swarm_size = 100
	return swarm_size

def get_pso_params(data_used):
	if data_used == 'iris':
		inertial_weight = 0.5
		cognitive_coefficient = 1.5
		social_coefficient = 1.2
		boundary = 3
	elif data_used == 'penguins':
		inertial_weight = 0.5
		cognitive_coefficient = 1.4
		social_coefficient = 1.3
		boundary = 4
	elif data_used == 'wheat':
		inertial_weight = 0.6
		cognitive_coefficient = 1.3
		social_coefficient = 1.1
		boundary = 5
	elif data_used == 'wine':
		inertial_weight = 0.3
		cognitive_coefficient = 1.6
		social_coefficient = 1.4
		boundary = 7
	elif data_used == 'breast':
		inertial_weight = 0.4
		cognitive_coefficient = 1.4
		social_coefficient = 1.1
		boundary = 7
	elif data_used == 'ionosphere':
		inertial_weight = 0.3
		cognitive_coefficient = 1.3
		social_coefficient = 1.3
		boundary = 9
	return inertial_weight, cognitive_coefficient, social_coefficient, \
		boundary

"""
DE-NN Specific Parameters
"""

def get_de_population_size():
	population_size = 50
	return population_size

def get_de_params(data_used):
	if data_used == 'iris':
		crossover_rate = 0.90
		differential_weight = 0.25
	elif data_used == 'penguins':
		crossover_rate = 0.90
		differential_weight = 0.35
	elif data_used == 'wheat':
		crossover_rate = 0.90
		differential_weight = 0.25
	elif data_used == 'wine':
		crossover_rate = 0.90
		differential_weight = 0.20
	elif data_used == 'breast':
		crossover_rate = 0.90
		differential_weight = 0.15
	elif data_used == 'ionosphere':
		crossover_rate = 0.90
		differential_weight = 0.10
	return crossover_rate, differential_weight

"""
BA-NN Specific Parameters
"""

def get_ba_population_size():
	population_size = 100
	return population_size

def get_ba_params(data_used):
	if data_used == 'iris':
		frequency_min = 0
		frequency_max = 2
		boundary = 3
		alpha = 0.9
		gamma = 0.9
	elif data_used == 'penguins':
		frequency_min = 0
		frequency_max = 2
		boundary = 4
		alpha = 0.9
		gamma = 0.9
	elif data_used == 'wheat':
		frequency_min = 0
		frequency_max = 2
		boundary = 5
		alpha = 0.9
		gamma = 0.9
	elif data_used == 'wine':
		frequency_min = 0
		frequency_max = 2
		boundary = 7
		alpha = 0.9
		gamma = 0.9
	elif data_used == 'breast':
		frequency_min = 0
		frequency_max = 2
		boundary = 7
		alpha = 0.9
		gamma = 0.9
	elif data_used == 'ionosphere':
		frequency_min = 0
		frequency_max = 2
		boundary = 9
		alpha = 0.9
		gamma = 0.9
	return frequency_min, frequency_max, boundary, alpha, gamma
