"""
General Network Parameters
"""

def get_epochs():
	num_epochs = 50
	return num_epochs

def get_hidden_size(data_used):
	if data_used == 'iris':
		# 4-3-3 = 27 dimensions
		hidden_layer_size = 3
	elif data_used == 'wheat':
		# 7-5-3 = 58 dimensions
		hidden_layer_size = 5
	elif data_used == 'wine':
		# 13-6-3 = 105 dimensions
		hidden_layer_size = 6
	elif data_used == 'breast':
		# 31-8-2 = 274 dimensions
		hidden_layer_size = 8
	return hidden_layer_size

"""
BP-NN Specific Parameters
"""

def get_bp_params(data_used):
	if data_used == 'iris':
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
	return learning_rate, momentum_rate

"""
GA-NN Specific Parameters
"""

def get_population_size():
	population_size = 100
	return population_size

def get_ga_params(data_used):
	if data_used == 'iris':
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
		boundary = 5
	return inertial_weight, cognitive_coefficient, social_coefficient, \
		boundary