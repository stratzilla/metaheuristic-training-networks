"""
General Network Parameters
"""

def get_epochs():
	num_epochs = 100
	return num_epochs

def get_hidden_size(data_used):
	if data_used == 'iris':
		hidden_layer_size = 3
	elif data_used == 'wheat':
		hidden_layer_size = 5
	elif data_used == 'wine':
		hidden_layer_size = 7
	elif data_used == 'breast':
		hidden_layer_size = 10
	return hidden_layer_size

"""
GA-NN Specific Parameters
"""

def get_population_size():
	population_size = 100
	return population_size

def get_ga_params(data_used):
	crossover_rate = 0.90
	mutation_rate = 0.05
	elite_proportion = 0.05
	tournament_proportion = 0.03
	return crossover_rate, mutation_rate, \
		elite_proportion, tournament_proportion

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
		social_coefficient = 1.3
	elif data_used == 'wheat':
		inertial_weight = 0.52
		cognitive_coefficient = 1.4
		social_coefficient = 1.1
	elif data_used == 'wine':
		inertial_weight = 0.4
		cognitive_coefficient = 1.3
		social_coefficient = 1.2
	elif data_used == 'breast':
		inertial_weight = 0.2
		cognitive_coefficient = 1.1
		social_coefficient = 1.5
	return inertial_weight, cognitive_coefficient, social_coefficient