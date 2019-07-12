from config import Constants
from input_reader import read_data
from genetic_algorithm import GeneticAlgorithm

def train_population(neural_net_population, dataset):
	for neural_net in neural_net_population:
		print('*' * 3, 'Training', neural_net.hyperparameters, '-' * 10)
		neural_net.train(dataset)

def calculate_avg_fitness(neural_net_population):
	return sum([ nn.accuracy for nn in neural_net_population ]) / float(len(neural_net_population))

def show_results(final_population):
	avg_fitness = calculate_avg_fitness(final_population)
	print("Final Generation: {}".format(avg_fitness))

	for neural_net in sorted(final_population, key=lambda nn: nn.accuracy):
		neural_net.show_configuration()

def natural_selection(dataset):
	ga = GeneticAlgorithm()
	neural_net_population = ga.create_initial_population()

	for i in range(Constants.num_generations-1):
		print("Generation {}".format(i), '-' * 20)
		train_population(neural_net_population, dataset)
		avg_fitness = calculate_avg_fitness(neural_net_population)

		print('*' * 5, "Generation {}: {}".format(i, avg_fitness))
		neural_net_population = ga.evolution(neural_net_population)

	show_results(neural_net_population)

def main():
	dataset = read_data()
	natural_selection(dataset)

if __name__ == '__main__':
    main()