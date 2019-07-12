from config import Constants
from neural_network import NeuralNetwork
import random

class GeneticAlgorithm():
    def create_initial_population(self):
        population = []

        while len(population) < Constants.initial_population_size:
            random_hyp = {}   
            for param in Constants.gene_hyperparameters:
                random_hyp[param] = random.choice(Constants.gene_hyperparameters[param])

            if random_hyp not in population:
                population.append(random_hyp)

        return [NeuralNetwork(hyperparam) for hyperparam in population]

    def mutation(self, neural_net):
        mut = random.choice( list(Constants.gene_hyperparameters.keys()) )
        neural_net.hyperparameters[mut] = random.choice( Constants.gene_hyperparameters[mut] )

        return neural_net

    def crossing_over(self, mother, father):
        children = []
        
        for i in range(Constants.number_of_children):
            hyperparams = {}

            for param in Constants.gene_hyperparameters:
                hyperparams[param] = random.choice( [ mother.hyperparameters[param], father.hyperparameters[param] ] )
                neural_net = NeuralNetwork(hyperparams)

            if random.random() < Constants.mutation_prob:
                neural_net = self.mutation(neural_net)

            children.append(neural_net)

        return children

    def random_selection(self, selected, not_selected):
        for candidate in not_selected:
            if random.random() < Constants.random_select_prob:
                selected.append(candidate)

        return selected

    def reproduction(self, desired_population_size, selected):
        children = []
        while len(children) < (desired_population_size - len(selected)):
            male = female = 0
            while male == female:
                male   = random.randint(0, len(selected)-1)
                female = random.randint(0, len(selected)-1)

            new_candidates = self.crossing_over(selected[female], selected[male])
            for candidate in new_candidates:
                if len(children) < (desired_population_size - len(selected)):
                    children.append(candidate)

        selected.extend(children)
        return selected

    def evolution(self, population):
        population = sorted(population, key=lambda nn: nn.accuracy, reverse=True)
        survival_size = int(Constants.survival_rate * len(population))

        selected = population[:survival_size]
        not_selected = population[survival_size:]

        selected = self.random_selection(selected, not_selected)
        selected = self.reproduction(len(population), selected)

        return selected
