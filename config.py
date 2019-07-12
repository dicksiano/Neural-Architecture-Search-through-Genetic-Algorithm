class Constants():
    num_generations = 15
    initial_population_size = 40
    population_size = 20

    gene_hyperparameters = {
                            'optimizer': 
                                        [ 
                                            'sgd', 
                                            'rmsprop', 
                                            'adam'
                                        ],

                            'activation': 
                                        [
                                            'sigmoid',
                                            'relu', 
                                            'tanh'
                                        ],

                            'nb_layers': 
                                    [
                                            1, 
                                            2, 
                                            4,
                                            8
                                        ],

                            'nb_neurons': 
                                        [
                                            3, 
                                            6, 
                                            8,
                                            16, 
                                            32, 
                                            64
                                        ]
    }

    survival_rate = 0.4
    random_select_prob = 0.1
    mutation_prob = 0.2
    number_of_children = 2
