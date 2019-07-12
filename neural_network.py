from keras.models import Sequential
from keras.layers import Dense

class NeuralNetwork():
    def __init__(self, hyperparameters):
        self.hyperparameters = hyperparameters
        self.accuracy = 0
        self.model = None

    def train(self, dataset):
        if self.model == None:
            self.compile_model(dataset[0].shape[1])
            self.train_model(dataset)
            self.accuracy = self.evaluate_model(dataset)

    def show_configuration(self):
        print(self.hyperparameters, ' - Accuracy:', self.accuracy)

    def compile_model(self, input_format):
        nb_layers  = self.hyperparameters['nb_layers']
        nb_neurons = self.hyperparameters['nb_neurons']
        activation = self.hyperparameters['activation']
        optimizer  = self.hyperparameters['optimizer']

        model = Sequential()
        model.add(Dense(nb_neurons, activation=activation, input_shape=(input_format,)))

        for layer in range(nb_layers - 1):
            model.add(Dense(nb_neurons, activation=activation))

        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy']) # binary classifier


        self.model = model

    def train_model(self, dataset):
        [x_train, y_train, x_test, y_test] = dataset
        self.model.fit(x_train, y_train, nb_epoch=2 , validation_data=(x_test, y_test))

    def evaluate_model(self, dataset):
        [x_train, y_train, x_test, y_test] = dataset
        eval = self.model.evaluate(x_test, y_test, verbose=0)

        return eval[1] # 1 is for acc, 0 is for loss