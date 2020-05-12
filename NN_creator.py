from tensorflow import keras


class NNCreator:

    def __init__(self, look_back):
        self._look_back = look_back
        self._model = keras.Sequential()
        self._model_info = None

    def assemble_network(self, num_of_lstm_neurons):
        self._model.add(keras.layers.LSTM(num_of_lstm_neurons, activation='relu', input_shape=(self._look_back, 1)))
        self._model.add(keras.layers.Dense(1, activation="linear"))

    def compile_network(self, optimizer, loss):
        self._model.compile(optimizer, loss)
        self._model_info = self._model.summary()

    @property
    def model_info(self):
        return self._model_info

    @property
    def model(self):
        return self._model



