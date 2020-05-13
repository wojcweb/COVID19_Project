from NN_creator import NNCreator
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
import numpy as np
import datetime
import pandas as pd



class TrainerPredictor(NNCreator):

    def __init__(self, look_back, batch_size, epochs, optimizer, loss, num_of_lstm_neurons):
        super().__init__(look_back)
        self.__batch_size = batch_size
        self.__epochs = epochs
        self.__optimizer = optimizer
        self.__loss = loss
        self.assemble_network(num_of_lstm_neurons)

    def generate_timeseries(self, dataset):
        train_generator = TimeseriesGenerator(dataset, dataset, length=self._look_back, batch_size=self.__batch_size)
        return train_generator

    def train_model(self, dataset):
        self.compile_network(self.__optimizer, self.__loss)
        train_generator = self.generate_timeseries(dataset)
        self._model.fit_generator(train_generator, epochs=self.__epochs, verbose=1)

    def do_forecast(self, date_to_predict, dataset, dataset_raw):
        last_date = dataset_raw["dateRep"].iloc[-1]
        last_date = last_date + datetime.timedelta(days=1)
        prediction_dates = pd.date_range(last_date, date_to_predict).tolist()

        prediction_list = dataset[-self._look_back:]
        for _ in range(len(prediction_dates)):
            x = prediction_list[-self._look_back:]
            x = x.reshape((1, self._look_back, 1))
            out = self._model.predict(x)[0][0]
            prediction_list = np.append(prediction_list, out)
        prediction_list = prediction_list[self._look_back:]

        return prediction_list, prediction_dates
