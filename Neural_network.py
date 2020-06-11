from tensorflow import keras
import numpy as np
import pandas as pd
import datetime


class NeuralNetwork:

    def __init__(self, look_back, prediction, epochs):
        self.look_back = look_back
        self.num_prediction = prediction
        self.num_epochs = epochs
        self.model = keras.Sequential()

    def get_train_generator(self, cases_data):
        return keras.preprocessing.sequence.TimeseriesGenerator(cases_data,
                                                                cases_data,
                                                                length=self.look_back,
                                                                batch_size=5) #10

    def define_model(self): #put outside loop? where to define model?
        #what about compiling already compiled model model in loop? (no re-declaring)
        self.model = keras.Sequential()
        self.model.add(keras.layers.LSTM(128, activation='relu', input_shape=(self.look_back, 1),
                                         return_sequences=True))
        self.model.add(keras.layers.LSTM(64, activation='relu'))
        self.model.add(keras.layers.Dense(64, activation='relu'))
        self.model.add(keras.layers.Dense(16, activation='relu'))
        self.model.add(keras.layers.Dense(1, activation='linear'))
        self.model.add(keras.layers.Activation(activation='relu'))


    def train_model(self, cases_data):
        opt = keras.optimizers.Adam(learning_rate=0.0005)
        train_generator = self.get_train_generator(cases_data)
        self.model.compile(optimizer=opt, loss='mae')
        self.model.fit(train_generator, epochs=self.num_epochs, verbose=1)#verbose=1 talks
        self.model.predict(train_generator)

    def predict(self, model, cases_data):
        prediction_list = cases_data[-self.look_back:]
        for _ in range(self.num_prediction):
            x = prediction_list[-self.look_back:]
            x = x.reshape((1, self.look_back, 1))
            out = model.predict(x)[0][0]
            prediction_list = np.append(prediction_list, out)
        return prediction_list[self.look_back:]

    def predict_dates(self, covid_dataset):
        last_date = covid_dataset['dateRep'].iloc[-1]
        last_date = last_date + datetime.timedelta(days=1)
        return pd.date_range(last_date, periods=self.num_prediction).tolist()

    def get_forecast(self, cases_data, country_dataset):
        forecast = self.predict(self.model, cases_data)
        forecast_dates = self.predict_dates(country_dataset)
        forecast = np.reshape(forecast, (-1, 1))
        return forecast, forecast_dates

    def get_parameters(self):
        return [self.look_back, self.num_prediction, self.num_epochs]
