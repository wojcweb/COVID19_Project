from tensorflow import keras
import numpy as np
import pandas as pd
import datetime


class NeuralNetwork:

    def __init__(self, look_back, prediction, epochs):
        self.look_back = look_back
        self.num_prediction = prediction
        self.num_epochs = epochs

    def get_train_generator(self, cases_data):
        return keras.preprocessing.sequence.TimeseriesGenerator(cases_data,
                                                                cases_data,
                                                                length=self.look_back,
                                                                batch_size=10)

    def define_model(self):
        model = keras.Sequential()
        model.add(keras.layers.LSTM(16, activation='relu',
                                    input_shape=(self.look_back, 1)))
        model.add(keras.layers.Dense(1, activation='linear'))
        return model

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

    def get_forecast(self, model, cases_data, country_dataset):
        forecast = self.predict(model, cases_data)
        forecast_dates = self.predict_dates(country_dataset)
        forecast = np.reshape(forecast, (-1, 1))
        return forecast, forecast_dates
