from data_handler import DataSetHandler
from trainer_predictor import TrainerPredictor
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import mean_absolute_error, mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime
url = "https://opendata.ecdc.europa.eu/covid19/casedistribution/csv"
date_to_predict = datetime.datetime(2020, 5, 19, 0, 0, 0)

data_handler = DataSetHandler(url)
results = []
report_date = []
all_dates = []
list_of_countries = data_handler.list_of_countries
opt = Adam(learning_rate=0.0002)
loss = mean_absolute_error
network_handler = TrainerPredictor(look_back=10, batch_size=10, epochs=250, optimizer=opt, loss=loss,
                                   num_of_lstm_neurons=32)
# list_of_countries = [list_of_countries[0]]
for country in list_of_countries:
    data_to_train, raw_data = data_handler.get_dataset_for_country(country)

    network_handler.train_model(data_to_train)

    forecast, forecast_date = network_handler.do_forecast(date_to_predict, data_to_train, raw_data)
    forecast = np.reshape(forecast, (-1, 1))
    all_data = np.concatenate([data_to_train, forecast])
    all_dates = list(raw_data['dateRep']) + forecast_date
    # plt.title(country)
    markers_on = all_data.copy()
    markers_on[:len(raw_data)] = np.nan
    # plt.title(country)
    # plt.plot(all_dates, all_data, color='b')
    # plt.plot(all_dates, markers_on, color='r')
    # plt.show()

    all_data = all_data.cumsum()
    results.append((country, all_dates[-1], np.floor(all_data[-1]), np.floor(forecast)))

results = pd.DataFrame(results)
results.to_csv("result3_{}.csv".format(str(all_dates[-1])))
