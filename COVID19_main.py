from data_handler import DataSetHandler
from trainer_predictor import TrainerPredictor
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import mean_absolute_error, mean_squared_error
import numpy as np
import matplotlib.pyplot as plt

url = "https://opendata.ecdc.europa.eu/covid19/casedistribution/csv"

data_handler = DataSetHandler(url)

list_of_countries = data_handler.list_of_countries
opt = Adam(learning_rate=0.0005)
loss = mean_squared_error
network_handler = TrainerPredictor(look_back=5, batch_size=10, epochs=500, optimizer=opt, loss=loss,
                                   num_of_lstm_neurons=256)

data_to_train, raw_data = data_handler.get_dataset_for_country(list_of_countries[0])

network_handler.train_model(data_to_train)

forecast, forecast_date = network_handler.do_forecast(7, data_to_train, raw_data)
forecast = np.reshape(forecast, (-1, 1))
all_data = np.concatenate([data_to_train, forecast])
all_dates = list(raw_data['dateRep']) + forecast_date
# plt.title(country)
markers_on = all_data.copy()
markers_on[:len(raw_data)] = np.nan
plt.plot(all_dates, all_data, color='b')
plt.plot(all_dates, markers_on, color='r')
plt.show()

# all_data = all_data.cumsum()
# results.append((country, all_dates[-1], np.floor(all_data[-1]), np.floor(forecast)))
# results = pd.DataFrame(results)
# results.to_csv("result9.csv")
