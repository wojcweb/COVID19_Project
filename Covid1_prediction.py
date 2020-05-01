import pandas as pd
import numpy as np
import os.path
from pathlib import Path
import urllib.request
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
import matplotlib.pyplot as plt

data_file = Path("./data_covid.csv")
if data_file.is_file():
    print("File doesn't exist")
else:
    url = "https://opendata.ecdc.europa.eu/covid19/casedistribution/csv"
    urllib.request.urlretrieve(url, "./data_covid.csv")

covid_dataset = pd.read_csv("./data_covid.csv")

# Clean dataset
covid_dataset.drop(["day", "month", "year", "geoId", "countryterritoryCode"], axis=1, inplace=True)
covid_dataset = covid_dataset.loc[covid_dataset["continentExp"] == "Europe"]
list_of_countries = list(covid_dataset["countriesAndTerritories"].unique())
covid_dataset = covid_dataset.iloc[::-1]
covid_dataset.reset_index(drop=True, inplace=True)

# if u want to train model on cummulative sum uncomment
# for europe_country in list_of_countries:
#     country_indexes = covid_dataset.loc[covid_dataset["countriesAndTerritories"] == europe_country].index
#     cum_sum = covid_dataset.iloc[country_indexes, 1].cumsum()
#     covid_dataset.iloc[country_indexes, 1] = cum_sum

country_for_test = list_of_countries[16]
covid_dataset = covid_dataset.loc[covid_dataset["countriesAndTerritories"]==country_for_test]

cases_data = np.array(covid_dataset['cases'].values)
cases_data = np.reshape(cases_data, (-1, 1))
split_percent = 0.80
#split = int(split_percent*len(cases_data))

look_back = 5

train_generator = TimeseriesGenerator(cases_data, cases_data, length=look_back, batch_size=5 )

# model

model = tf.keras.Sequential()
model.add(tf.keras.layers.LSTM(10, activation='relu', input_shape=(look_back, 1)))
model.add(tf.keras.layers.Dense(1, activation='linear'))

opt = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=opt, loss='mse')
print(model.summary())
num_epochs = 1000
model.fit_generator(train_generator, epochs=num_epochs, verbose=1)
prediction = model.predict_generator(train_generator)


def predict(num_prediction, model):
    prediction_list = cases_data[-look_back:]

    for _ in range(num_prediction):
        x = prediction_list[-look_back:]
        x = x.reshape((1, look_back, 1))
        out = model.predict(x)[0][0]
        prediction_list = np.append(prediction_list, out)
    prediction_list = prediction_list[look_back - 1:]

    return prediction_list


def predict_dates(num_prediction):
    last_date = covid_dataset['dateRep'].values[-1]
    prediction_dates = pd.date_range(last_date, periods=num_prediction + 1).tolist()
    return prediction_dates


num_prediction = 8
forecast = predict(num_prediction, model)
forecast_dates = predict_dates(num_prediction)


plt.plot(cases_data)
plt.plot(prediction)
plt.show()