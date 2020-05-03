import pandas as pd
import datetime
import json
import numpy as np
import os.path
from pathlib import Path
import urllib.request
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
import matplotlib.pyplot as plt

try:
    os.remove("./data_covid.csv")
except FileNotFoundError:
    print("doesn't exist")

url = "https://opendata.ecdc.europa.eu/covid19/casedistribution/csv"
urllib.request.urlretrieve(url, "./data_covid.csv")

# Reading file
covid_dataset = pd.read_csv("./data_covid.csv")

# Clean dataset fix datetime
covid_dataset.drop(["day", "month", "year", "geoId", "countryterritoryCode"], axis=1, inplace=True)
covid_dataset = covid_dataset.loc[covid_dataset["continentExp"] == "Europe"]
list_of_countries = list(covid_dataset["countriesAndTerritories"].unique())
covid_dataset = covid_dataset.iloc[::-1]
covid_dataset.reset_index(drop=True, inplace=True)
covid_dataset_ref = covid_dataset.copy()
# if u want to train model on cummulative sum uncomment
# for europe_country in list_of_countries:
#     country_indexes = covid_dataset.loc[covid_dataset["countriesAndTerritories"] == europe_country].index
#     cum_sum = covid_dataset.iloc[country_indexes, 1].cumsum()
#     covid_dataset.iloc[country_indexes, 1] = cum_sum

country_for_test = list_of_countries[-13]
results = []

for country in list_of_countries:
    covid_dataset = covid_dataset_ref.loc[covid_dataset_ref["countriesAndTerritories"] == country]
    # cp_dataset = covid_dataset.copy()
    # first_date = covid_dataset["dateRep"].iloc[0]
    # first_date = pd.to_datetime(first_date, format="%d/%m/%Y")
    # first_correct_date = datetime.datetime(year=first_date.year, month=first_date.month, day=first_date.day)
    #
    # prediction_dates = pd.date_range(first_date, periods=len(covid_dataset), freq='d')
    covid_dataset["dateRep"] = pd.to_datetime(covid_dataset["dateRep"], format="%d/%m/%Y")
    cases_data = np.array(covid_dataset['cases'].values)
    cases_data = np.reshape(cases_data, (-1, 1))

    look_back = 3

    train_generator = TimeseriesGenerator(cases_data, cases_data, length=look_back, batch_size=10)

    # model

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(16, activation='relu', input_shape=(look_back, 1)))
    model.add(tf.keras.layers.Dense(1, activation='linear'))

    opt = tf.keras.optimizers.Adam(learning_rate=0.0005)
    model.compile(optimizer=opt, loss='mse')
    print(model.summary())
    num_epochs = 500
    model.fit_generator(train_generator, epochs=num_epochs, verbose=1)
    prediction = model.predict_generator(train_generator)


    def predict(num_prediction, model):
        prediction_list = cases_data[-look_back:]

        for _ in range(num_prediction):
            x = prediction_list[-look_back:]
            x = x.reshape((1, look_back, 1))
            out = model.predict(x)[0][0]
            prediction_list = np.append(prediction_list, out)
        prediction_list = prediction_list[look_back:]

        return prediction_list


    def predict_dates(num_prediction):
        last_date = covid_dataset['dateRep'].iloc[-1]
        last_date = last_date + datetime.timedelta(days=1)
        prediction_dates = pd.date_range(last_date, periods=num_prediction).tolist()
        return prediction_dates


    num_prediction = 9
    forecast = predict(num_prediction, model)
    forecast_dates = predict_dates(num_prediction)

    forecast = np.reshape(forecast, (-1, 1))

    all_data = np.concatenate([cases_data, forecast])
    all_dates = list(covid_dataset['dateRep']) + forecast_dates

    # plt.title(country)
    # plt.plot(all_dates, all_data)
    # plt.show()

    all_data = all_data.cumsum()
    results.append((country, all_dates[-1], all_data[-1], forecast))

results = pd.DataFrame(results)
results.to_csv("result5.csv")
