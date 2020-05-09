import pandas as pd
import datetime
import numpy as np
import os.path
import urllib.request
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator


def import_covid_data():
    try:
        os.remove("./data_covid.csv")
    except FileNotFoundError:
        print("doesn't exist")

    url = "https://opendata.ecdc.europa.eu/covid19/casedistribution/csv"
    urllib.request.urlretrieve(url, "./data_covid.csv")
    return pd.read_csv("./data_covid.csv")


def get_europe_data(covid_dataset):
    covid_dataset.drop(["day", "month", "year", "geoId",
                        "countryterritoryCode"], axis=1, inplace=True)
    covid_dataset = covid_dataset.loc[covid_dataset["continentExp"]
                                      == "Europe"]
    list_of_countries = list(covid_dataset["countriesAndTerritories"].unique())
    covid_dataset = covid_dataset.iloc[::-1]
    covid_dataset.reset_index(drop=True, inplace=True)
    #covid_dataset_ref = covid_dataset.copy()
    return list_of_countries, covid_dataset


def get_country_data(covid_dataset, country):
    country_dataset = \
        covid_dataset.loc[covid_dataset["countriesAndTerritories"] == country]
    country_dataset["dateRep"] = pd.to_datetime(country_dataset["dateRep"],
                                                format="%d/%m/%Y")
    cases_data = np.array(country_dataset['cases'].values)
    cases_data = np.reshape(cases_data, (-1, 1))
    return country_dataset, cases_data


def get_train_generator(cases_data, look_back):
    return TimeseriesGenerator(cases_data, cases_data,
                               length=look_back, batch_size=10)


def define_model(look_back):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(16, activation='relu',
                                   input_shape=(look_back, 1)))
    model.add(tf.keras.layers.Dense(1, activation='linear'))
    return model


def predict(num_prediction, model, cases_data, look_back):
    prediction_list = cases_data[-look_back:]
    for _ in range(num_prediction):
        x = prediction_list[-look_back:]
        x = x.reshape((1, look_back, 1))
        out = model.predict(x)[0][0]
        prediction_list = np.append(prediction_list, out)
    return prediction_list[look_back:]


def predict_dates(num_prediction, covid_dataset):
    last_date = covid_dataset['dateRep'].iloc[-1]
    last_date = last_date + datetime.timedelta(days=1)
    return pd.date_range(last_date, periods=num_prediction).tolist()


def save_results(results):
    results = pd.DataFrame(results)
    results.to_csv("results.csv")


def main():
    look_back = 3
    num_prediction = 7
    num_epochs = 300
    covid_dataset = import_covid_data()
    list_of_countries, covid_dataset = get_europe_data(covid_dataset)
    results = []
    for country in list_of_countries:
        country_dataset, cases_data = get_country_data(covid_dataset, country)
        train_generator = get_train_generator(cases_data, look_back)
        model = define_model(look_back)

        opt = tf.keras.optimizers.Adam(learning_rate=0.0005)
        model.compile(optimizer=opt, loss='mae')
        model.fit_generator(train_generator, epochs=num_epochs, verbose=1)
        model.predict_generator(train_generator)

        forecast = predict(num_prediction, model, cases_data, look_back)
        forecast_dates = predict_dates(num_prediction, country_dataset)
        forecast = np.reshape(forecast, (-1, 1))

        all_data = np.concatenate([cases_data, forecast])
        all_data = all_data.cumsum()
        all_dates = list(country_dataset['dateRep']) + forecast_dates
        results.append((country, all_dates[-1], np.floor(all_data[-1]), np.floor(forecast)))
    save_results(results)


if __name__ == "__main__":
    main()
