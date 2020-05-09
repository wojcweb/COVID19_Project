from os import remove
from urllib import request
import pandas as pd
import numpy as np


class DataHandler:
    covid_dataset = pd.DataFrame()
    list_of_countries = list()
    country_dataset = pd.DataFrame()
    cases_data = np.array(1)
    results = []

    @classmethod
    def download_data(cls):
        try:
            remove("./data_covid.csv")
        except FileNotFoundError:
            print("doesn't exist")

        url = "https://opendata.ecdc.europa.eu/covid19/casedistribution/csv"
        request.urlretrieve(url, "./data_covid.csv")
        cls.covid_dataset = pd.read_csv("./data_covid.csv")

    @classmethod
    def get_europe_data(cls):
        cls.covid_dataset.drop(["day", "month", "year", "geoId",
                                "countryterritoryCode"], axis=1, inplace=True)
        cls.covid_dataset = cls.covid_dataset.loc[cls.covid_dataset["continentExp"]
                                                  == "Europe"]
        cls.list_of_countries = list(cls.covid_dataset["countriesAndTerritories"].unique())
        cls.covid_dataset = cls.covid_dataset.iloc[::-1]
        cls.covid_dataset.reset_index(drop=True, inplace=True)

    @classmethod
    def get_country_data(cls, country):
        cls.country_dataset = \
            cls.covid_dataset.loc[cls.covid_dataset["countriesAndTerritories"] == country]
        cls.country_dataset["dateRep"] = pd.to_datetime(cls.country_dataset["dateRep"],
                                                        format="%d/%m/%Y")
        cls.cases_data = np.array(cls.country_dataset['cases'].values)
        cls.cases_data = np.reshape(cls.cases_data, (-1, 1))

    @classmethod
    def append_results(cls, country, forecast, forecast_dates):
        all_data = np.concatenate([cls.cases_data, forecast])
        all_data = all_data.cumsum()
        all_dates = list(DataHandler.country_dataset['dateRep']) + forecast_dates
        cls.results.append((country, all_dates[-1], np.floor(all_data[-1]), np.floor(forecast)))

    @classmethod
    def save_results(cls):
        results = pd.DataFrame(cls.results)
        results.to_csv("results.csv")
