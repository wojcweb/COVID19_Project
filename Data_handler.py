from os import remove
from urllib import request
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime


class DataHandler:

    def __init__(self, continent="Europe"):
        self.covid_dataset = pd.DataFrame()
        self.list_of_countries = list()
        self.country_dataset = pd.DataFrame()
        self.cases_data = np.array(1)
        self.results = []
        self.all_dates = list()
        self.all_data = np.array(1)
        self.all_countries_cases = np.zeros(300)
        self.all_countries_cases_forecast = np.zeros(300)
        self.continent = continent
        self.script_dir = os.path.dirname(__file__)
        self.results_dir = os.path.join(self.script_dir, 'Results/')
        self.current_working_dir = os.path.join(self.results_dir,
                                                datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
        self.plots_dir = os.path.join(self.current_working_dir, 'plots/')
        self.menage_directories()

    def menage_directories(self):
        if not os.path.isdir(self.results_dir):
            os.makedirs(self.results_dir)
        os.makedirs(self.current_working_dir)
        os.makedirs(self.plots_dir)
        try:
            remove("./data_covid.csv")
        except FileNotFoundError:
            print("doesn't exist")

    def download_data(self):
        url = "https://opendata.ecdc.europa.eu/covid19/casedistribution/csv"
        request.urlretrieve(url, "./data_covid.csv")
        self.covid_dataset = pd.read_csv("./data_covid.csv")

    def get_continent_data(self):
        self.covid_dataset.drop(["day", "month", "year", "geoId",
                                "countryterritoryCode"], axis=1, inplace=True)
        self.covid_dataset = self.covid_dataset.loc[self.covid_dataset["continentExp"]
                                                    == self.continent]
        self.list_of_countries = list(self.covid_dataset["countriesAndTerritories"].unique())
        self.covid_dataset = self.covid_dataset.iloc[::-1]
        self.covid_dataset.reset_index(drop=True, inplace=True)

    def get_country_data(self, country):
        self.country_dataset = \
            self.covid_dataset.loc[self.covid_dataset["countriesAndTerritories"] == country]
        self.country_dataset["dateRep"] = pd.to_datetime(self.country_dataset["dateRep"],
                                                         format="%d/%m/%Y")
        self.cases_data = np.array(self.country_dataset['cases'].values)
        self.cases_data = np.reshape(self.cases_data, (-1, 1))

    def append_results(self, country, forecast, forecast_dates):
        self.all_data = np.concatenate([self.cases_data[:-41], forecast])
        self.all_data_cum_forecast = self.all_data.cumsum()
        self.all_data_cum = self.cases_data[:-11].cumsum()
        dates = self.country_dataset['dateRep']
        self.all_dates = list(dates[:-41]) + forecast_dates
        self.results.append((country, self.all_dates[-1], np.floor(self.all_data_cum_forecast[-1]), np.floor(forecast)))

    def save_results(self):
        results = pd.DataFrame(self.results)
        results.to_csv("{}/results.csv".format(self.current_working_dir))

    def save_params(self, params):
        parameters = params + [self.continent]
        file = open("{}/parameters".format(self.current_working_dir), 'w')
        for element in parameters:
            file.write(str(element)+'\n')
        file.close()

    def plot_result(self, country):
        plt.title(country)
        plt.plot(self.all_dates, self.all_data)
        plt.plot(self.all_dates[-30:], self.all_data[-30:])
        plt.plot(self.all_dates[-30:], self.cases_data[-41:-11])
        plt.savefig(self.plots_dir + "daily_" + country)
        # plt.show()
        plt.close()
        plt.plot(self.all_dates, self.all_data_cum_forecast)
        plt.plot(self.all_dates[-30:], self.all_data_cum_forecast[-30:])
        plt.plot(self.all_dates[-30:], self.all_data_cum[-30:])
        plt.savefig(self.plots_dir + "AAll_" + country)
        plt.close()

    def add_country_cases(self):
            self.all_countries_cases[-self.all_data_cum.shape[0]:] += self.all_data_cum
            self.all_countries_cases_forecast[-self.all_data_cum_forecast.shape[0]:] += self.all_data_cum_forecast

    def plot_cumsum(self):
        self.all_countries_cases = self.all_countries_cases[self.all_countries_cases != 0]
        self.all_countries_cases_forecast = self.all_countries_cases_forecast[self.all_countries_cases_forecast != 0]
        plt.title('dupa')
        plt.plot(self.all_dates[-self.all_countries_cases.shape[0]:], self.all_countries_cases)
        plt.plot(self.all_dates[-30:], self.all_countries_cases_forecast[-30:])
        plt.savefig(self.plots_dir + "daily_dupa")
        plt.show()
        plt.close()