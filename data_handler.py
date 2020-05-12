import pandas as pd
import urllib.request
import numpy as np


class DataSetHandler:

    def __init__(self, data_url, str_path="./data_covid.csv"):
        self.__data_url = data_url
        url = data_url
        urllib.request.urlretrieve(url, str_path)
        self.__data_covid = pd.read_csv(str_path)
        self.__data_covid.drop(["day", "month", "year", "geoId", "countryterritoryCode"], axis=1, inplace=True)
        condition_mask_europe = (self.__data_covid["continentExp"] == "Europe")
        condition_mask_other = (self.__data_covid["countriesAndTerritories"] == "Turkey") | \
                               (self.__data_covid["countriesAndTerritories"] == "Kazakhstan")

        self.__data_covid = self.__data_covid.loc[(condition_mask_europe | condition_mask_other)]
        self.__data_covid = self.__data_covid.iloc[::-1]
        self.__data_covid.reset_index(drop=True, inplace=True)
        self.__list_of_countries = list(self.__data_covid["countriesAndTerritories"].unique())
        self.__data_covid_ref = self.__data_covid.copy()

    @property
    def list_of_countries(self):
        return self.__list_of_countries

    def get_dataset_for_country(self, country):
        self.__data_covid_ref = self.__data_covid.loc[self.__data_covid["countriesAndTerritories"] == country]
        self.__data_covid_ref["dateRep"] = pd.to_datetime( self.__data_covid_ref["dateRep"], format="%d/%m/%Y")
        cases_data = np.array( self.__data_covid_ref['cases'].values)
        cases_data = np.reshape(cases_data, (-1, 1))
        return cases_data, self.__data_covid_ref
