import pandas as pd
import os.path
from pathlib import Path
import urllib.request
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

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

for europe_country in list_of_countries:
    country_indexes = covid_dataset.loc[covid_dataset["countriesAndTerritories"] == europe_country].index
    cum_sum = covid_dataset.iloc[country_indexes, 1].cumsum()
    covid_dataset.iloc[country_indexes, 1] = cum_sum

country_for_test = list_of_countries[16]
covid_dataset = covid_dataset.loc[covid_dataset["countriesAndTerritories"]==country_for_test]