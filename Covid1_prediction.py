import tensorflow as tf
from Data_handler import DataHandler
from Neural_network import NeuralNetwork
import os

def main():
    look_back = 3
    num_prediction = 15
    num_epochs = 300
    DataHandler.menage_directories()
    network = NeuralNetwork(look_back, num_prediction, num_epochs)
    DataHandler.download_data()
    DataHandler.get_europe_data()

    for country in DataHandler.list_of_countries[-15:-12]:
        DataHandler.get_country_data(country)
        network.define_model()
        network.train_model(DataHandler.cases_data)

        forecast, forecast_dates = network.get_forecast(DataHandler.cases_data,
                                                        DataHandler.country_dataset)
        DataHandler.append_results(country, forecast, forecast_dates)
        DataHandler.plot_result(country)

    DataHandler.save_results()


if __name__ == "__main__":
    main()
