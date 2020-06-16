import tensorflow as tf
from Data_handler import DataHandler
from Neural_network import NeuralNetwork
import timeit

def main():
    look_back = 15
    num_prediction =7
    num_epochs = 350
    data_menager = DataHandler()#optional: continent, default: europe
    data_menager.download_data()
    data_menager.get_continent_data()
    network = NeuralNetwork(look_back, num_prediction, num_epochs)
    # countries = ["Poland", "Spain"]# "Italy", "Czechia"]
    for country  in data_menager.list_of_countries[1:10]:
        begin = timeit.default_timer()
        print("**********************************************************")
        print("******************** " + country + " **********************")
        print("**********************************************************")
        data_menager.get_country_data(country)

        network.define_model()
        network.train_model(data_menager.cases_data)
        forecast, forecast_dates = network.get_forecast(data_menager.cases_data,
                                                        data_menager.country_dataset)

        data_menager.append_results(country, forecast, forecast_dates)
        data_menager.plot_result(country)
        print('Time: ' + str(timeit.default_timer() - begin))
    data_menager.save_results()
    data_menager.save_params(network.get_parameters())


if __name__ == "__main__":
    main()
