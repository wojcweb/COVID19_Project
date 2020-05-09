import tensorflow as tf
from Data_handler import DataHandler
from Neural_network import NeuralNetwork


def main():
    look_back = 3
    num_prediction = 7
    num_epochs = 300
    network = NeuralNetwork(look_back, num_prediction, num_epochs)
    DataHandler.download_data()
    DataHandler.get_europe_data()

    for country in DataHandler.list_of_countries[-15:-12]:
        DataHandler.get_country_data(country)
        train_generator = network.get_train_generator(DataHandler.cases_data)
        model = network.define_model()

        opt = tf.keras.optimizers.Adam(learning_rate=0.0005)
        model.compile(optimizer=opt, loss='mae')
        model.fit_generator(train_generator, epochs=num_epochs, verbose=1)
        model.predict_generator(train_generator)

        forecast, forecast_dates = network.get_forecast(model, DataHandler.cases_data,
                                                        DataHandler.country_dataset)
        DataHandler.append_results(country, forecast, forecast_dates)
    DataHandler.save_results()


if __name__ == "__main__":
    main()
