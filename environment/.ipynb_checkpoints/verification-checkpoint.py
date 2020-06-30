import pandas as pd

from tensorflow.keras.backend import clear_session
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

from . import configurator as config
from . import dataset


class VerificationSystem:
    def __init__(self, model_name, dataset_path, config_path, result_path):
        self.model_name = model_name
        self.result_path = result_path
        self.configurator = config.ModelConfigurator(f"{config_path}/model_conf.csv")
        self.dataset_manager = dataset.DatasetManager(dataset_path)
        
    def create_model(self, input_unit, hidden_units, output_unit):
        model = Sequential()
        model.add(Dense(
            input_dim=input_unit,
            units=hidden_units[0],
            activation="tanh",
            kernel_initializer="glorot_uniform"))
        for i in range(len(hidden_units) - 1):
            model.add(Dense(
                input_dim=hidden_units[i], 
                units=hidden_units[i + 1], 
                activation="tanh",
                kernel_initializer="glorot_uniform"))
        model.add(Dense(input_dim=hidden_units[-1], units=output_unit))
        model.compile(loss="mse", optimizer=Adam())
        return model
    
    def initialize(self, epochs, sample, learn_data, test_data):
        # ++ verification configuration ++
        self.epochs = epochs
        self.sample = sample
        x_index, y_index = self.configurator.get_index(self.model_name)
        self.dataset = self.dataset_manager.load(learn_data, test_data, x_index, y_index)
        
        # ++ model configuration ++
        self.units = self.configurator.get_units(self.model_name)
        self.batch_size = self.configurator.get_batch_size(self.model_name)
        
    def verification(self):
        for i in range(self.sample):
            clear_session()
            print(f"\rModel is {self.model_name}, sampling: {i + 1}/{self.sample}", end="")
            
            model = self.create_model(self.dataset["learn-x"].shape[1], self.units, self.dataset["learn-y"].shape[1])
            loss = model.fit(self.dataset["learn-x"], self.dataset["learn-y"], batch_size=self.batch_size, epochs=self.epochs, verbose=0)
            resp = model.predict(self.dataset["test-x"])[:, 0]
            error = self.dataset["test-y"] - resp

            if i == 0:
                df_loss = pd.DataFrame({f"#{i}": loss.history["loss"]})
                df_resp = pd.DataFrame({f"#{i}": resp})
                df_error = pd.DataFrame({f"#{i}": error})
            else:
                df_loss[f"#{i}"] = loss.history["loss"]
                df_resp[f"#{i}"] = resp
                df_error[f"#{i}"] = error

        df_loss.to_csv(f"{self.result_path}/vrf_{self.model_name}_loss.csv")
        df_resp.to_csv(f"{self.result_path}/vrf_{self.model_name}_resp.csv")
        df_error.to_csv(f"{self.result_path}/vrf_{self.model_name}_error.csv")