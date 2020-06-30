import numpy as np
import pandas as pd

import optuna
from optuna.integration import KerasPruningCallback

from tensorflow.keras.backend import clear_session
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

from . import dataset
from . import configurator as config

class NetworkOptimizer:
    def __init__(self, model_name, study_ver, load_study:bool, dataset_path, config_path, log_path):
        self.model_name = model_name
        self.study_ver = study_ver
        self.storage = f"sqlite:///{log_path}/optimize_{model_name}.db"
        self.logger = LoggerClass(f"{log_path}/score_{model_name}_{study_ver}.csv", f"{log_path}/result_{model_name}_{study_ver}.csv")
        self.configurator = config.OptimizeConfigurator(f"{config_path}/optimize_conf.csv", f"{config_patb}/model_conf.csv")
        self.dataset_manager = dataset.DatasetManager(dataset_path)
        
        if "Prop." in model_name:
            self.set_hidden_units = self.set_hidden_units_for_prop
        else:
            self.set_hidden_units = self.set_hidden_units_for_conv
        
        
        if load_study:
            self.study = optuna.load_study(
                study_name=self.study_ver,
                storage=self.storage, 
                pruner=optuna.pruners.MedianPruner())
        else:
            self.study = optuna.create_study(
                study_name=self.study_ver,
                storage=self.storage,
                direction="minimize",
                pruner=optuna.pruners.MedianPruner())

    def initialize(self, sample, epochs, filter_object):
        self.sample = sample
        self.epochs = epochs
        self.filter = filter_object
        self.elements = self.configurator.get_elements(self.model_name)
        self.max_units = self.configurator.get_unit_range(self.model_name)
        self.batch_size = self.configurator.get_batch_size(self.model_name)
            
        x_index, y_index = self.configurator.get_index(self.model_name)
        learn_name, test_name = self.configurator.get_dataset(self.model_name)
        self.dataset = self.dataset_manager.load(learn_name, test_name, x_index, y_index)
        
    def set_hidden_units_for_conv(self, trial):
        units = []
        for i in range(len(self.max_units)):
            units.append(trial.suggest_int(f"num_unit{i + 1}", 1, int(self.max_units[0])))           
        return units

    def set_hidden_units_for_prop(self, trial):
        unit1 = trial.suggest_int("num_unit1", 1, int(self.max_units[0]))
        max_unit2 = int((self.elements - self.dataset["learn-x"].shape[1] * unit1) / unit1 + 1)
        if max_unit2 > 200:
            max_unit2 = 200
        unit2 = trial.suggest_int("num_unit2", 1, max_unit2)
        return [unit1, unit2]
    
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
    
    def objective(self, trial):
        hidden_units = self.set_hidden_units(trial)
        score_list = []
        for i in range(self.sample):
            clear_session()
            print(f"\r#{trial.number} -- unit: {hidden_units}, sampling: {i + 1}/{self.sample}", end="")
            model = self.create_model(self.dataset["learn-x"].shape[1], hidden_units, self.dataset["learn-y"].shape[1])
            model.fit(self.dataset["learn-x"], self.dataset["learn-y"], batch_size=self.batch_size, epochs=self.epochs, verbose=0)
            score_list.append(model.evaluate(self.dataset["test-x"], self.dataset["test-y"], batch_size=self.batch_size, verbose=0))
        score_list_flt = self.filter.filtering(score_list)
        mean, std = pd.Series(score_list).describe().loc[["mean", "std"]]
        samples, mean_f, std_f = score_list_flt.describe().loc[["count", "mean", "std"]]
        self.logger.save_score(trial.number, hidden_units, score_list)
        self.logger.save_result(trial.number, hidden_units, samples, mean, std, mean_f, std_f)
        print(f"\r#{trial.number} -- unit: {hidden_units}, samples: {samples}/{self.sample}, mean: {mean:.4e}, std: {std:.4e}")
        return mean
    
    def optimize(self, num_trial):
        print(f"+++ Optimization for {self.model_name} +++")
        print(f"study_name: {self.model_name}_{self.study_ver}\n")
        self.study.optimize(self.objective, num_trial)
        print("\n\n*** All Trial are finished!! ***")


class LoggerClass:
    def __init__(self, score_file, result_file):
        self.file_path = {
            "score": score_file,
            "result": result_file
        }

    def save_score(self, trial_id, units, data_list):
        with open(self.file_path["score"], "w" if trial_id == 0 else "a") as file:
            file.write(f"#{trial_id}")
            for num_unit in units:
                file.write(f", {num_unit}")
            for data in data_list:
                file.write(f", {data:.6e}")
            file.write("\n")

    def save_result(self, trial_id, units, sample, mean, std, mean_f, std_f):
        if trial_id == 0:
            header = "Trials"
            for i in range(len(units)):
                header += f", Unit-{i + 1}"
            header += f", Samples"
            header += ", Estimated loss, Standard-deviation"
            header += ", Estimated loss(filter), Standard-deviation(filter)\n"
        else:
            header = ""

        with open(self.file_path["result"], "w" if trial_id == 0 else "a") as file:
            file.write(header)
            file.write(f"#{trial_id}")
            for num_unit in units:
                file.write(f", {num_unit}")
            file.write(f", {sample}, {mean:.6e}, {std:.6e}, {mean_f:.6e}, {std_f:.6e}\n")

        