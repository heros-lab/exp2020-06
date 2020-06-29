# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import filter
import optuna
from optuna.integration import KerasPruningCallback
from tensorflow.keras.backend import clear_session
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam


class NetworkOptimizerClass:
    def __init__(self, epochs, sample, max_units, batch_size, dataset, score_file_path, result_file_path, model_type, num_all_elements=None):
        self.num_elements = num_all_elements
        self.epochs = epochs
        self.sample = sample
        self.max_units = max_units
        self.batch_size = batch_size
        self.learn_x = dataset["learn-x"]
        self.learn_y = dataset["learn-y"]
        self.test_x = dataset["test-x"]
        self.test_y = dataset["test-y"]

        self.hidden_units_generator = HiddenUnitsGenerator(model_type)
        self.filter = filter.Filter_with_IQR()
        self.logger = LoggerClass(self.sample, score_file_path, result_file_path)

    def create_model(self, input_unit, hidden_units, output_unit):
        model = Sequential()
        model.add(
            Dense(input_dim=input_unit, units=hidden_units[0],
                  activation="tanh", kernel_initializer="glorot_uniform"))
        for i in range(len(hidden_units) - 1):
            model.add(
                Dense(input_dim=hidden_units[i], units=hidden_units[i + 1],
                      activation="tanh", kernel_initializer="glorot_uniform"))
        model.add(Dense(input_dim=hidden_units[-1], units=output_unit))
        model.compile(loss="mse", optimizer=Adam())
        return model

    def objective(self, trial):
        hidden_units = self.hidden_units_generator.create(trial)
        score_list = []
        for i in range(self.sample):
            # ++ initialization ++
            clear_session()
            print(f"\r#{trial.number:2} -- unit: {hidden_units}, sampling: {i + 1}/{self.sample}", end="")
            # ++ model creation and learning ++
            model = self.create_model(self.learn_x.shape[1], hidden_units, self.learn_y.shape[1])
            model.fit(self.learn_x, self.learn_y, batch_size=self.batch_size, epochs=self.epochs, verbose=0)
            # ++ evaluate to use test-data ++
            score = model.evaluate(self.test_x, self.test_y, batch_size=self.test_x.shape[0], verbose=0)
            score_list.append(score)
        # ++ filtering process ++
        score_list_flt = self.filter.filtering(score_list)
        # ++ calculate mean and standard-deviation ++
        mean, std = pd.Series(score_list).describe().loc[["mean", "std"]]
        samples, mean_f, std_f = score_list_flt.describe().loc[["count", "mean", "std"]]
        # ++ logging for full-score and each result ++
        self.logger.save_score(trial.number, hidden_units, score_list)
        self.logger.save_result(trial.number, hidden_units, samples, mean, std, mean_f, std_f)
        print(f"\r#{trial.number} -- unit: {hidden_units}, samples: {samples}/{self.sample}, mean: {mean:.4e}, std: {std:.4e}")
        return mean


class HiddenUnitsGenerator:
    def __init__(self, type):
        self.type = type

    def hidden_units_for_conv(self, trial):
        return [trial.suggest_int(f"num_unit{i + 1}", 1, self.max_units[i]) for i in range(len(self.max_units))]

    def hidden_units_for_prop(self, trial):
        unit1 = trial.suggest_int("num_unit1", 1, self.max_units[i])
        max_unit2 = int((self.num_elements - self.test_x.shape[1] * unit1) / unit1 + 1)
        if max_unit2 > 200:
            max_unit2 = 200
        unit2 = trial.suggest_int("num_unit2", 1, max_unit2)
        return [unit1, unit2]

    def create(self, trial):
        if self.type == "conv":
            hidden_units = self.hidden_units_for_conv(trial)
        elif self.type == "prop":
            hidden_units = self.hidden_units_for_prop(trial)
        return hidden_units


class LoggerClass:
    def __init__(self, full_sample, score_file_path, result_file_path):
        self.full_sample = full_sample
        self.file_path = {
            "score": score_file_path,
            "result": result_file_path
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
            header += f", Samples(Full:{self.full_sample:d})"
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
