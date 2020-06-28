import numpy as np
import pandas as pd
import re

class DatasetManager:
    def __init__(self, dataset_directory, dataset_config):
        self.data_dir = dataset_directory
        self.data_conf = pd.read_csv(f"{self.data_dir}/dataset_config.csv", index_col=0)
        self.index_conf = pd.read_csv(dataset_config, index_col=0).replace(-1, np.nan)

    def select_index(self, model_type):
        if "conv." in model_type:
            model_id = re.compile("\d+").findall(model_type)[0]
            common_type = self.index_conf.columns.str.endswith(model_id)
            x_index = None
            y_index = self.index_conf.loc["y_index", common_type].values.astype("int32")
        else:
            x_index = self.index_conf[model_type].loc[["x_index"]].dropna().values.astype("int32")
            y_index = self.index_conf[model_type].loc[["y_index"]].dropna().values.astype("int32")
        return x_index, y_index

    def load_dataset(self, model_type, data_type):
        model_id = re.compile("\d+").findall(model_type)[0]  # extract model-ID from model-type
        data_name = self.data_conf[f"model{model_id}"].loc[data_type]
        data_x = pd.read_csv(f"{self.data_dir}/{data_name}_x.csv", index_col=0).values
        data_y = pd.read_csv(f"{self.data_dir}/{data_name}_y.csv", index_col=0).values
        print(f"Model-type: {model_type}, Load-data >> {data_name}")
        return data_x, data_y

    def learn_data(self, model_type):
        x_index, y_index = self.select_index(model_type)
        learn_x, learn_y = self.load_dataset(model_type, "learn-data")
        if x_index is not None:
            learn_x = learn_x[:, x_index]
            learn_y = learn_y[:, y_index]
        else:
            learn_x = learn_x
            learn_y = learn_y[:, y_index]
        print(f"Index: x >> {x_index}, y >> {y_index}")
        return learn_x, learn_y

    def test_data(self, model_type):
        x_index, y_index = self.select_index(model_type)
        test_x, test_y = self.load_dataset(model_type, "test-data")
        if x_index is not None:
            test_x = test_x[:, x_index]
            test_y = test_y[:, y_index]
        else:
            test_x = test_x
            test_y = test_y[:, y_index]
        print(f"Index: x >> {x_index}, y >> {y_index}")
        return test_x, test_y

if __name__ == '__main__':
    dataset_dir = "dataset"
    dataset_config = "conf/dataset_config.csv"
    dm = DatasetManager(f{data, "index_config.csv\")