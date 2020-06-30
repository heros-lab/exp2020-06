import numpy as np
import pandas as pd


class DatasetManager:
    def __init__(self, dataset_directory):
        self.dir_path = dataset_directory
        
    def load(self, learn_name, test_name, x_index, y_index):
        if type(learn_name) == list:
            # create learing-data
            learn_x = pd.read_csv(f"{self.dir_path}/{learn_name[0]}_x.csv", index_col=0).values[:, x_index]
            learn_y = pd.read_csv(f"{self.dir_path}/{learn_name[0]}_y.csv", index_col=0).values[:, y_index]
            for label in learn_name[1:]:
                learn_x = np.vstack((learn_x, pd.read_csv(f"{self.dir_path}/{label}_x.csv", index_col=0).values[:, x_index]))
                learn_y = np.vstack((learn_y, pd.read_csv(f"{self.dir_path}/{label}_y.csv", index_col=0).values[:, y_index]))
            # create test-data
            test_x = pd.read_csv(f"{self.dir_path}/{test_name[0]}_x.csv", index_col=0).values[:, x_index]
            test_y = pd.read_csv(f"{self.dir_path}/{test_name[0]}_y.csv", index_col=0).values[:, y_index]
            for label in test_name[1:]:
                test_x = np.vstack((test_x, pd.read_csv(f"{self.dir_path}/{label}_x.csv", index_col=0).values[:, x_index]))
                test_y = np.vstack((test_y, pd.read_csv(f"{self.dir_path}/{label}_y.csv", index_col=0).values[:, y_index]))
        else:
            learn_x = pd.read_csv(f"{self.dir_path}/{learn_name}_x.csv", index_col=0).values[:, x_index]
            learn_y = pd.read_csv(f"{self.dir_path}/{learn_name}_y.csv", index_col=0).values[:, y_index]
            test_x = pd.read_csv(f"{self.dir_path}/{test_name}_x.csv", index_col=0).values[:, x_index]
            test_y = pd.read_csv(f"{self.dir_path}/{test_name}_y.csv", index_col=0).values[:, y_index]
            
        dataset = {
            "learn-x": learn_x,
            "learn-y": learn_y,
            "test-x": test_x,
            "test-y": test_y
        }
        
        return dataset
    