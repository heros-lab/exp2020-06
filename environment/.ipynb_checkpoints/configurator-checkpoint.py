import pandas as pd


class ModelConfigurator:
    def __init__(self, config_path):
        self.conf = pd.read_csv(config_path, index_col=0)
        
    def decode_index(self, index_text):
        index_text = list(index_text[1:-1].split(","))
        return [int(x) for x in index_text]
    
    def get_index(self, model_name):
        x_index, y_index = self.conf[model_name].loc[["x-index", "y-index"]]
        x_index = self.decode_index(x_index)
        y_index = self.decode_index(y_index)
        return x_index, y_index
    
    def get_units(self, model_name):
        return self.conf[model_name].loc[["Unit1", "Unit2"]].astype("int").values
    
    def get_batch_size(self, model_name):
        return self.conf[model_name].loc[["Batch-size"]].astype("int").values[0]
    
    def get_config(self, model_name):
        x_index, y_index = self.get_index(model_name)
        num_units = self.get_units(model_name)
        batch_size = self.get_batch_size(model_name)
        conf = {
            "x-index": x_index,
            "y-index": y_index,
            "units": num_units,
            "batch-size": batch_size
        }
        return conf

    
class OptimizeConfigurator:
    def __init__(self, optimize_config_path, model_config_path):
        self.model_conf = ModelConfigurator(model_config_path)
        self.optimize_conf = pd.read_csv(optimize_config_path, index_col=0)
        
    def detect_model_type(self, model_name):
        return "Model-" + model_name[-1]
        
    def get_dataset(self, model_name):
        model_type = self.detect_model_type(model_name)
        learn_data = self.optimize_conf[model_type].loc["Learning-data"]
        test_data = self.optimize_conf[model_type].loc["Test-data"]
        return learn_data, test_data
        
    def get_unit_range(self, model_name):
        model_type = self.detect_model_type(model_name)
        return self.optimize_conf[model_type].loc[["Range-unit1", "Range-unit2"]].astype("int").values
    
    def get_elements(self, model_name):
        model_type = self.detect_model_type(model_name)
        return self.optimize_conf[model_type].loc[["Elements"]].astype("int").values[0]
    
    def get_config(self, model_name):
        return self.model_conf.get_config(model_name)
    
    # Wrapper method of ModelConfigurator
    def get_index(self, model_name):
        return self.model_conf.get_index(model_name)
    
    def get_units(self, model_name):
        return self.model_conf.get_units(model_name)
    
    def get_batch_size(self, model_name):
        return self.model_conf.get_batch_size(model_name)
    
    def get_model_config(self, model_name):
        return self.model_conf.get_config(model_name)
    
