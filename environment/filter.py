# -*- coding: utf-8 -*-
import pandas as pd


class FilterClass:
    def __init__(self):
        self.filter_type = "Super Class of FilterClass"

    def type(self):
        pass # print filter type of "filter class" object

    def filtering(self, data_list):
        pass # process of filtering


class Filter_with_IQR(FilterClass):
    def __init__(self):
        self.filter_type = "IQR Filter"

    def type(self):
        print(f"This Filter is {self.filter_type}")

    def filtering(self, data_list):
        pd_series = pd.Series(data_list)
        q1 = pd_series.quantile(.25)
        q3 = pd_series.quantile(.75)
        iqr = q3 - q1
        lim_upper = q3 + iqr * 1.5
        lim_lower = q1 - iqr * 1.5
        return pd_series[pd_series.apply(lambda x: lim_lower < x < lim_upper)]
    
    
class Filter_with_2sigma(FilterClass):
    def __init__(self):
        self.filter_type = "2-sigma Filter"

    def type(self):
        print(f"This Filter is {self.filter_type}")

    def filtering(self, data_list):
        pd_series = pd.Series(data_list)
        mean, std = pd_series.describe().loc[["mean", "std"]]
        lim_upper = mean + std * 2.0
        lim_lower = mean - std * 2.0
        return pd_series[pd_series.apply(lambda x: lim_lower < x < lim_upper)]