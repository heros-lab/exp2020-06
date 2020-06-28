#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd
from copy import copy


class CorrelationAnalyzer:
    def __init__(self, data_list, data_dir, label_check: bool = True):
        self.data_list = data_list
        self.data_dir = data_dir
        self.df_x = []
        self.df_y = []
        self.df_list = []
        self.df_corr = []
        for data_name in self.data_list:
            self.df_x.append(pd.read_csv(f"{self.data_dir}/{data_name}_x.csv", index_col=0))
            self.df_y.append(pd.read_csv(f"{self.data_dir}/{data_name}_y.csv", index_col=0))
        x_label = list(self.df_x[0].columns)
        y_label = list(self.df_y[0].columns)
        self.label_organizer = LabelOrganizer(x_label, y_label, label_check)

    def evaluation(self):
        x_label, y_label = self.label_organizer.get_labels()
        self.df_list = []
        self.df_corr = []
        for i in range(len(self.df_x)):
            self.df_list.append(self.df_x[i][x_label].join(self.df_y[i][y_label]))
            self.df_corr.append(self.df_list[-1].corr()[x_label][len(x_label):])

        strong_corrs = sum(self.df_corr[i].apply(lambda x: np.abs(x) >= 0.7).values for i in range(len(self.df_corr)))
        self.prop_index = {label: np.where(strong_corrs[y_label.index(label)] > 0)[0] for label in y_label}

    def get_dataframe(self):
        return self.df_list

    def get_corr(self):
        return self.df_corr

    def get_index(self):
        return prop_index

    # Wrapper method for "Label organizer" object
    def get_labels(self):
        return self.label_organizer.get_labels()

    def check_labels(self):
        self.label_organizer.check_labels()

    def remove_xlabel(self, drop_label, label_check=False):
        self.label_organizer.remove_label("x-label", drop_label)
        if label_check:
            self.label_organizer.check_labels()

    def remove_ylabel(self, drop_label, label_check=False):
        self.label_organizer.remove_label("y-label", drop_label)
        if label_check:
            self.label_organizer.check_labels()

    def reset_xlabel(self, label_check=False):
        self.label_organizer.reset_label("x-label")
        if label_check:
            self.label_organizer.check_labels()

    def reset_ylabel(self, label_check=False):
        self.label_organizer.reset_label("y-label")
        if label_check:
            self.label_organizer.check_labels()

    def reset_labels(self, label_check=False):
        self.label_organizer.reset_label("both")
        if label_check:
            self.label_organizer.check_labels()


class LabelOrganizer:
    def __init__(self, x_label, y_label, label_check: bool = True):
        self.x_label_ref = x_label
        self.y_label_ref = y_label
        self.reset_label("both")
        if label_check:
            self.check_labels()

    def check_labels(self):
        print(f"x-label -> {self.x_label}")
        print(f"y-label -> {self.y_label}")

    def select_label(self, label_name):
        if label_name == "x-label":
            label_list = self.x_label
        elif label_name == "y-label":
            label_list = self.y_label
        else:
            print(f"Error: LabelOrganizer.select_label: {label_name} is not defined.")
        return label_list

    def remove_label(self, label_name, drop_label):
        label_list = self.select_label(label_name)
        if type(drop_label) is str:
            label_list.remove(drop_label)
        else:
            for label in drop_label:
                label_list.remove(label)
        return label_list

    def reset_label(self, label_name):
        if label_name == "x-label":
            self.x_label = copy(self.x_label_ref)
        elif label_name == "y-label":
            self.y_label = copy(self.y_label_ref)
        elif label_name == "both":
            self.x_label = copy(self.x_label_ref)
            self.y_label = copy(self.y_label_ref)
        else:
            print(f"Error: CorrelationAnalyzer.LabelOrganizer.reset_label: {label_name} is not defined.")

    def get_labels(self):
        return copy(self.x_label), copy(self.y_label)
