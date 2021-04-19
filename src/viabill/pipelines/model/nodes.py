from typing import Any, Dict, Tuple

from sklearn.tree import DecisionTreeClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from itertools import compress
import pandas as pd
import numpy as np


class DecisionTreeDiscretizer_DF(BaseEstimator, TransformerMixin):
    """
    My own DecisionTreeDiscretizer
    !Rememeber to reset DF index before use!!!!!
    Discretize numeric values using DecisionTreeClassifier,
    Treats NULL are seprate category
    """
    def __init__(self, max_depth=3, min_samples_prc_leaf=0.1, bins=None,
                 **kwargs):
        self.max_depth = max_depth
        self.min_samples_prc_leaf = min_samples_prc_leaf
        self.bins = bins
        self.kwargs = kwargs

    def fit(self, x, y=None):

        # type(x)!=pd.core.frame.DataFrame:
        #    raise ValueError('{} works only with Pandas Data Frame')

        if type(x) == pd.core.frame.DataFrame:
            self.columnNames = x.columns
            self.numberofcolumns = x.shape[1]

        if type(x) == pd.core.series.Series:
            self.columnNames = [x.name]
            self.numberofcolumns = 1

        if type(y) == list:
            min_samples_leaf = int(self.min_samples_prc_leaf*len(y))
        else:
            min_samples_leaf = int(self.min_samples_prc_leaf*y.shape[0])

        self.trees = {}

        if not self.bins:
            self.bins = {}

        for nr_col, name in enumerate(self.columnNames):
            if name not in self.bins.keys():
                self.bins[name] = {}
            if self.numberofcolumns == 1:
                _df = x.copy()
            else:
                _df = x[name].copy()

            _df = _df.to_frame()
            _df['target'] = y
            _df_nona = _df.dropna().copy()

            if "bins" not in self.bins[name]:
                self.trees[name] = DecisionTreeClassifier(
                    criterion='gini',
                    random_state=666,
                    max_depth=self.max_depth,
                    min_samples_leaf=min_samples_leaf)

                # index 0 becaouse _df is only one feature and target
                self.trees[name].fit(_df_nona.iloc[:, 0].to_frame(), _df_nona['target'])

                self.bins[name]["bins"] = [-float("inf")] + \
                        list(sorted(set(self.trees[name].tree_.threshold)))[1:] + [float("inf")]
            else:
                self.bins[name]["bins"] = sorted(list(set(self.bins[name]["bins"])))
                if self.bins[name]["bins"][0] != -float("inf"):
                    self.bins[name]["bins"] = [-float("inf")]+self.bins[name]["bins"]
                if self.bins[name]["bins"][-1] != float("inf"):
                    self.bins[name]["bins"] = self.bins[name]["bins"]+[float("inf")]

            # create lower bin bound
            self.bins[name]["bins_l"] = np.array(self.bins[name]["bins"][:-1]).reshape(1, -1)
            # create upper bin bound
            self.bins[name]["bins_u"] = np.array(self.bins[name]["bins"][1:]).reshape(1,-1)
            # creat bin names
            self.bins[name]["bin_names"] = ["NULL"]+ \
                ["["+str(round(i[0], 5))+"<->"+str(round(i[1], 5))+")" for i in zip(
                    list(self.bins[name]["bins_l"].reshape(-1)),
                    list(self.bins[name]["bins_u"].reshape(-1)))]
        return self

    def get_feature_names(self):
        if hasattr(self, "columnNames"):
            return self.columnNames
        else:
            return None

    def transform(self, x):

        if type(x) == pd.core.frame.DataFrame:
            _transform_columnNames = x.columns
            _transform_numberofcolumns = x.shape[1]

        if type(x) == pd.core.series.Series:
            _transform_columnNames = [x.name]
            _transform_numberofcolumns = 1

        DF = pd.DataFrame()

        for nr_col, name in enumerate(_transform_columnNames):
            # select data to discretize and convert to np array
            if _transform_numberofcolumns == 1:
                _data_to_disc = np.array(x).reshape(-1, 1)
            else:
                _data_to_disc = np.array(x[name]).reshape(-1, 1)

            # opperation on arrays:
            # 1. If value is bigget then upper bound,
            # 2. If value is smaller then lower bpund
            # 3. Point column that reach 1 and 2 condition
            _selected_bin = np.logical_and(_data_to_disc >= self.bins[name]["bins_l"],
                                          _data_to_disc < self.bins[name]["bins_u"])

            _selected_bin_arg = np.argmax(_selected_bin,axis=1).reshape(-1,1)
            # fill nan with -99
            _selected_bin_arg[np.isnan(_data_to_disc)] = -99

            # create values to change
            if len(_selected_bin_arg[np.isnan(_data_to_disc)]) == 0:
                _old_values = [-99]+list(np.sort(np.unique(_selected_bin_arg)))
            else:
                _old_values = list(np.sort(np.unique(_selected_bin_arg)))

            _name_dic = {}

            for i in zip(_old_values,self.bins[name]["bin_names"]):
                _name_dic[i[0]] = i[1]

            _s = pd.Series(pd.Categorical(_selected_bin_arg[:,0], ordered=True))
            DF[name] = _s.cat.rename_categories(_name_dic)

        return DF