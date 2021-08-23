from sklearn.base import TransformerMixin, BaseEstimator
import pandas as pd
import numpy as np


class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        try:
            return X[self.columns]
        except KeyError:
            cols_error = list(set(self.columns) - set(X.columns))
            raise KeyError(f"DataFrame does not contains {cols_error}")
