from sklearn.base import TransformerMixin
from sklearn.preprocessing import MinMaxScaler

class CustomMinMaxScaler(TransformerMixin):
    def __init__(self, cols):
        self.cols = cols

    def transform(self, df=None):
        X = df.copy()
        _minmax_scaler = MinMaxScaler()

        for col in self.cols:
            X[self.cols] = _minmax_scaler.fit_transform(X[self.cols])
        return X

    def fit(self, *_):
        return self
