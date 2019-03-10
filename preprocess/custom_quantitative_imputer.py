from sklearn.base import TransformerMixin
from sklearn.preprocessing import Imputer

class CustomQuantitativeImputer(TransformerMixin):

    def __init__(self, cols=None, strategy='mean'):

        if not cols:
            raise ValueError("'{}' cannot be {}".format('cols', None))

        if not strategy:
            raise ValueError("'{}' cannot be {}".format('strategy', None))

        if not isinstance(cols, list):
            raise TypeError("'{}' should be of {}".format('cols', list))

        if not isinstance(strategy, str):
            raise TypeError("'{}' should be of {}".format('strategy', str))

        self.cols = cols
        self.strategy = strategy

    def _impute_null_with_strategy(self, col=None, X=None):

        # Categorising the data into Dogs and Cats
        dog_data = X[X['Type']==1]
        cat_data = X[X['Type']==2]

        _dog, _cat = None, None

        # Finding per category Mean / Median
        if self.strategy == 'median':
            _dog = dog_data[col].median()
            _cat = cat_data[col].median()
        else:
            _dog = dog_data[col].mean()
            _cat = cat_data[col].mean()

        # Finding the rows (indices) with missing values in a column
        dog_missing_indices = dog_data[dog_data[col].isnull()].index
        cat_missing_indices = cat_data[cat_data[col].isnull()].index

        # Imputing the missing values in column(s) with Mean / Median value per category
        X.loc[dog_missing_indices, col] = _dog
        X.loc[cat_missing_indices, col] = _cat

        return X[col]

    def transform(self, df):
        X = df.copy()

        # TODO: Impute the missing values using self.strategy in each column
        for col in self.cols:
            X[col] = self._impute_null_with_strategy(col=col, X=X)

        return X

    def fit(self, *_):
        return self
