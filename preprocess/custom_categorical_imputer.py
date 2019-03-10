from sklearn.base import TransformerMixin
from sklearn.preprocessing import Imputer

class CustomCategoricalImputer(TransformerMixin):

    def __init__(self, cols=None):

        if not cols:
            raise ValueError("'{}' cannot be {}".format('cols', cols))

        if not isinstance(cols, list):
            raise TypeError("'{}' should be of {}".format('cols', list))

        self.cols = cols

    def _impute_null_per_category(self, col=None, X=None):
        """
        Return the column (col) in pd.Dataframe (X)
        imputed with mode value in missing rows

        Args:
            col: Name of the column with missing values
            X: Pandas Dataframe
        """

        dog_data = X[X['Type']==1]
        cat_data = X[X['Type']==2]

        _dog = dog_data[col].value_counts().index[0]
        _cat = cat_data[col].value_counts().index[0]

        # TODO: Try to replace below four lines of code with `fillna`

        # Finding the rows (indices) with missing values in a column
        dog_missing_indices = dog_data[dog_data[col].isnull()].index
        cat_missing_indices = cat_data[cat_data[col].isnull()].index

        # Imputing the missing values in column(s) with Mean / Median value per category
        X.loc[dog_missing_indices, col] = _dog
        X.loc[cat_missing_indices, col] = _cat

        return X[col]

    def transform(self, df):
        X = df.copy()

        for col in self.cols:
            X[col] = self._impute_null_per_category(col=col, X=X)

        return X

    def fit(self, *_):
        return self
