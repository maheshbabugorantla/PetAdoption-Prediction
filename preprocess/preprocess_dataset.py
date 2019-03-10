import csv
import pandas as pd
from sklearn.pipeline import Pipeline

from preprocess.custom_categorical_imputer import CustomCategoricalImputer
from preprocess.custom_quantitative_imputer import CustomQuantitativeImputer
from preprocess.custom_dummifier import CustomDummifier
from preprocess.custom_minmax_scaler import CustomMinMaxScaler


def load_training_dataset(dataset=None):
    """
    Returns the dataframe loading the csv dataset

    Keyword Arguments:
        dataset {str} -- Filepath to the training dataset

    Returns:
        pandas.DataFrame -- pandas DataFrame with csv dataset
    """

    if not dataset:
        raise ValueError("'dataset' cannot be None or Empty")

    if not isinstance(dataset, str):
        raise TypeError("'{}' should be of {}".format('dataset', str))

    return pd.read_csv(dataset)

def print_classes_per_column(dataframe=None):
    """
        Prints distinct values per column in the dataframe
    """

    # print the No. of classes per each column
    columns = list(dataframe.columns.values)

    for column in columns:
        classes = dataframe[column].unique()
        print("Classes in '{}' column are {}".format(column, set(classes)))

def get_missing_values_per_column(dataframe=None):
    """
        Returns a count of rows in each column that are empty in a dataframe
    """
    return dataframe.isnull().sum()

def drop_columns_from_dataframe(dataframe=None, columns=None):
    """
    Drops a list of columns in the dataframe

    Keyword Arguments:
        dataframe {pandas.DataFrame} -- Pandas DataFrame (default: {None})
        columns {list} -- list of columns in a pandas DataFrame (default: {None})

    Returns:
        pandas.DataFrame -- DataFrame after dropping the columns
    """

    if not columns:
        return dataframe
    return dataframe.drop(labels=columns, axis=1)

def replace_with_none(value=None, column=None, dataframe=None):
    """
    Replaces the value in column inside a dataframe with None

    Keyword Arguments:
        value {type} -- default None value in a column (default: {None})
        column {str} -- Label name of the column (default: {None})
        dataframe {pandas.DataFrame} -- Pandas DataFrame (default: {None})
    """
    dataframe[column] = dataframe[column].map(lambda x: x if x != value else None)

def preprocess_dataset(dataset=None):

    """
    Preprocesses the dataset

    Keyword Arguments:
        dataset {str} -- filepath to the dataset

    Returns:
        str -- filepath to the preprocessed data
    """


    if not dataset:
        raise ValueError("'dataset' file path cannot be None or Empty")


    # Load the Training dataset
    dframe = load_training_dataset(dataset='./data/train/train.csv')

    # Returns the missing rows per each column in the dataset
    get_missing_values_per_column(dataframe=dframe)

    drop_columns = ['Name', 'PetID', 'Fee', 'State', 'RescuerID', 'VideoAmt', 'PhotoAmt', 'Description']

    # Drop Irrelevant columns from the dataframe
    dframe = drop_columns_from_dataframe(dataframe=dframe, columns=drop_columns)

    # Convert Age from months => years
    dframe['Age'] /= 12

    # Replacing Age: 0 => None
    replace_with_none(value=0, column='Age', dataframe=dframe)

    # Replacing Breed1: 0 => None
    replace_with_none(value=0, column='Breed1', dataframe=dframe)

    # Replacing Color2 and Color3: 0 => None
    replace_with_none(value=0, column='Color2', dataframe=dframe)
    replace_with_none(value=0, column='Color3', dataframe=dframe)

    # Imputing the missing values with appropriate value imputer methods
    imputer = Pipeline([('quant', CustomQuantitativeImputer(cols=['Age'])), ('category', CustomCategoricalImputer(cols=['Breed1', 'Color2', 'Color3']))])
    df_imputed = imputer.fit_transform(dframe)

    # dog_breeds = df_imputed[(df_imputed['Breed1']>=1) & (df_imputed['Breed1']<=241)]
    # cat_breeds = df_imputed[(df_imputed['Breed1']>=242) & (df_imputed['Breed1']<=307)]

    # Correcting anomalies in the data
    df_imputed[(df_imputed['Breed1']>=1) & (df_imputed['Breed1']<=241) & (df_imputed['Type'] == 2)] = 1
    df_imputed[(df_imputed['Breed1']>=242) & (df_imputed['Breed1']<=307) & (df_imputed['Type'] == 1)] = 2

    categorical_columns = [
        'Type', 'Breed1', 'Breed2', 'Gender', 'Color1',
        'Color2', 'Color3', 'MaturitySize', 'FurLength',
        'Vaccinated', 'Sterilized', 'Dewormed', 'Health', 'Quantity']

    custom_dummifier = CustomDummifier(cols=categorical_columns)
    df_imputed = custom_dummifier.fit_transform(df_imputed)

    minmax_scaler = CustomMinMaxScaler(cols=['Age'])
    df_imputed = minmax_scaler.fit_transform(df_imputed)

    dataset_ = dataset[:-4]

    df_imputed.to_csv('{}_scaled.csv'.format(dataset_), index=False)

    return '{}_scaled.csv'.format(dataset_)
