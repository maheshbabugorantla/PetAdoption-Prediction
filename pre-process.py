import csv
from pandas import read_csv


def print_classes_per_column(csv_file=None):

    df = read_csv(csv_file)

    # Print the No. of classes per each column
    columns = list(df.columns.values)

    for col in columns:
        column = df[col].unique()
        print("Classes in '{}' column are {}".format(col, set(column)))

def main():
    print_classes_per_column(csv_file='./data/train/train_new.csv')

if __name__ == "__main__":
    main()
