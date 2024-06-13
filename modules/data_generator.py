import pandas as pd
import numpy as np
from sklearn.datasets import load_boston
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split

class Data_Module:
  def __init__(self, filepath, target_column='MEDV', degree=2, correlation_threshold=0.9):
    self.X_origin = None
    self.y_target = None
    self.boston = None
    self.filepath = filepath
    self.target_column = target_column
    self.degree = degree
    self.df = None
    self.datasets = {}
    self.X_poly_df = None
    self.y = None
    self.correlation_threshold = correlation_threshold

  def add_dataset(self, name, dataframe):
    self.datasets[name] = dataframe.copy()

  def check_missing_data(self):
    # Verificar la existencia de datos perdidos
    X_origin_missing_data = self.X_origin.isnull().sum()
    y_origin_missing_data = self.y_target.isnull().sum()

    return X_origin_missing_data, y_origin_missing_data

  def split_missing_data(self):
    # Dividir las observaciones con valores perdidos
    self.X_origin_missing = self.X_origin[self.X_origin.isnull().any(axis=1)].copy()
    self.y_target_missing = self.y_target[self.y_target.isnull()].copy()

  def prepare_data(self):
    # Eliminar duplicados si existen
    self.X_origin.drop_duplicates(inplace=True)
    self.y_target.drop_duplicates(inplace=True)

  def load_data(self):
    self.df = pd.read_csv(self.filepath)
    self.X_origin = self.df.drop(self.target_column, axis=1)
    self.y_target = self.df[self.target_column]

  def preprocess_data(self, dataset_name=''):

    full_data_set = self.df

    if dataset_name in self.datasets:
      full_data_set = self.datasets[dataset_name]

    X = full_data_set.drop(self.target_column, axis=1)

    self.y = full_data_set[self.target_column]

    poly = PolynomialFeatures(degree=self.degree)

    X_poly = poly.fit_transform(X)

    self.X_poly_df = pd.DataFrame(X_poly, columns=poly.get_feature_names_out(X.columns))

  def split_data(self, test_size=0.2, random_state=42):
      return train_test_split(self.X_poly_df, self.y, test_size=test_size, random_state=random_state)

  def calculate_correlation_matrix(self, dataset_name=''):

    full_data_set = self.df

    if dataset_name in self.datasets:
      full_data_set = self.datasets[dataset_name]

    self.correlation_matrix = full_data_set.corr()

    return self.correlation_matrix

  def remove_highly_correlated_features(self, dataset_name='', exceptions=None):

    if exceptions is None:
      exceptions = [self.target_column]

    if self.correlation_matrix is None:
      self.calculate_correlation_matrix(dataset_name)

    upper = self.correlation_matrix.where(np.triu(np.ones(self.correlation_matrix.shape), k=1).astype(np.bool))
    to_drop = [column for column in upper.columns if any(upper[column].abs() > self.correlation_threshold) and column not in exceptions]

    print(to_drop)

    if dataset_name in self.datasets:
      self.datasets[dataset_name] = self.datasets[dataset_name].drop(to_drop, axis=1)
    else:
      self.df = self.df.drop(to_drop, axis=1)

  def get_numeric_columns(self, dataset_name=''):
    full_data_set = self.df

    if dataset_name in self.datasets:
      full_data_set = self.datasets[dataset_name]

    return full_data_set.select_dtypes(include=[np.number, 'float64', 'int64']).columns

  def get_data(self, dataset_name=''):
    full_data_set = self.df

    if dataset_name in self.datasets:
      full_data_set = self.datasets[dataset_name]

    return full_data_set
