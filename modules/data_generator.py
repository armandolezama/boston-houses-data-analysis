import pandas as pd
from sklearn.datasets import load_boston
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split

class Data_Module:
  def __init__(self, filepath, target_column='MEDV', degree=2):
      self.X_origin = None
      self.y_target = None
      self.boston = None
      self.filepath = filepath
      self.target_column = target_column
      self.degree = degree
      self.df = None
      self.X_poly_df = None
      self.y = None

  def load_and_explore_data(self):
      # Cargar el conjunto de datos
      self.boston = load_boston()
      self.X_origin = pd.DataFrame(self.boston.data, columns=self.boston.feature_names)
      self.y_target = pd.Series(self.boston.target, name="MEDV")

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

  def preprocess_data(self):
      X = self.df.drop(self.target_column, axis=1)
      self.y = self.df[self.target_column]
      poly = PolynomialFeatures(degree=self.degree)
      X_poly = poly.fit_transform(X)
      self.X_poly_df = pd.DataFrame(X_poly, columns=poly.get_feature_names_out(X.columns))

  def split_data(self, test_size=0.2, random_state=42):
      return train_test_split(self.X_poly_df, self.y, test_size=test_size, random_state=random_state)
