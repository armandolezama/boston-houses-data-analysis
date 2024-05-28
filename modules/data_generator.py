import pandas as pd
from sklearn.datasets import load_boston

class Data_Module:
  def __init__(self):
      self.X_origin = None
      self.y_target = None
      self.boston = None

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
