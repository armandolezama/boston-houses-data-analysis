from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV

class Analysis:
  def __init__(self):
        self.linear_model = LinearRegression()
        self.ridge_model = Ridge()
        self.lasso_model = Lasso()

  def train_models(self, X_train, y_train):
      self.linear_model.fit(X_train, y_train)
      self.ridge_model.fit(X_train, y_train)
      self.lasso_model.fit(X_train, y_train)

  def evaluate_model(self, model, X_test, y_test):
      y_pred = model.predict(X_test)
      mse = mean_squared_error(y_test, y_pred)
      r2 = r2_score(y_test, y_pred)
      return mse, r2

  def hyperparameter_optimization(self, model, params, X_train, y_train, cv=5, scoring='r2'):
      grid = GridSearchCV(model, params, cv=cv, scoring=scoring)
      grid.fit(X_train, y_train)
      return grid.best_params_, grid.best_estimator_