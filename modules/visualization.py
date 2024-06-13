import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import learning_curve
import math

class Visualizer:
  def plot_correlation_matrix(self, correlation_matrix):
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Matrix')
    plt.show()

  def plot_learning_curve(self, estimator, title, X, y, ylim=None, cv=None, n_jobs=None, train_sizes=np.linspace(0.1, 1.0, 5)):
    plt.figure(facecolor='white')
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel('Training examples')
    plt.ylabel('Score')
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color='r')
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color='g')
    plt.plot(train_sizes, train_scores_mean, 'o-', color='r',
              label='Training score')
    plt.plot(test_scores_mean, 'o-', color='g',
              label='Cross-validation score')

    plt.legend(loc='best')
    return plt

  def plot_distributions(self, data, columns):
    num_columns = len(columns)
    grid_size = math.ceil(math.sqrt(num_columns))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(grid_size * 6, grid_size * 4))

    for i, column in enumerate(columns):
        row = i // grid_size
        col = i % grid_size
        self._plot_single_histogram(axes[row, col], data[column], column)

    # Remove any empty subplots
    for j in range(num_columns, grid_size * grid_size):
        fig.delaxes(axes[j // grid_size, j % grid_size])

    plt.tight_layout()
    return plt

  def show_plot(self, plot_instance):
    plot_instance.show()

  def _plot_single_histogram(self, ax, data, column_name):
    sns.histplot(data, kde=True, ax=ax)
    ax.set_title(f'Distribución de {column_name}')
    ax.set_xlabel(column_name)
    ax.set_ylabel('Frecuencia')
