import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import joblib
import os
from matplotlib.colors import ListedColormap

plt.style.use("fivethirtyeight")

def prepare_data(df):
    """This method is used to seperate the dependent and independent variables.

    Args:
        df (pd.DataFrame): It takes in input pandas DataFrame object.

    Returns:
        tuple: It returns a tuple of dependent and independent variables.
    """
    X=df.drop('y', axis=1)
    y=df['y']
    return X, y

def save_model(model, filename):
    """This method saves the trained model.

    Args:
        model (python object): Trained model object.
        filename (str): Path to save the model.
    """
    model_dir="models/"
    os.makedirs(model_dir, exist_ok=True)
    filePath=os.path.join(model_dir, filename)
    joblib.dump(model, filePath)

def save_plot(df, filename, model):
    """This method saves the plot.

    Args:
        df (pd.DataFrame): pandas DataFrame.
        filename (str): Path to save the plot.
        model (python object): Trained model object.
    """
    # Internal funtions: 
    def _create_base_plot(df):
        df.plot(kind="scatter", x="x1", y="x2", c="y", s=100, cmap="winter")
        plt.axhline(y=0, color="black", linestyle="--", linewidth=1)
        plt.axvline(x=0, color="black", linestyle="--", linewidth=1)
        figure=plt.gcf() # Get current figure
        figure.set_size_inches(10, 8)    

    def _plot_decision_regions(X, y, classifier, resolution=0.02):
        colors=('red', 'blue', 'lightgreen', 'gray', 'cyan')
        n_classes=len(np.unique(y))
        cmap=ListedColormap(colors[: n_classes])

        X=X.values # Dataframe to array
        x1max, x1min=X[:, 0].max()+1, X[:, 0].min()-1
        x2max, x2min=X[:, 1].max()+1, X[:, 1].min()-1
        xx1, xx2=np.meshgrid(np.arange(x1min, x1max, resolution), np.arange(x2min, x2max, resolution))
    
        Z=classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
        Z=Z.reshape(xx1.shape)
        plt.contourf(xx1, xx2, Z, alpha=0.2, cmap=cmap)
        plt.xlim(xx1.min(), xx1.max())
        plt.ylim(xx2.min(), xx2.max())
        plt.plot()


    X, y=prepare_data(df)
    _create_base_plot(df)
    _plot_decision_regions(X, y, model)

    plot_dir="plots"
    os.makedirs(plot_dir, exist_ok=True)
    plotPath=os.path.join(plot_dir, filename)
    plt.savefig(plotPath)