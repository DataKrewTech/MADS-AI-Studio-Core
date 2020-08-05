import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.neural_network import MLPClassifier
import dill
from sklearn.cluster import KMeans
#from statsmodels.tsa.arima_model import ARIMA

def save_model(model, path):
    '''Saves a given machine learning model to a path.\n
    Example:\n
    \t save_model(model, 'model.dill')\n
    Keyword Arguments:\n
    model : A trained model (sklearn)\n
    path : Name of the binary in which model will be saved on disk\n
    '''
    with open(path,'wb+') as f:
        dill.dump(model, f)

def load_model(path):
    '''Loads machine learning model to a path.\n
    Example:\n
    \t model = load_model('model.dill')\n
    Keyword Arguments:\n
    path : Name of the binary from which the model should be loaded
    '''
    with open(path, 'rb') as in_strm:
        model = dill.load(in_strm)
    return model

def LinearRegression_train(X, y, path = None, **kwargs):
    '''Trains a LinearRegression model on the input feature vectors (X) and the
    input labels (y). If path is specified (not None) then the model will be
    coverted to a binary file and saved in the location given by path.\n
    Example:\n
    \t model = LinearRegression_train(X, y, 'linear_regression_model.dill')\n
    Keyword Arguments:\n
    X : Input Feature Vectors for training. This should be a Numpy array or a
    Python List. The number of vectors should match the number of input labels.\n
    y : Input labels for training. This should be a Numpy array or a Python
    List.\n
    path :  This should be a file path where the model should be saved.
    (Ex. 'path/to/binary'). If it is not specified then the model will not be
    saved.\n
    Returns :\n
    Trained model of type sklearn.linear_model.LinearRegression\n
    '''
    regressor = LinearRegression(**kwargs)
    regressor.fit(X, y)
    if path is not None:
      save_model(regressor, path)
    return regressor

def LogisticRegression_train(X, y, path = None, **kwargs):
    '''Trains a LogisticRegression model on the input feature vectors (X) and
    the input labels (y). If path is specified (not None) then the model will
    be coverted to a binary file and saved in the location given by path.\n
    Example:\n
    \t model = LogisticRegression_train(X, y, 'logistic_regression_model.dill')\n
    Keyword Arguments:\n
    X : Input Feature Vectors for training. This should be a Numpy array or a
    Python List. The number of vectors should match the number of input labels.\n
    y : Input labels for training. This should be a Numpy array or a Python
    List.\n
    path :  This should be a file path where the model should be saved.
    (Ex. 'path/to/binary'). If it is not specified then the model will not be
    saved.\n
    Returns :\n
    Trained model of type sklearn.linear_model.LogisticRegression\n
    '''
    clf = LogisticRegression(**kwargs).fit(X, y)
    if path is not None:
        save_model(clf, path)
    return clf

def SVC_train(X, y, path = None, **kwargs):
    '''Trains a Support Vector Classification model on the input feature
    vectors (X) and the input labels (y). If path is specified (not None) then
    the model will be coverted to a binary file and saved in the location given
    by path.\n
    Example:\n
    \t model = SVC_train(X, y, 'svc_model.dill')\n
    Keyword Arguments:\n
    X : Input Feature Vectors for training. This should be a Numpy array or a
    Python List. The number of vectors should match the number of input labels.\n
    y : Input labels for training. This should be a Numpy array or a Python
    List.\n
    path :  This should be a file path where the model should be saved.
    (Ex. 'path/to/binary'). If it is not specified then the model will not be
    saved.\n
    Returns :\n
    Trained model of type sklearn.svm.SVC\n
    '''
    clf = svm.SVC(**kwargs).fit(X, y)
    if path is not None:
        save_model(clf, path)
    return clf

def MLP_train(X, y, path = None, **kwargs):
    '''Trains a Multilayer Perceptron model on the input feature vectors (X)
    and the input labels (y). If path is specified (not None) then the model
    will be coverted to a binary file and saved in the location given by path.\n
    Example:\n
    \t model = MLP_train(X, y, 'mlp_model.dill')\n
    Keyword Arguments:\n
    X : Input Feature Vectors for training. This should be a Numpy array or a
    Python List. The number of vectors should match the number of input labels.\n
    y : Input labels for training. This should be a Numpy array or a Python
    List.\n
    path :  This should be a file path where the model should be saved.
    (Ex. 'path/to/binary'). If it is not specified then the model will not be
    saved.\n
    Returns :\n
    Trained model of type sklearn.neural_network.MLPClassifier\n
    '''
    clf = MLPClassifier(**kwargs).fit(X, y)
    if path is not None:
        save_model(clf, path)
    return clf

def k_cluster_train(X, k, path = None, **kwargs):
    '''Trains a k-Means Clustering model on the input feature vectors (X).
    If path is specified (not None) then the model will
    be coverted to a binary file and saved in the location given by path.\n
    Example:\n
    \t model = k_cluster_train(X, 3, 'k_cluster_model.dill')\n
    Keyword Arguments:\n
    X : Input Feature Vectors for clustering. This should be a Numpy array or a
    Python List.\n
    k : The number of clusters to form.\n
    path :  This should be a file path where the model should be saved.
    (Ex. 'path/to/binary'). If it is not specified then the model will not be
    saved.\n
    Returns :\n
    Trained model of type sklearn.cluster.KMeans\n
    '''
    kmeans = KMeans(n_clusters = k, **kwargs).fit(X)
    if path is not None:
        save_model(kmeans, path)
    return kmeans

'''
To Be Implemented...

def ARIMA_predict(X, y, n, **kwargs):
    model = ARIMA([X,y], order=(0,0,1))
    model_fit = model.fit(disp = 0)
'''
