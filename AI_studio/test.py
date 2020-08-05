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
    '''Loads machine learning model from a path.\n
    Example:\n
    \t model = load_model('model.dill')\n
    Keyword Arguments:\n
    path : Name of the binary from which the model should be loaded\n
    '''
    with open(path, 'rb') as in_strm:
        model = dill.load(in_strm)
    return model

def LinearRegression_Inference(path, X, **kwargs):
    '''Loads LinearRegression model (sklearn.linear_model.LinearRegression)
    from path. Uses this model on input features X to make and return
    predictions.\n
    Example:\n
    \t preds = LinearRegression_Inference('linear_regression_model.dill', X)\n
    Keyword Arguments:\n
    path : Name of the binary from which the model should be loaded. This
    should be a file path. (Ex. 'path/to/binary')\n
    X : Input Feature Vector for inference. This should be a Numpy array or a
    Python List. Must match shape of the data on which LinearRegression model
    was trained\n
    Returns :\n
    Predicted values (Inference Results) obtained after running the model on X.\n
    '''
    regressor = load_model(path)
    return regressor.predict(X)

def LogisticRegression_Inference(path, X, **kwargs):
    '''Loads LogisticRegression model (sklearn.linear_model.LogisticRegression)
    from path. Uses this model on input features X to make and return
    predictions.\n
    Example:\n
    \t preds = LogisticRegression_Inference('logistic_regression.dill', X)\n
    Keyword Arguments:\n
    path : Name of the binary from which the model should be loaded. This
    should be a file path. (Ex. 'path/to/binary')\n
    X : Input Feature Vector for inference. This should be a Numpy array or a
    Python List. Must match shape of the data on which LogisticRegression model
    was trained\n
    Returns :\n
    Predicted values (Inference Results) obtained after running the model on X.\n
    '''
    clf = load_model(path)
    return clf.predict(X)

def SVC_Inference(path, X, **kwargs):
    '''Loads a Support Vector Classifier model(sklearn.svm.svc) from path. Uses
    this model on input features X to make and return predictions.\n
    Example:\n
    \t preds = SVC_Inference('svc.dill', X)\n
    Keyword Arguments:\n
    path : Name of the binary from which the model should be loaded. This
    should be a file path. (Ex. 'path/to/binary')\n
    X : Input Feature Vector for inference. This should be a Numpy array or a
    Python List. Must match shape of the data on which the SVC model was
    trained.\n
    Returns :\n
    Predicted values (Inference Results) obtained after running the model on X.\n
    '''
    clf = load_model(path)
    return clf.predict(X)

def MLP_Inference(path, X, **kwargs):
    '''Loads a Multilayer Perceptron model
    (sklearn.neural_network.MLPClassifier) from path. Uses this model on
    input features X to make and return predictions.\n
    Example:\n
    \t preds = MLP_Inference('mlp.dill', X)\n
    Keyword Arguments:\n
    path : Name of the binary from which the model should be loaded. This
    should be a file path. (Ex. 'path/to/binary')\n
    X : Input Feature Vector for inference. This should be a Numpy array or a
    Python List. Must match shape of the data on which the MLP model was
    trained\n
    Returns :\n
    Predicted values (Inference Results) obtained after running the model on X.\n
    '''
    clf = load_model(path)
    return clf.predict(X)

def k_cluster_Inference(path, X, **kwargs):
    '''Loads k-Means clustering model (sklearn.linear_model.LogisticRegression)
    from path. Uses this model on input features X to cluster the input values.\n
    Example:\n
    \t clusters = k_cluster_Inference('k_cluster.dill', X)\n
    Keyword Arguments:\n
    path : Name of the binary from which the model should be loaded. This
    should be a file path. (Ex. 'path/to/binary')\n
    X : Input Feature Vector for inference. This should be a Numpy array or a
    Python List. Must match shape of the data on which k_cluster_Inference
    model was trained\n
    Returns :\n
    Predicted values of the clusters obtained after running the model on X.\n
    '''
    kmeans = load_model(path)
    return kmeans.predict(X)
