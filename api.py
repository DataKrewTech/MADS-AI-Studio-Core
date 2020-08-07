#from AI_studio import test, train
import flask
from flask import Flask, request
import pandas as pd
from AI_studio.train import SVC_train, LinearRegression_train, LogisticRegression_train, MLP_train, k_cluster_train
from AI_studio.test import load_model

app = Flask(__name__)

@app.route('/')
def home():
    return "Hi! This is the debug page..."

@app.route('/train', methods=['POST'])
def train():
    file_path = request.json['file_path']
    df = pd.read_csv(file_path)
    algo = request.json['algo']
    X_rows = request.json['X']
    if 'hyperparams' in request.json:
        hp = dict(request.json['hyperparams'])
    else:
        hp = {}
    if algo is not 'k_cluster':
        Y_rows = request.json['Y']
        y = df[Y_rows].to_numpy()
    model_name = request.json['model_name']
    X = df[X_rows].to_numpy()
    if algo == 'LinearRegression':
        LinearRegression_train(X, y, model_name, **hp)
    if algo == 'LogisticRegression':
        LogisticRegression_train(X, y, model_name, **hp)
    if algo == 'SVC':
        SVC_train(X, y, model_name, **hp)
    if algo == 'MLP':
        MLP_train(X, y, model_name, **hp)
    if algo == 'k_cluster':
        k_cluster_train(X, k, model_name, **hp)
    return flask.jsonify({
    'status' : 'success',
    'path' : model_name
    })

@app.route('/test', methods=['POST'])
def test():
    file_path = request.json['file_path']
    df = pd.read_csv(file_path)
    model_name = request.json['model_name']
    model = load_model(model_name)
    print('Loaded')
    preds = model.predict(df.values)
    print('Predicted')
    df['Predicted_Y_Data'] = preds
    df.to_csv(file_path.split('.')[0]+'_withpreds.csv', index = False)
    print('File Saved')
    return flask.jsonify({
    'status' : 'success',
    'path' : file_path+'_withpreds.csv'
    })

if __name__ == '__main__':
    app.run(debug=True)
