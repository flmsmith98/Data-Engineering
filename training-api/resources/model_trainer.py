# MLP for Pima Indians Dataset saved to single file
# see https://machinelearningmastery.com/save-load-keras-deep-learning-models/
import logging
import os
import json
import numpy as np
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import pandas as pd
from flask import jsonify


def train(dataset):
    # split into input (X) and output (Y) variables
    X = df.drop(['BodyFat', 'Density'], axis=1)
    Y = df['Density']

    X['Bmi'] = 703 * X['Weight'] / (X['Height'] * X['Height'])
    X['ACratio'] = X['Abdomen'] / X['Chest']
    X['HTratio'] = X['Hip'] / X['Thigh']
    X.drop(['Weight', 'Height', 'Abdomen', 'Chest', 'Hip', 'Thigh'], axis=1, inplace=True)

    # Splitting the data for the model
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    #Transformer
    trans = PowerTransformer()
    X_train = trans.fit_transform(X_train)
    X_test = trans.transform(X_test)

    # define and fit model
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    # evaluate the model
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    text_out = {
        "R2:": r2,
        "RMSE": rmse,
    }
    logging.info(text_out)
    print(text_out)
    # Saving model in a given location provided as an env. variable
    model_repo = os.environ['MODEL_REPO']
    if model_repo:
        file_path_model = os.path.join(model_repo, "model.pkl")
        file_path_transformer = os.path.join(model_repo, "transformer.pkl")
        pickle.dump(model, open(file_path_model, 'wb'))
        pickle.dump(model, open(file_path_transformer, 'wb'))
        logging.info("Saved the model and the transformer to the location : " + model_repo)
        return jsonify(text_out), 200
    else:
        pickle.dump(model, open('model.pkl', 'wb'))
        pickle.dump(model, open('transformer.pkl', 'wb'))
        return jsonify({'message': 'The model and transformer were saved locally.'}), 200
