import json

import pandas as pd
from flask import jsonify
import pickle
import sklearn

class BodyFatPredictor:
    def __init__(self):
        self.model = None
        self.trans = None

    def predict_single_record(self, prediction_input):
        print(prediction_input)
        if self.model is None:
            self.model = pickle.load(open('model.pkl', 'rb'))
        print(json.dumps(prediction_input))
        df = pd.read_json(json.dumps(prediction_input), orient='records')
        print(df)

        # Pre-calculations made for the prediction
        df['Bmi'] = 703 * df['Weight'] / (df['Height'] * df['Height'])
        df['ACratio'] = df['Abdomen'] / df['Chest']
        df['HTratio'] = df['Hip'] / df['Thigh']
        df.drop(['Weight', 'Height', 'Abdomen', 'Chest', 'Hip', 'Thigh'], axis=1, inplace=True)

        # Transform to create a uniform dataset
        if self.trans is None:
            self.trans = pickle.load(open('transformer.pkl', 'rb'))

        df_t = self.trans.transform(df)

        # Prediction step
        y_pred = self.model.predict(df_t)
        print(y_pred)
        status = (y_pred > 0.5)
        print(type(status))
        # return the prediction outcome as a json message. 200 is HTTP status code 200, indicating successful completion
        return jsonify({'result': str(status[0])}), 200
