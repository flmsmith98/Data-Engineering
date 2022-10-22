import json
import os

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
            model_repo = os.environ['MODEL_REPO']
            if model_repo:
                file_path = os.path.join(model_repo, "model.pkl")
                self.model = pickle.load(open(file_path, 'rb'))
            else:
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

        # Density prediction step and calculate bodyfat
        density = self.model.predict(df_t)
        fat = ((4.95/density[0]) - 4.5)*100
        # return the prediction outcome as a json message. 200 is HTTP status code 200, indicating successful completion
        return jsonify({'Density': density[0], 'Bodyfat': fat}), 200
