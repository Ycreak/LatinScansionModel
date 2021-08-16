# Latin Scansion Model 2021
# This simple FLASK server interfaces with
# the OSCC and the LSM
from flask import Flask, request
from flask_cors import CORS, cross_origin
from flask_restful import Resource, Api
from flask_jsonpify import jsonify

from json import dumps

# INSTALL INSTRUCTIONS
# pipInstall flask flask_cors flask_restful flask_jsonpify

# RUN INSTRUCTIONS
# FLASK_APP=<filename>.py FLASK_ENV=development flask run --port 5002

# from tensorflow.keras import models
import numpy as np
import pickle
import configparser

import utilities as util

app = Flask(__name__)
api = Api(app)

CORS(app)

@app.route("/Get_neural_data")
def Get_neural_data():
    cf = configparser.ConfigParser()
    cf.read("config.ini")

    line_number = int(request.args.get('line_number'))
    book_number = int(request.args.get('book_number'))

    # with open('./pickle/X.npy', 'rb') as f:
        # X = np.load(f, allow_pickle=True)
    # with open('./pickle/y.npy', 'rb') as f:
        # y = np.load(f, allow_pickle=True)

    # model = models.load_model('./pickle/model')

    # # This works fine for binary classification
    # yhat = model.predict(X)

    # # Predict and test the first 10 lines. Also, print the similarity of predicted and expected
    # expected = y[line_number-1]

    df = util.Pickle_read(cf.get('Pickle', 'path'), cf.get('Pickle', 'prediction_df'))

    df_filtered = df[(df['book'] == book_number) & (df['line'] == line_number)].reset_index()

    predicted = df_filtered['predicted'][0]
    expected = df_filtered['expected'][0]
    syllables = df_filtered['syllable'][0]

    predicted_int = [round(num) for num in predicted]
    similarity, correct_list = Calculate_list_similarity(expected, predicted_int)

    confidence = [int(round(num*100)) for num in predicted]

    labels_predicted = ['—' if i==1 else '⏑' if i==0 else i for i in predicted_int]
    labels_expected = ['—' if i==1 else '⏑' if i==0 else i for i in expected]

    result = {
        "expected" : list(expected),
        "predicted" : list(predicted_int),
        "similarity" : similarity,
        "syllables" : syllables,
        "correct_list" : correct_list,
        "confidence" : confidence,
        "labels_predicted": labels_predicted,
        "labels_expected": labels_expected,
    }

    return jsonify(result)

def Calculate_list_similarity(list1, list2):
    # Calculates the similarity between two lists (entry for entry)
    score = 20

    correct_list = []

    for i in range(len(list1)):

        if list1[i] != list2[i]:
            score -= 1
            correct_list.append('orange')
        else:
            correct_list.append('lightgreen')

    score = int(score / len(list1) * 100)

    return score, correct_list

# MAIN
if __name__ == '__main__':
   app.run(host='0.0.0.0', port=5002)
