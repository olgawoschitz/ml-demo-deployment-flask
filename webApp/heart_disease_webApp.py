from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)


@app.route("/")
def status():
    return render_template("index.html")

#
# def set_random(request_json):
#     """
#     set_random is a function to deal with missing data
#     :return: random DataFrame with all necessary features
#     """
#     random_df = {"age": np.arange(10, 100, 5),
#                  "sex": 1,
#                  "cp": np.arange(0, 4, 1),
#                  "trestbps": np.arange(120, 220, 5),
#                  "chol": 124,
#                  "fbs": 1,
#                  "restecg": 1,
#                  "thalach": 173,
#                  "exang": 0,
#                  "oldpeak": 0.2,
#                  "slope": 2,
#                  "ca": 1,
#                  "thal": 3}
#     input_data = {}
#     for key in random_df:
#         if key in request_json:
#             input_data[key] = [request_json[key]]
#         else:
#             input_data[key] = [random_df[key]]
#
#     return pd.DataFrame(input_data)


@app.route("/predictions", methods=["POST"])
def predict():
    # load model
    model = pickle.load(open("model/rf_model_heart_disease.pkl", "rb"))
    print("got model")
    # get data from input
    data = [np.array([int (x) for x in request.form.values()])]
    print("got data", data)
    # make predictions
    predictions = model.predict(data)
    print("Made predictions", predictions)

    out = predictions[0]
    # set response
    #response = {"heart disease": int(predictions[0])}
    return render_template("index.html", prediction_text=f"It is {out}")



if __name__ == "__main__":
    app.run()
