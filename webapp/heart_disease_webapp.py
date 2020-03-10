from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)

# load model
model = pickle.load(open("model/rf_model_heart_disease.pkl", "rb"))


@app.route("/")
def status():
    return render_template("index.html")


@app.route("/predictions", methods=["POST"])
def predict():
    # save data from input in dictionary
    dict_input = request.form.to_dict()
    if len(dict_input) != 13:
        return render_template("index.html", prediction_text="Predicting not possible, all features needed.")

    # prep data for prediction
    X_test = [np.array([value for value in dict_input.values()])]

    # make predictions
    pred_proba = model.predict_proba(X_test)
    return render_template("index.html",
                           prediction_text=f"There is {pred_proba[0][1] * 100}% chance of having a heart disease.")


if __name__ == "__main__":
    app.run()
