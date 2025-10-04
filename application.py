from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app = application

## Import ridge regressor and standerd scaler pcikle
ridge_model = pickle.load(open('models/ridge.pkl', 'rb'))
standard_scaler = pickle.load(open('models/scaler.pkl', 'rb'))



@app.route("/")
def index():
    return render_template("index.html")

@app.route('/predictdata', methods=['POST', 'GET'])
def predict_datapoint():
    if request.method == 'POST':
        Temperature = float(request.form.get('Temperature'))
        RH = float(request.form.get('RH'))
        Ws = float(request.form.get('Ws'))
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        ISI = float(request.form.get('ISI'))
        Classes = float(request.form.get('Classes'))
        Region = float(request.form.get('Region'))

        input_data = pd.DataFrame([[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]],
                                  columns=['Temperature','RH','Ws','Rain','FFMC','DMC','ISI','Classes','Region'])

        new_data_scaled = standard_scaler.transform(input_data)
        result = ridge_model.predict(new_data_scaled)[0]


        return render_template('home.html',result = result)
    else:
        return render_template('home.html',result = None)


if __name__ == "__main__":
    app.run(host="0.0.0.0")
