from flask import Flask, render_template, request, redirect, url_for, flash
import numpy as np
from joblib import load

app = Flask(__name__)


@app.route('/',methods=['POST','GET'])
def index():
    if request.method =='POST':
        day = request.form['day']
        month = request.form['month']
        forecast = request.form['forecast']
        systemload = request.form['systemload']
        smpea = request.form['smpea']
        temperature = request.form['temperature']
        windspeed = request.form['windspeed']
        co2int = request.form['co2int']
        actualwind = request.form['actualwind']
        loadsystem = request.form['loadsystem']
        print(day, month, forecast, systemload, smpea, temperature, windspeed, co2int, actualwind, loadsystem)
        model = load('model.joblib')
        print('loaded')
        # features = np.array([[9, 1, 54, 4241, 49, 9, 14, 491, 54, 4426]]) 
        features = np.array([[day, month, forecast, systemload, smpea, temperature, 
        windspeed, co2int, actualwind, loadsystem]]) 
        print(features)
        prediction = model.predict(features)
        print('done')
        print(prediction)
        return render_template('home.html',prediction=prediction)
    else:
        return render_template('home.html',prediction=0)


if __name__ == "__main__":
    app.run(debug=True)