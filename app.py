import pandas as pd
from flask import Flask,render_template, request
import pickle
import numpy as np

app = Flask(__name__)
data=pd.read_csv('Dataset\cleaned_data.csv')
pipe=pickle.load(open("RidgeModel.pkl",'rb'))

@app.route('/')
def index():
    
    locations=sorted(data['Location'].unique())
    return render_template('index.html',locations=locations)

@app.route('/predict',methods=['POST'])
def predict():
    location=request.form.get('Location')
    bhk=request.form.get('bhk')
    GYM=request.form.get('GYM')
    Area=request.form.get('Area')

    print(location,bhk,GYM,Area)
    input=pd.DataFrame([[location,Area,GYM,bhk]],columns=['Location','Area','GYM','bhk'])
    prediction=pipe.predict(input)[0]

    return str(np.round(prediction))

if __name__ == '__main__':
    app.run(debug=True)
