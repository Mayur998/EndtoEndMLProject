import pickle
from flask import Flask, request,app,jsonify,url_for,render_template,redirect,flash,session,escape
import numpy as np
import pandas as pd
import statsmodels.api as sm


app = Flask(__name__)   # staring point of application
## Load the model
pickled_model = pickle.load(open('regression.pkl','rb'))
scaling_model = pickle.load(open('scaling.pkl','rb'))

@app.route('/')
def home():
       return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
       data = request.json['data']
       input_list = [1]
       
       for i in data.keys():
              if i in ['temp', 'windspeed']:
                     input_list.append(data[i])
              else:
                     input_list.append(data[i])

       print(input_list)
       print(np.array(input_list).shape)
       prediction = pickled_model.predict(np.array(input_list))
       print(type(prediction))
       return str(prediction[0])

if __name__=="__main__":
       app.run(debug=True)

# input = np.array([1.        , 0.        , 1.        , 0.83178337, 0.08421855,
#        0.        , 0.        , 1.        , 0.        , 0.        ,
#        1.        ])
# print(pickled_model.predict(input))



