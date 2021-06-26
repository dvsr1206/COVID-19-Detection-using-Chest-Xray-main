# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 05:05:36 2021

@author: anand
"""

# Import the necessary packages
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import jsonify
import requests
import pickle
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import pandas as pd
from keras.layers import *
from keras.models import *
from keras.metrics import *
import os

# Giving a folder for uploading files
UPLOAD_FOLDER = './upload'

# Configuring the Flask environment
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the model we saved and compile it
model = load_model('cnn-cxr-acc-98.44_bs-32_epochs-20.h5')
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

@app.route('/', methods = ['GET', 'POST'])
def Home():
    return render_template("index.html")

@app.route("/predict", methods = ['POST'])

def predict():
    # Getting form information
    if (request.method == "POST"):
        # Getting image from form
        file = request.files['Image']
        filename = secure_filename(file.filename)
        path = os.path.join(app.config['UPLOAD_FOLDER'], filename)     
        file.save(path)
        
        # Dimensions of our images
        img_width, img_height = 224, 224

        # Preprocessing the image
        img = image.load_img(path, target_size = (img_width, img_height))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        images = np.vstack([x])
        
        # Predicting the image class
        predict_class = model.predict(images)
        classified = predict_class[0][0]
        
        # Displaying the output dependino upon the predictions
        if (int(classified) == 0):
            return render_template('output.html', prediction_text = "Reslut: COVID +ve.", note = "The Model predicted this XRay as COVID.")
        else:
            return render_template('output.html', prediction_text = "Reslut: COVID -ve.", note = "The Model predicted this XRay as Non-COVID.")
            
        # Delete the uploaded file
        os.remove(path)
    else:
        return render_template('index.html')

if __name__ == "__main__":
    app.run(debug = True)
            
                                        