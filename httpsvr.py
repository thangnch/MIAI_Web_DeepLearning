import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import tensorflow as tf

from flask import Flask
from flask import request, jsonify
from flask_cors import CORS, cross_origin

import numpy as np
from keras.models import load_model
import cv2
import sys

# INIT
sess = tf.Session()
graph = tf.get_default_graph()


# Dinh nghia class
class_name = ['ok','xxx']

# Load model da train
with sess.as_default():
    with graph.as_default():
        my_model = load_model("cool_model.h5")

###### HTTP SERVER ############################################

app = Flask(__name__)
CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


@app.route('/')
@cross_origin(origin='*')
def index():
    return "Server is running!"

@app.route('/upload',methods=['POST'])
@cross_origin(origin='*')
def upload():
    global sess, graph, my_model
    if request.method == 'POST':
            f = request.files['file']
            image = cv2.imdecode(np.fromstring(f.read(), np.uint8), cv2.IMREAD_COLOR)

            image = cv2.resize(image, dsize=(200, 200))
            # Convert to tensor
            image = np.expand_dims(image, axis=0)

            with sess.as_default():
                with graph.as_default():
                    predict = my_model.predict(image)

                    print("This picture is: ", class_name[np.argmax(predict)])
                    # f.save(secure_filename(f.filename))
            return class_name[np.argmax(predict)]
    return "No file found"

if __name__ == '__main__':
   app.run(debug = True, port=8000)