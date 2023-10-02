import os
import tensorflow as tf
import numpy as np
from keras.preprocessing import image
from PIL import Image
import cv2
from keras.models import load_model
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename

app = Flask(__name__)

model = load_model('brainTumor10EpochsCategorical.keras')
print('Model Loded. Check http://127.0.0.1:5000/')

def get_className(classNo):
    if classNo == 0:
        return "Brain Tumor is not Present"
    elif classNo == 1:
        return "Brain Tumor is Present"
    
def getResult(img):
    image= cv2.imread(img)
    img = Image.fromarray(image)
    img = img.resize((64, 64))
    img = np.array(img)
    img = img.reshape(1, 64, 64, 3)

    class_probabilities = model.predict(img)
# Find the class with the highest probability
    results = np.argmax(class_probabilities)

# Print the predicted class
    print(results)
    
    #result=model.predict_classes(input_img)
    return results


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']

        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'datasets\pred', secure_filename(f.filename))
        f.save(file_path)
        print(file_path)
        value=getResult(file_path)
        result=get_className(value) 
        return result
    return None


if __name__ == '__main__':
    app.run(debug=True)
