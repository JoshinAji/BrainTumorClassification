import cv2
from keras.models import load_model
from PIL import Image
import numpy as np
from sklearn.metrics import accuracy_score



INPUT_SIZE = 64


model = load_model('brainTumor10ClassificationEpochsCategoricalFinal3.keras')
image = cv2.imread('datasets\\Testing\\pituitary_tumor\\image(11).jpg')

img = Image.fromarray(image)
img = img.resize((64, 64))
img = np.array(img)
img = img.reshape(1, 64, 64, 3)

class_probabilities = model.predict(img)
# Find the class with the highest probability
results = np.argmax(class_probabilities)

# Print the predicted class
print(results)




