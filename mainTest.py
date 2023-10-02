import cv2
from keras.models import load_model
from PIL import Image
import numpy as np



INPUT_SIZE = 64


model = load_model('brainTumor10EpochsCategorical.keras')
image = cv2.imread('D:\\Python Programming A Practical Approach\\7th Sem Project\\Project\\datasets\\pred\\pred9.jpg')

img = Image.fromarray(image)
img = img.resize((64, 64))
img = np.array(img)
img = img.reshape(1, 64, 64, 3)

class_probabilities = model.predict(img)
# Find the class with the highest probability
results = np.argmax(class_probabilities)

# Print the predicted class
print(results)




