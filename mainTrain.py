import os
import cv2
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import normalize,to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, GlobalMaxPooling2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense


datasets = []
label = []


#Performing Data Extraction from the file
image_directory = "datasets/"
no_tumor_image = os.listdir(image_directory + "no/")
yes_tumor_glioma_image = os.listdir(image_directory + "yes/" + "glioma_tumor/")
yes_tumor_pituitary_image = os.listdir(image_directory + "yes/" + "pituitary_tumor/")
yes_tumor_meningioma_image = os.listdir(image_directory + "yes/" + "meningioma_tumor/")


INPUT_SIZE = 64

#print(no_tumor_image)
for i, image_name in enumerate(no_tumor_image):
    if(image_name.split('.')[1] == 'jpg'):
        image = cv2.imread(image_directory+"no/"+image_name)
        image = Image.fromarray(image, 'RGB') #Image editor that converst the image into array
        image = image.resize((INPUT_SIZE, INPUT_SIZE))
        datasets.append(np.array(image))
        label.append(0)

for i, image_name in enumerate(yes_tumor_glioma_image):
    if(image_name.split('.')[1] == 'jpg'):
        image = cv2.imread(image_directory+"yes/"+ "glioma_tumor/" + image_name)
        image = Image.fromarray(image, 'RGB') #Image editoor that converst the image into array
        image = image.resize((INPUT_SIZE, INPUT_SIZE))
        datasets.append(np.array(image)) #importing images as a numpy array into dataset list
        label.append(1)

for i, image_name in enumerate(yes_tumor_pituitary_image):
    if(image_name.split('.')[1] == 'jpg'):
        image = cv2.imread(image_directory+"yes/" + "pituitary_tumor/" + image_name)
        image = Image.fromarray(image, 'RGB') #Image editoor that converst the image into array
        image = image.resize((INPUT_SIZE, INPUT_SIZE))
        datasets.append(np.array(image)) #importing images as a numpy array into dataset list
        label.append(2)

for i, image_name in enumerate(yes_tumor_meningioma_image):
    if(image_name.split('.')[1] == 'jpg'):
        image = cv2.imread(image_directory+"yes/" + "meningioma_tumor/" + image_name)
        image = Image.fromarray(image, 'RGB') #Image editoor that converst the image into array
        image = image.resize((INPUT_SIZE, INPUT_SIZE))
        datasets.append(np.array(image)) #importing images as a numpy array into dataset list
        label.append(3)


#Preprocessing Step
datasets = np.array(datasets)
label = np.array(label)

# for i, label in enumerate(label):
#     print(label)


x_train, x_test, y_train, y_test = train_test_split(datasets, label, test_size=0.2, random_state=0)
print(x_train.shape) #No. of images on the train, size of both dimension (x and y), 3 channels(RGB)

#Reshape the input data to add the color channel dimension (3 for RGB)
x_train = x_train.reshape(-1, INPUT_SIZE, INPUT_SIZE, 3)
x_test = x_test.reshape(-1, INPUT_SIZE, INPUT_SIZE, 3)


x_train = normalize(x_train, axis=1)
x_test = normalize(x_test, axis=1)

# print(x_train.shape)
# print(y_train.shape)

y_train = to_categorical(y_train, num_classes=4)
y_test = to_categorical(y_test, num_classes=4)




# Model Building 
#64, 64, 3
model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=(INPUT_SIZE, INPUT_SIZE, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3,3), kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3,3), kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(4))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=8, 
          verbose=True, epochs=80, 
          validation_data=(x_test, y_test),
          shuffle=True)

model.save('brainTumor10ClassificationEpochsCategoricalFinal.keras')