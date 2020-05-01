import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import utils
import numpy
import math 
import os
import cv2

# path to the dataset
paths = ['gestures']
x_train = []  # training lists
y_train = []
nb_classes=36
batch_size=32
nepoch=10

classes = {
    '0': 0,
    '1': 1,
    '2': 2,
    '3': 3,
    '4': 4,
    '5': 5,
    '6': 6,
    '7': 7,
    '8': 8,
    '9': 9,
    'a': 10,
    'b': 11,
    'c': 12,
    'd': 13,
    'e': 14,
    'f': 15,
    'g': 16,
    'h': 17,
    'i': 18,
    'j': 19,
    'k': 20,
    'l': 21,
    'm': 22,
    'n': 23,
    'o': 24,
    'p': 25,
    'q': 26,
    'r': 27,
    's': 28,
    't': 29,
    'u': 30,
    'v': 31,
    'w': 32,
    'x': 33,
    'y': 34,
    'z': 35,
}

def load_data_set():
    print("Creating dataset")
    for path in paths:
        for root, directories, filenames in os.walk(path):
            for filename in filenames:
                if filename.endswith(".jpg"):
                    fullpath = os.path.join(root, filename)
                    img = load_img(fullpath)
                    img = img_to_array(img)
                    x_train.append(img)
                    t = fullpath.rindex('\\')
                    fullpath = fullpath[0:t]
                    n = fullpath.rindex('\\')
                    y_train.append(classes[fullpath[n + 1:t]])
    print("Dataset created")

def train_model(model, X_train, Y_train):
    print('compiling model')
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss=keras.losses.categorical_crossentropy,
         optimizer=sgd,
         metrics=['accuracy'])
    print('compiled and  fitting')
    model.fit(X_train, Y_train, batch_size=batch_size, epochs=nepoch)
    print('Model trained')

def get_image_size():
    img = cv2.imread('gestures/1/10.jpg', 0)
    return img.shape




image_x, image_y = get_image_size()
def make_network(x_train):
    print('Creating network')
    classifier = Sequential([
        keras.layers.Conv2D(16, (2,2), input_shape=(image_x, image_y, 3), activation='relu'),
        keras.layers.Conv2D(16, (2,2), activation='relu'),
        keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'),
        keras.layers.Conv2D(32, (3,3), activation='relu'),
        keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(3, 3), padding='same'),
        keras.layers.Conv2D(64, (5,5), activation='relu'),
        keras.layers.MaxPooling2D(pool_size=(5, 5), strides=(5, 5), padding='same'),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(nb_classes, activation='softmax')
        ])
    print('Network created')
    return classifier


def trainData():
    load_data_set()
    print('Creating y_train')
    a = numpy.asarray(y_train)
    y_train_new = a.reshape(a.shape[0], 1)
    print(len(y_train))
    X_train = numpy.asarray(x_train).astype('float32')
    X_train = X_train / 255.0
    print(len(X_train))
    Y_train = tf.keras.utils.to_categorical(y_train_new, nb_classes)
    print('Creating Model')
    # run this if model is not saved.
    # model = make_network(numpy.asarray(x_train)) #to train model
    
    #for already loaded model use this to train for more example
    model = keras.models.load_model('keras.h5')
    print('Training Model')
    train_model(model,X_train,Y_train)
    model.save('keras.h5')
    print('Model saved')
    return model

model = trainData()
