import numpy as np
import os
import config
import pickle
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.keras import Sequential, losses, optimizers, Model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D


def one_hot(Y):
    return np.vstack((np.array(np.where(Y == 1, 1, 0)), np.array(np.where(Y == 0, 1, 0))))


with open(config.TRAINING_FILE, "rb") as f:
    df = pickle.load(f)


im = plt.imread(f"{config.TRAINING_IMGS}20803.jpg")
img_width, img_height = im.shape
pixel_count = img_width * img_height

img_names = df['image_name'].values.reshape(1, df['image_name'].values.size)[0]
X_train = np.array([plt.imread(f"{config.TRAINING_IMGS}{img_name}") for img_name in img_names]) / 255
Y_train = np.where(df["Male"] == "male", 1, 0).T

model = Sequential()
model.add(Conv2D(32, (3, 3), activation=tf.nn.relu,  input_shape=(img_width, img_height, 1)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3), activation=tf.nn.relu))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation=tf.nn.relu))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dropout(.5))
model.add(Dense(128, activation=tf.nn.relu))
model.add(Dense(32, activation=tf.nn.relu))
model.add(Dense(2, activation=tf.nn.softmax))

loss = "categorical_crossentropy"
optimizer = optimizers.SGD(learning_rate=.01)
metrics = ["accuracy"]

model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
Y_one_hot = one_hot(Y_train).T
model.summary()
model.fit(X_train, Y_one_hot, verbose=1, batch_size=100, epochs=10)
