from keras import models
from keras import Model
from keras import layers
from keras.datasets import reuters
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import tensorflow as tf
from keras.utils.np_utils import to_categorical
import lfw_input
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import os
from sklearn.model_selection import train_test_split
from keras.layers import Activation, Dense, Convolution2D, MaxPooling2D, Flatten, Input
from random import shuffle

label_dict = {}

def to_one_hot(labels, dimension=3023):
    results = np.zeros((len(labels), dimension))
    for i, label in enumerate(labels):
        results[i, label] = 1.
    return results

def return_full_name(list):
    value = ""
    for name in list:
        value += name
        if name != list[-1]:
            value += "_"
    return value


def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results


is_train = True

input_directory = "C:/Users/foggy/facialRecognition/lfw/"

model = models.Sequential()

model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(3023, activation='softmax'))
"""
x = Input((28, 28, 1))


y = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(x)
y = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(y)
y = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(y)
y = MaxPooling2D((2, 2))(y)
y = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(y)
y = MaxPooling2D((2, 2))(y)
y = Flatten()(y)
y = Dense(1024, activation='relu')(y)

digit1 = Dense(10, activation="softmax")(y)
digit2 = Dense(10, activation="softmax")(y)
digit3 = Dense(10, activation="softmax")(y)
digit4 = Dense(10, activation="softmax")(y)
digit5 = Dense(10, activation="softmax")(y)
model = Model(input=x, output=[digit1, digit2, digit3, digit4, digit5])
"""
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

onlyfiles = []

for r in os.listdir(input_directory):
    if len(os.listdir(input_directory + r)) < 50:
        continue
    for f in os.listdir(input_directory + r):
        if os.path.isfile(os.path.join(input_directory, r, f)):
            onlyfiles.append(os.path.join(input_directory, r, f))

shuffle(onlyfiles)

filelabels = []

for filename in onlyfiles:
    label = filename.split("\\")[-1]
    label = label.split("_")
    label.pop()
    label = return_full_name(label)
    isin = False
    for val in label_dict.items():
        if val[1] == label:
            isin = True
            filelabels.append(val[0])

    if not isin:
        leng = len(label_dict)
        label_dict[leng] = label
        filelabels.append(leng)

print(filelabels[1], label_dict[filelabels[1]], onlyfiles[1])

image_width = 180
image_height = 180
ratio = 2

image_width = int(image_width / ratio)
image_height = int(image_height / ratio)

channels = 3
nb_classes = 1

dataset = np.ndarray(shape=(len(onlyfiles), channels, image_height, image_width),
                     dtype=np.int8)


for i in range(len(onlyfiles)):
    img = load_img(onlyfiles[i])  # this is a PIL image
    img.thumbnail((image_width, image_height))
    # Convert to Numpy Array
    x = img_to_array(img)
    x = x.reshape((3, 90, 90))
    # Normalize
    x = (x - 128.0) / 128.0
    dataset[i] = x
    i += 1
    if i % 250 == 0:
        print("%d images to array" % i)
print("All images to array!")


train_data, test_data, train_labels, test_labels = train_test_split(dataset, filelabels, test_size=0.2, random_state=33)

indices = np.arange(train_data.shape[0])
np.random.shuffle(indices)


x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

one_hot_train_labels = to_one_hot(train_labels)
one_hot_test_labels = to_one_hot(test_labels)

print(len(x_train))
x_val = x_train[:200]
partial_x_train = x_train[700:]
y_val = one_hot_train_labels[:200]
partial_y_train = one_hot_train_labels[700:]

history = model.fit(x_train,
                    one_hot_train_labels,
                    epochs=1000,
                    batch_size=1024,
                    validation_data=(x_val, y_val))


loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.clf()
acc = history.history['acc']
val_acc = history.history['val_acc']
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

print(model.evaluate(x_test, one_hot_test_labels))
