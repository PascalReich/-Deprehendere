import tensorflow as tf
import numpy as np
# import time
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
import json, os

"""
testdatagen = ImageDataGenerator(rescale=1./255)

# test_it = testdatagen.flow_from_directory('dbs/test', shuffle=False, target_size=(80, 80), class_mode='binary')
# print(test_it.next()[0].shape)
test_it = testdatagen.flow_from_directory('dbs/lfw-cropped', shuffle=False, target_size=(80, 80), class_mode='binary')

labels = dict((v,k) for k,v in test_it.class_indices.items())
"""
models = {
    "id": tf.keras.models.load_model("models/id/idk-id.h5"),
    "gen": tf.keras.models.load_model("models/gen/15-altb-gen.h5"),
    "age": tf.keras.models.load_model("models/age/idk-long-age.h5")

}


def get_indices(name):
    if name in ("id", "gen", "age"):
        with open(f"models/{name}/{name}-indices.json") as file:
            return json.load(file)


def get_label(name, model, img):
    pred = model.predict_on_batch(tf.expand_dims(img, 0))
    # print(get_indices("id")[str(np.argmax(pred))])
    if name in ("id", "age"):
        return get_indices(name)[str(np.argmax(pred))], 0  # pred[0][np.argmax(pred)].numpy()
    elif name == "gen":
        return get_indices("gen")[str(int(round(pred[0][0].numpy())))], pred[0][0].numpy() if not int(
            round(pred[0][0].numpy())) else 100 - pred[0][0].numpy()
    else:
        raise RuntimeError


def predictFromPath(img_path="dbs/test/Ariel_Sharon_0006.jpg"):
    path = tf.constant(img_path)
    image = tf.io.read_file(path)
    image = tf.io.decode_image(image)
    image = tf.image.resize(image, (80, 80))

    returns = {
        "id": {
            "class": None,
            "confidence": None
        },
        "gen": {
            "class": None,
            "confidence": None
        },
        "age": {
            "class": None,
            "confidence": None
        }
    }

    # print(image.shape)

    for mod in models.keys():
        labels = get_label(mod, models[mod], image)
        # print(labels[0], f"{labels[1] * 100}%")
        returns[mod]["class"], returns[mod]["confidence"] = labels[0], f"{labels[1] * 100}%"

    return returns, img_path


def predictFromTensor(tensor):
    image = tf.convert_to_tensor(tensor, dtype=tf.float32)
    image = tf.image.resize(image, (80, 80))

    returns = {
        "id": {
            "class": None,
            "confidence": None
        },
        "gen": {
            "class": None,
            "confidence": None
        },
        "age": {
            "class": None,
            "confidence": None
        }
    }

    # print(image.shape)

    for mod in models.keys():
        labels = get_label(mod, models[mod], image)
        # print(labels[0], f"{labels[1] * 100}%")
        returns[mod]["class"], returns[mod]["confidence"] = labels[0], f"{labels[1] * 100}%"

    return returns


if __name__ == '__main__':
    predictFromPath()
