import tensorflow as tf
import numpy as np
import json, os
import requests
import io
import h5py


def load_from_github(path):
    url = "https://github.com/22preich/-Deprehendere/blob/showcase2020/" + path + "?raw=true"
    print(f"retrieved {path} from github")
    r = requests.get(url)
    open(path, 'wb').write(r.content)
    # return h5py.File(io.BytesIO(requests.get(url).content), 'r')
    return path


models = {
    "id": tf.keras.models.load_model(load_from_github("models/id/idk-id.h5")),
    "gen": tf.keras.models.load_model(load_from_github("models/gen/15-altb-gen.h5")),
    "age": tf.keras.models.load_model(load_from_github("models/age/idk-long-age.h5"))
}


def get_indices(name):
    if name in ("id", "gen", "age"):
        with open(f"models/{name}/{name}-indices.json") as file:
            return json.load(file)


def get_label(name, model, img):
    pred = model.predict_on_batch(tf.expand_dims(img, 0))
    if name in ("id", "age"):
        return get_indices(name)[str(np.argmax(pred))], 0
    elif name == "gen":
        return get_indices("gen")[str(int(round(pred[0][0])))], pred[0][0] if not int(
            round(pred[0][0])) else 100 - pred[0][0]
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

    for mod in models.keys():
        labels = get_label(mod, models[mod], image)
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

    for mod in models.keys():
        labels = get_label(mod, models[mod], image)
        returns[mod]["class"], returns[mod]["confidence"] = labels[0], f"{labels[1] * 100}%"

    return returns


if __name__ == '__main__':
    predictFromPath()
