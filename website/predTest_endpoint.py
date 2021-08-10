import urllib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from tensorflow.keras.models import load_model
from sklearn.preprocessing import minmax_scale
from flask import Flask, Response, request

app = Flask(__name__)


# loading model
model = load_model("..\models\99.62\99.62.h5")
model.load_weights("..\models\99.62\checkpoint.hdf5")


def img_preproc(data):
    """Convert an image uri into an image, grayscale, invert, resize, minmax, flatten"""
    img = Image.open(urllib.request.urlopen(data))
    img = np.invert(np.array([np.array(
        ImageOps.grayscale(
            img.resize((28, 28))
        ).getdata())
    ]).reshape(28, 28))/255

    return minmax_scale(img).reshape(1, 28*28)


@app.route("/", methods=["POST"])
def digit_recognizer():
    data = request.data.decode("utf-8")

    img = img_preproc(data)
    pred = np.argmax(model.predict(img), axis=1)

    print(f"\n\n{pred}\n\n")

    resp = Response(f'kk')
    resp.headers['Access-Control-Allow-Origin'] = '*'
    return resp


if __name__ == "__main__":
    app.run()
