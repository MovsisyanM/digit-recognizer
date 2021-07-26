import skimage.io as skio
import skimage.filters as skfltr
import skimage.color as skclr
import pickle
import pandas as pd
import numpy as np
from joblib import load
from PIL import Image, ImageOps
import urllib
from flask import Flask, request, Response
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
print("Loading model")
app = Flask(__name__)


# Choosing Otsu as the best threshold
def binarize_images(image_collection):
    """
    Inverts a collection of images and makes it binary
    """

    imges = []

    for img in image_collection:
        thresh = skfltr.threshold_mean(img)
        binary = img < thresh
        imges.append(binary)

    return imges


onevrest = pickle.load(open(r"..\models\ovr.pkl", "rb"))

print("Complete!")


@app.route("/", methods=["POST"])
def hello():
    global imgCounter
    data = request.data.decode("utf-8")

    img = Image.open(urllib.request.urlopen(data))
    print(img.size)
    img = np.array(binarize_images(
        [np.array(ImageOps.grayscale(img.resize((28, 28)))
                  .getdata())])).reshape(1, 28*28)

    pred = onevrest.predict(img)

    print(f"\n\nI guess {pred}\n\n")

    resp = Response(f'kk')
    resp.headers['Access-Control-Allow-Origin'] = '*'
    return resp


if __name__ == "__main__":
    app.run()
