from flask import Flask, request, Response
import urllib
from PIL import Image, ImageOps
app = Flask(__name__)

imgCounter = 0


@app.route("/", methods=["POST"])
def hello():
    global imgCounter
    data = request.data.decode("utf-8")
    print(f"received image data\n\n{imgCounter}\n\n")

    img = Image.open(urllib.request.urlopen(data))
    print(img.size)
    img = ImageOps.grayscale(img.resize((28, 28)))
    img.save(f"images/drawn{imgCounter}.png")
    resp = Response(f'kk')
    resp.headers['Access-Control-Allow-Origin'] = '*'
    imgCounter += 1
    return resp


if __name__ == "__main__":
    app.run()
