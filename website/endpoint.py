from flask import Flask, request, Response
import urllib
app = Flask(__name__)


@app.route("/", methods=["POST"])
def hello():
    data = request.data.decode("utf-8")
    print("received image data")
    resp = Response(f'<img src="{data}"></img>')
    resp.headers['Access-Control-Allow-Origin'] = '*'
    return resp


if __name__ == "__main__":
    app.run()
