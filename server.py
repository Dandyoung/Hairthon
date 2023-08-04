from flask import Flask
import sys
application = Flask(__name__)

@application.route("/predict")
def hello():
    return "드디어 된다 ㅜㅜ"

if __name__ == "__main__":
    application.run(host='0.0.0.0', port=80)