from operator import mod
import Utils
from ModelClass import Voc_Params
from ModelClass import Voc
from ModelClass import MyDataset
from ModelClass import Dataset_Params
from ModelClass import Seq2SeqModel
from flask import Flask
from flask_cors import cross_origin
import json

app = Flask(__name__)
model = Utils.loadModel()

@app.route("/")
def hello_world():
    return "<p>经典装样子</p>"

@app.route("/testDataset")
def test():
    Utils.testDataset(model)
    return "test success"

@app.route("/result/<input>")
@cross_origin()
def getResult(input):
    Utils.saveInput(input)
    result = Utils.getResult(model)
    return result

@app.route("/options/")
@cross_origin()
def getSelectOptions():
    return json.dumps(Utils.getOptions())



if __name__ == "__main__" :
    app.run(host='0.0.0.0')