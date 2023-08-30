import pickle as pkl
import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)

with open("pipeline.pkl", "rb") as f:
    pipeline = pkl.load(f)

mapping = {
    0: "normal",
    1: "suspect",
    2: "bad"
}

@app.route("/", methods=["POST"])
def root():
    content = request.json
    data = pd.DataFrame([content], columns=pipeline["columns"])
    data = pipeline["scaler"].transform(data)
    [prediction] = pipeline["model"].predict(data)
    
     
    return { "prediction":mapping[prediction] }

    
