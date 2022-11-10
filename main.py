from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
import joblib
from datetime import datetime
from pydantic import BaseModel, Field
from typing import Optional, List
import pandas as pd
import os

from starter.ml.data import process_data
from starter.ml.model import inference

class Census(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int
    marital_status: str = Field(alias='marital-status')
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int = Field(alias='capital-gain')
    capital_loss: int = Field(alias='capital-loss')
    hours_per_week: int = Field(alias='hours-per-week')
    native_country: str = Field(alias='native-country')


# Instantiate the app.
app = FastAPI()
rf_model = joblib.load('model/rf.joblib')
encoder = joblib.load('model/encoder.joblib')
lb = joblib.load('model/lb.joblib')

# Define a GET on the specified endpoint.
@app.get("/")
async def say_hello():
    return {"greeting": "Hello World!"}

@app.post('/predict')
async def basic_predict(data: Census):
    # Getting the JSON from the body of the request
    data_json = jsonable_encoder(data)
    df = pd.DataFrame(data=data_json.values(), index=data_json.keys()).T

    cat_features = [
            "workclass",
            "education",
            "marital-status",
            "occupation",
            "relationship",
            "race",
            "sex",
            "native-country",
        ]

    # load model
    model_dir = 'model'
    with open(os.path.join(model_dir, "rf.joblib"), 'rb') as f:
        model = joblib.load(f)
    with open(os.path.join(model_dir, "encoder.joblib"), 'rb') as f:
        encoder = joblib.load(f)
    with open(os.path.join(model_dir, "lb.joblib"), 'rb') as f:
        lb = joblib.load(f)

    X, _, _, _ = process_data(
        df,
        categorical_features=cat_features,
        encoder=encoder,
        lb=lb,
        training=False)

    preds = inference(model, X)
    prediction = {"prediction": lb.inverse_transform(preds)[0]}
    return prediction
