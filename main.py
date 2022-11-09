from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
import joblib
from datetime import datetime
from pydantic import BaseModel
from typing import Optional, List
import pandas as pd

class Census(BaseModel):
    age: int = 39
    workclass: str = "State-gov"
    fnlgt: str = "77516"
    education: str = "Bachelors"
    education_num: int = 13
    marital_status: str = "Never-married"
    occupation: str = "Adm-clerical"
    relationship: str = "Not-in-family"
    race: str = "White"
    sex: str = "Male"
    capital_gain: int = 2174
    capital_loss: int = 0
    hours_per_week: int = 40
    native_country: str = "United-States"


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
    return df
