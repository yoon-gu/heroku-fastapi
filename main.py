from fastapi import FastAPI, Request
import joblib
from datetime import datetime
from pydantic import BaseModel
from typing import Optional, List

class Census(BaseModel):
    age: int
    workclass: str
    fnlgt: str
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: str


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
async def basic_predict(a: Census):
    # Getting the JSON from the body of the request
    return a
