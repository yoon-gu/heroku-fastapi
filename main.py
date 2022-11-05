from fastapi import FastAPI, Request
import joblib
from datetime import datetime
from pydantic import BaseModel
from typing import Optional, List

class User(BaseModel):
    id: int
    name = 'John Doe'
    signup_ts: Optional[datetime] = None
    friends: List[int] = []


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
async def basic_predict(a: int, b: int, c: str):
    # Getting the JSON from the body of the request
    return str(rf_model)
