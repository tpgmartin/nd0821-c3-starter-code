# Put the code for your API here. 
from statistics import mode
from fastapi import FastAPI
import numpy as np
import os
import pandas as pd
import pickle
from pydantic import BaseModel, Field
from .starter.ml.data import process_data
from .starter.ml.model import inference

if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")

# Declare the data object with its components and their type.
class CensusItem(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int = Field(alias='education-num')
    marital_status: str = Field(alias='marital-status')
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int = Field(alias='capital-gain')
    capital_loss: int = Field(alias='capital-loss')
    hours_per_week: int = Field(alias='hours-per-week')
    native_country: str = Field(alias='native-country')
    salary: str

cat_features = [
    "workclass",
    "education",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native_country",
]

model_pth = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model/model.pkl')
encoder_pth = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model/encoder.pkl')
lb_pth = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model/lb.pkl')

with open(model_pth, 'rb') as f:
    model = pickle.load(f)
with open(encoder_pth, 'rb') as f:
    encoder = pickle.load(f)
with open(lb_pth, 'rb') as f:
    lb = pickle.load(f)

# Instantiate the app.
app = FastAPI()

@app.get("/")
async def say_hello():
    return {"greeting": "Hello World!"}

@app.post("/predict")
async def predict(item: CensusItem):

    X = pd.DataFrame(item.dict(), index=[0])

    X_test, _, _, _ = process_data(
        X, categorical_features=cat_features, label='salary', training=False, encoder=encoder, lb=lb
    )

    prediction = lb.inverse_transform(inference(model, X_test))[0]

    return {"prediction": prediction}
