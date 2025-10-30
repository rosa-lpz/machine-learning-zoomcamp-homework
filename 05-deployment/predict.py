from fileinput import filename
import pickle
from typing import Literal
from pydantic import BaseModel, Field


from fastapi import FastAPI
import uvicorn

app = FastAPI(title="churn-prediction")

@app.get("/ping")

with open(filename, 'rb') as f_in:
    pipeline = pickle.load(f_in)



def predict_single(customer):
    result = pipeline.predict_proba(customer)[0, 1]
    return float(result)


def predict(customer):
    converted = predict_single(customer)[0,1]
    return {
        "converted_probability": prob,
        "converted": bool(prob >= 0.5)
    }