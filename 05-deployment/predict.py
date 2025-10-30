import pickle
from typing import Literal
from pydantic import BaseModel, Field


from fastapi import FastAPI
import uvicorn



class Customer(BaseModel):
    'lead_source': 'paid_ads',
 'number_of_courses_viewed': 1,
 'annual_income': 79450.0,
 'interaction_count': 4,
 'lead_score': 0.94 }

#X = dv.transform(customer)

# predict probability of churning - 54.15 %
converted = pipeline.predict_proba(customer)[0,1]

print('Prob of convert: ',converted)

if converted>=0.5:
    print("send email with promo")
else:
    print("don't do anything")


class PredictResponse(BaseModel):
    churn_probability: float
    churn: bool


app = FastAPI(title="customer-churn-prediction")

with open('model.bin', 'rb') as f_in:
    pipeline = pickle.load(f_in)


def predict_single(customer):
    result = pipeline.predict_proba(customer)[0, 1]
    return float(result)


@app.post("/predict")
def predict(customer: Customer) -> PredictResponse:
    prob = predict_single(customer.model_dump())

    return PredictResponse(
        churn_probability=prob,
        churn=prob >= 0.5
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9696)