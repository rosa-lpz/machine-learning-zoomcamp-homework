#from fileinput import filename
import pickle
from typing import Dict, Any #Literal
#from pydantic import BaseModel, Field


from fastapi import FastAPI
import uvicorn

app = FastAPI(title="conversion-prediction")



with open('model.bin', 'rb') as f_in:
    pipeline = pickle.load(f_in)



def predict_single(customer):
    # [customer 0, probability of converstion 1]
    result = pipeline.predict_proba(customer)[0, 1]
    return float(result)

#@app.post("/predict")
@app.post("/predict")
def predict(customer: Dict[str, Any]):
    converted = predict_single(customer)[0,1]
    return {
        "converted_probability": prob,
        "converted": bool(prob >= 0.5)
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9696)