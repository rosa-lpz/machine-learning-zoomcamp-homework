# Steps

## train.py
This code does two things
* Train the model
* Make predictions


# Function to train model
```python
def train_model(df):

    categorical = ['lead_source']
    numerical = ['number_of_courses_viewed', 'annual_income']
    
    y_train= df.converted
    # Converts df to a list of dictionaries
    train_dict = df[categorical + numerical].to_dict(orient='records')
    
    # Combine dv and model`nto one: 
    pipeline = make_pipeline(
        DictVectorizer(),
        LogisticRegression(solver='liblinear')
    )
    
    pipeline.fit(train_dict,y_train)
    return pipeline




def save_model(filename, model):
    with open(filename, 'wb') as f_out:
        pickle.dump(model, f_out)
    print(f'model saved to {filename}')


 


df = load_data()
df =fill_null_values(df=df, cat_fill_value='NA', num_fill_value=0.0)
pipeline = train_model(df)
save_model('model.bin',pipeline)

print('Model saved to model.bin')
```

## FastAPI 
Install fast api and uvicorn
```bash
pip install fastapi uvicorn
```

### Creation of a fast api application (ping.py)

```python
from fastapi import FastAPI
import uvicorn

app = FastAPI(title="ping")

@app.get("/ping")
def ping():
    return "PONG"

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9696)

```

## Creation of predict.py file
