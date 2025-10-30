"""
This code does two things
* Train the model
* Make predictions
"""


import pandas as pd
import numpy as np
import sklearn
import pickle

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline


print(f'pandas=={pd.__version__}')
print(f'numpy=={np.__version__}')
print(f'sklearn=={sklearn.__version__}')



# Function to load data
def load_data():
    data_url = 'https://raw.githubusercontent.com/alexeygrigorev/datasets/master/course_lead_scoring.csv'
    df = pd.read_csv(data_url)
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    return df



# Function to fill missing values with a value
def fill_null_values(df, cat_fill_value, num_fill_value):
    cat_columns = df.select_dtypes(include=['object']).columns
    num_columns = df.select_dtypes(include=['int64','float64']).columns
    
    # Fill NaNs for categorical columns with the provided value
    df[cat_columns] = df[cat_columns].fillna(cat_fill_value)
    
    # Fill NaNs for numerical columns with the provided value
    df[num_columns] = df[num_columns].fillna(num_fill_value)
    
    return df


# Function to train model
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
 



