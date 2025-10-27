#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import sklearn
import pickle


# In[44]:


print(f'pandas=={pd.__version__}')
print(f'numpy=={np.__version__}')
print(f'sklearn=={sklearn.__version__}')


# In[45]:


from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression


# In[46]:


data_url = 'https://raw.githubusercontent.com/alexeygrigorev/datasets/master/course_lead_scoring.csv'
df = pd.read_csv(data_url)

df.columns = df.columns.str.lower().str.replace(' ', '_')
df


# In[47]:


# Function to fill missing values with a value
def fill_null_values(df, cat_fill_value, num_fill_value):
    cat_columns = df.select_dtypes(include=['object']).columns
    num_columns = df.select_dtypes(include=['int64','float64']).columns
    
    # Fill NaNs for categorical columns with the provided value
    df[cat_columns] = df[cat_columns].fillna(cat_fill_value)
    
    # Fill NaNs for numerical columns with the provided value
    df[num_columns] = df[num_columns].fillna(num_fill_value)
    
    return df


# In[48]:


df =fill_null_values(df=df, cat_fill_value='NA', num_fill_value=0.0)



# In[49]:


df_train=df.copy()
del df_train['converted']


# In[73]:


y_train= df.converted
y_train


# In[66]:


#categorical = list(df.select_dtypes(include=['object']).columns)
#numerical =list(df_train.select_dtypes(include=['int64','float64']).columns)
categorical = ['lead_source']
numerical = ['number_of_courses_viewed', 'annual_income']


# # Pipeline

# In[67]:


from sklearn.pipeline import make_pipeline


# In[74]:


#It's not convenient to deal with two objects: `dv` and `model`. 
#Let's combine them into one: 
pipeline = make_pipeline(
    DictVectorizer(),
    LogisticRegression(solver='liblinear')
)


# In[79]:


dv = DictVectorizer()
# Converts df to a list of dictionaries
train_dict = df[categorical + numerical].to_dict(orient='records')


# DicVectorizer dv converts df to a list of dictionaries  
#X_train = dv.fit_transform(train_dict)

# Model - Logistic Regression
#model = LogisticRegression(solver='liblinear')
#model.fit(X_train, y_train)

pipeline.fit(train_dict,y_train)


# In[80]:


train_dict[0]


# # Save model in pickle

# In[81]:


with open('model.bin','wb') as f_out:
    pickle.dump(pipeline, f_out)
with open('model.bin','rb') as f_in:
    pipeline = pickle.load(f_in)


# # Model for a customer

# In[84]:


customer={ 'lead_source': 'paid_ads',
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


# In[ ]:





# In[ ]:




