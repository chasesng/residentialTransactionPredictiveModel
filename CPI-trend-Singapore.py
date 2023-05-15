#!/usr/bin/env python
# coding: utf-8

# # Singapore CPI Trend

# The goal for this notebook is to train a machine learning model to predict the CPI & Inflation Rate of Singapore when inputted with the year and quarter.
# 
# Below are two models trained with the same dataset for me to experiment utilizing slightly differing training.

# 
# 
# Load relevant data of excel sheet containing Singapore's Quarterly CPI and Inflation Rates

# ### Predictive Model for CPI and Rate of Inflation in Singapore (1)

# In[2]:


import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


# In[8]:


data = pd.read_excel('M213161.xlsx', names=['Year', 'Quarter', 'Inflation Rate'])

quarter_encoder = LabelEncoder()
data['Quarter'] = quarter_encoder.fit_transform(data['Quarter'])

X = data.drop('Inflation Rate', axis=1)
y = data['Inflation Rate']


ct = ColumnTransformer([('scaler', StandardScaler(), [0, 1])], remainder='passthrough')
X = ct.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# dense layer with relu activation
model = Sequential([
    layers.Dense(64, activation='relu', input_shape=[X_train.shape[1]]),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
])
model.compile(optimizer='adam', loss='mse')

model.fit(X_train, y_train, epochs=200, batch_size=8, verbose=0)


# In[9]:


year = input("Enter year: ")
quarter = input("Enter quarter (e.g. 1Q): ")
quarter_num = quarter_encoder.transform([quarter])[0]
input_data = ct.transform([[year, quarter_num]])
predicted_rate = model.predict(input_data)[0][0]
print(f"Predicted inflation rate: {predicted_rate:.2f}")


# ### Predictive Model for CPI and Rate of Inflation in Singapore (2)

# In[27]:


df = pd.read_excel('M213161.xlsx')

# To turn the labeled quarters into recognizable indexes
quarter_dict = {'1Q': 1, '2Q': 2, '3Q': 3, '4Q': 4}
df['Quarter'] = df['Quarter'].map(quarter_dict)


X = df[['Year', 'Quarter']]
y = df[['CPI', 'Inflation Rate']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# scale the data
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# dense layer without activation function allows output to be any real number
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=[2]),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(2)
])

model.compile(loss='mean_squared_error', optimizer='adam')

# training found to have minimal effect on loss after 200 iterations, staggering around 80
history = model.fit(X_train_scaled, y_train, epochs=200)

def predict(year, quarter):
    # check if input already exists in sheet, if it does then return it as is
    existing_data = df[(df['Year'] == year) & (df['Quarter'] == quarter_dict[quarter])]
    if not existing_data.empty:
        cpi = existing_data.iloc[0]['CPI']
        inflation_rate = existing_data.iloc[0]['Inflation Rate']
        return cpi, inflation_rate

    input_df = pd.DataFrame([[year, quarter_dict[quarter]]], columns=['Year', 'Quarter'])

    input_data = scaler.transform(input_df)

    prediction = model.predict(input_data)[0]
    cpi = prediction[0]
    inflation_rate = prediction[1]
    return cpi, inflation_rate


# In[41]:


output = predict(1962,'2Q')
cpiRate = output[0]
inflationFromXMinus1 = output[1]

print("The predicted CPI is " + str(cpiRate))
print("The inflation rate is at " + str(inflationFromXMinus1))
print("\nThe predicted values may not necessarily carry over to the immediate next quarter.\nThis model is trained to predict each quarter of each year individually.")


# In[ ]:




