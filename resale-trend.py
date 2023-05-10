#!/usr/bin/env python
# coding: utf-8

# # Singapore Project Resale Price Trend

# The goal for this notebook is to train a machine learning model to predict the price /sqft of a property type in a specified postal district given historical trends.
# 

# 
# 
# Load relevant data of excel sheet containing Residential Transactions

# In[2]:


import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Embedding, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np


# In[3]:


filename = 'ResidentialTransaction20230510192045.xlsx'
df = pd.read_excel(filename)

df = df.loc[:, ['Project Name', 'Transacted Price ($)', 'Area (SQFT)', 'Unit Price ($ PSF)', 'Sale Date', 'Property Type', 'Postal District']]
df = df[df['Project Name'] != '-']
df.head(10)


# In[4]:


scaler = MinMaxScaler()
input_data = scaler.fit_transform(df['Area (SQFT)'].values.reshape(-1, 1))
output_data = scaler.fit_transform(df['Unit Price ($ PSF)'].values.reshape(-1, 1))


# In[5]:


train_size = int(len(input_data) * 0.6)
train_input = input_data[:train_size]
train_output = output_data[:train_size]
test_input = input_data[train_size:]
test_output = output_data[train_size:]


# In[6]:


model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=[1]),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])


# In[7]:


model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=0.01))


# In[9]:


history = model.fit(train_input, train_output, epochs=300, validation_split=0.2)


# In[13]:


import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

data = pd.read_excel('ResidentialTransactions.xlsx')

data['Sale Year'] = pd.to_datetime(data['Sale Date'], format='%b-%y').dt.year
data.drop('Sale Date', axis=1, inplace=True)

encoder = LabelEncoder()
data['Property Type'] = encoder.fit_transform(data['Property Type'])

X = data[['Sale Year', 'Postal District', 'Property Type']]
y = data['Unit Price ($ PSF)']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(3,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(X_train, y_train, epochs=1000, batch_size=128)

loss = model.evaluate(X_test, y_test)
print(f'Mean squared error: {loss}')





# In[14]:


X_train


# In[22]:


input_data = {
    'Sale Year': [2023],
    'Postal District': [22],
    'Property Type': ['Apartment']
}


input_df = pd.DataFrame(input_data)
input_df['Property Type'] = encoder.transform(input_df['Property Type'])
input_scaled = scaler.transform(input_df)

prediction = model.predict(input_scaled)
print(f'Predicted Unit Price ($ PSF): {prediction[0][0]}')


# In[81]:


model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


# Utilize the Model here, after all previous cells have been run.

# In[25]:


print("Residential Transactions Predictive Analysis: ")
saleYear = input("Year of Transaction: ")
postalDistrict = input("Postal District of Property (eg. 19, 20, 21): ")
print("Type of Property")
print("1. Apartment")
print("2. Condominium")

residence_type = input("Enter 1 or 2 to select your residence type: ")

if residence_type == "1":
    propertyType = "Apartment"
elif residence_type == "2":
    propertyType = "Condominium"
else:
    print("Invalid input. Please enter either 1 or 2 to select your residence type.")

input_data = {
    'Sale Year': [saleYear],
    'Postal District': [postalDistrict],
    'Property Type': [propertyType]
}
input_df = pd.DataFrame(input_data)
input_df['Property Type'] = encoder.transform(input_df['Property Type'])
input_scaled = scaler.transform(input_df)

prediction = model.predict(input_scaled)
print(f'Predicted Unit Price ($ PSF): {prediction[0][0]}')


# In[ ]:




