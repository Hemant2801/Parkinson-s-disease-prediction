#!/usr/bin/env python
# coding: utf-8

# # Importing the necessary dependencies

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score


# #  Data collection and preprocessing

# In[2]:


df = pd.read_csv('C:/Users/Hemant/jupyter_codes/ML Project 1/Parkinsons disease detection/parkinsons.csv')


# In[3]:


#print the first 5 rows of the dataset
df.head()


# In[4]:


#print the last 5 rows of the dataset
df.tail()


# In[5]:


# shape of the dataset
df.shape


# In[6]:


# getting some info about the datset
df.info()


# In[7]:


# checking for any missing value
df.isnull().sum()


# In[8]:


# Some statistical measure of the dataset
df.describe()


# In[9]:


# distribution of target variable
df['status'].value_counts()


# 0 ---> Parkinson's positive
# 
# 1 ---> Healthy

# In[10]:


# grouing the data based on target variable
df.groupby('status').mean()


# # Data preprocessing

# Separating features and labels

# In[11]:


X = df.drop(columns = ['name', 'status'], axis = 1)
Y = df['status']


# In[12]:


print(X)
print(Y)


# Splitting the data into training and testing data

# In[13]:


x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = .2, stratify = Y, random_state = 2)


# In[14]:


print(x_train.shape, x_test.shape)
print(y_train.shape, y_test.shape)


# Data standardization

# In[15]:


scaler = MinMaxScaler()


# In[16]:


scaler.fit(x_train)


# In[17]:


x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


# In[18]:


print(x_train)


# # Model training

# support vector machine model

# In[19]:


model = SVC(kernel = 'linear')


# In[20]:


model.fit(x_train, y_train)


# Model evaluation:
# 
# accuracy score

# In[21]:


#model evaluation on training data
training_prediction = model.predict(x_train)

training_accuracy = accuracy_score(y_train, training_prediction)
print('THE ACCURACCY ON TRAINING DATA IS :', training_accuracy)


# In[22]:


#model evaluation on testing data
testing_prediction = model.predict(x_test)

testing_accuracy = accuracy_score(y_test, testing_prediction)
print('THE ACCURACCY ON TESTING DATA IS :', testing_accuracy)


# # Building a predictive system

# In[30]:


input_data = input()

input_list = [float(i) for i in input_data.split(',')]
input_array = np.asarray(input_list)

reshaped_array = input_array.reshape(1, -1)

str_data = scaler.transform(reshaped_array)

prediction = model.predict(str_data)
print('\n',prediction)
if prediction == 1:
    print('THE PERSON HAS PARKINSON\'S DISEASE')
else:
    print('THE PERSON IS HEALTHY')


# In[ ]:




