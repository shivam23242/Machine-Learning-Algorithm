#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing the libraries we will use
#numpy and pandas are use tu creating array and dataframe and do many operations on dataset
import numpy as np
import pandas as pd
#train and test split use to spli our data into training and test set
from sklearn.model_selection import train_test_split
#LogisticRegression is used to buid our model
from sklearn.linear_model import LogisticRegression
#accuracy_score is used to find the accuracy of our model
from sklearn.metrics import accuracy_score


# data colection and preprocessing

# In[5]:


#loading our dataset with the help of pandas dataframes
dataset=pd.read_csv('E:/Ml_projects/Sonar Rock vs Mine prediction/Copy of sonar data.csv',header=None)


# In[6]:


#now check no. of rows and coulums in dataset
dataset.shape


# In[7]:


#now print first five row of the dataset
dataset.head()


# In[9]:


#get some statistical measures of the data
dataset.describe()


# In[12]:


#now seeing number of rock and mine given in data
dataset[60].value_counts()


# In[15]:


dataset.groupby(60).mean()


# In[35]:


x=dataset.drop(columns=60,axis=1).values
y=dataset[60]


# In[27]:


x


# In[19]:


y


# In[20]:


#now split data into x and y train
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=2)


# In[24]:


print(x.shape,x_train.shape,x_test.shape)


# In[21]:


#now training our model using logistic regression
model=LogisticRegression()
model.fit(x_train,y_train)


# In[22]:


#accuracy score of our model on training data
x_train_prediction=model.predict(x_train)
training_data_accuracy=accuracy_score(x_train_prediction,y_train)
print("accuracy score of our model on training data: ",training_data_accuracy)


# In[23]:


#accuracy score of our model on training data
x_test_prediction=model.predict(x_test)
test_data_accuracy=accuracy_score(x_test_prediction,y_test)
print("accuracy score of our model on training data: ",test_data_accuracy)


# In[46]:


x=(0.02,0.0371,0.0428,0.0207,0.0954,0.0986,0.1539,0.1601,0.3109,0.2111,0.1609,0.1582,0.2238,0.0645,0.066,0.2273,0.31,0.2999,0.5078,0.4797,0.5783,0.5071,0.4328,0.555,0.6711,0.6415,0.7104,0.808,0.6791,0.3857,0.1307,0.2604,0.5121,0.7547,0.8537,0.8507,0.6692,0.6097,0.4943,0.2744,0.051,0.2834,0.2825,0.4256,0.2641,0.1386,0.1051,0.1343,0.0383,0.0324,0.0232,0.0027,0.0065,0.0159,0.0072,0.0167,0.018,0.0084,0.009,0.0032)
input_data=np.asarray(x) #chaning data to a numpy array
#reshaping the np array as we are predicting for one instance
input_data=input_data.reshape(1,-1)
  
pred=model.predict(input_data)
print(pred)

if(pred=='R'):
    print('object is a rock')
else:
    print('objevt is a mine')

