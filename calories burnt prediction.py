#!/usr/bin/env python
# coding: utf-8
importing required libraries
# In[37]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn import metrics

#Data collection and processing
# In[8]:


#use pandas to load dataset
workout=pd.read_csv('E:/Ml_projects/calories burnt prediction/exercise.csv')


# In[9]:


workout.head(5)


# In[10]:


calories=pd.read_csv('E:/Ml_projects/calories burnt prediction/calories.csv')


# In[11]:


calories.head(5)


# In[13]:


dataset=pd.concat([workout,calories['Calories']],axis=1)


# In[14]:


dataset.head(5)


# In[16]:


#checking no. of rows and columns
dataset.shape


# In[18]:


#describing our datset
dataset.info()


# In[19]:


#now we will check for any null value present in our data
dataset.isnull().sum()


# In[21]:


#as we see there is no null value present then we dont need to handle missing value for our data

Data Analysis
# In[22]:


#get some statistical measures of the data
dataset.describe()


# Data Visualization

# In[23]:


sns.countplot(dataset['Gender'])


# In[24]:


sns.distplot(dataset['Age'])


# Finding correlation between the data

# In[25]:


correlation=dataset.corr()


# In[28]:


#constructing heatmap to understand the correlation
plt.figure(figsize=(10,10))
sns.heatmap(correlation,cbar=True,square=True,fmt='.1f',annot=True,annot_kws={'size':8},cmap='Blues')


# In[29]:


dataset.replace({"Gender":{'male':0,'female':1}},inplace=True)


# In[30]:


dataset.head(5)


# In[31]:


#seperating dataset into features and target
x=dataset.drop(columns=['User_ID','Calories'],axis=1)
y=dataset['Calories']


# In[32]:


x


# In[33]:


y


# In[34]:


#now split data into x and y train
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=5)


# In[35]:


print(x.shape,x_train.shape,x_test.shape)


# In[38]:


#training model
#xgboost regressor
model=xgb.XGBRegressor()


# In[39]:


model.fit(x_train,y_train)


# Evalutaion

# In[40]:


#predicting test data
y_predict=model.predict(x_test)


# In[41]:


y_predict


# In[43]:


#mean absolute error
mae=metrics.mean_absolute_error(y_test,y_predict)
mae


# In[47]:


y_predict = model.predict(x_test)
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_predict})
df

