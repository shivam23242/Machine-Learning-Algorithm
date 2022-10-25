#!/usr/bin/env python
# coding: utf-8

# In[1]:


#firstly we will import all the libraries that we will use
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


#import dateset with the pandas 
dataset=pd.read_csv('D:/Datasets/Housing.csv')


# In[3]:


dataset.head()


# In[4]:


sns.pairplot(dataset)


# In[5]:


dataset=dataset.replace(to_replace='yes',value=1)
dataset=dataset.replace(to_replace='no',value=0)
dataset=dataset.replace(to_replace='furnished',value=2)
dataset=dataset.replace(to_replace='semi-furnished',value=1)
dataset=dataset.replace(to_replace='unfurnished',value=0)
dataset


# In[6]:


dataset.shape


# In[7]:


features=dataset.iloc[:,1:].values
target=dataset.iloc[:,0].values


# In[8]:


x=features
y=target


# In[9]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.75,random_state=0)


# In[13]:


#from sklearn.preprocessing import StandardScaler
#sc=StandardScaler()
#x_train=sc.fit_transform(x_train)
#x_test=sc.fit_transform(x_test)


# In[10]:


from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(x_train,y_train)


# In[11]:


#The coefficient estimates for Ordinary Least Squares rely on the independence of the features
model.coef_


# In[12]:


model.intercept_


# In[13]:


y_pred=model.predict(x_test)
y_pred


# In[14]:


score=model.score(x_test,y_test)
score


# In[15]:


area=eval(input())
bedrooms=eval(input())
bathrooms=eval(input())
stories=eval(input())
mainroad=eval(input())
guestroom=eval(input())
basement=eval(input())
hotwaterheating=eval(input())
airconditioning=eval(input())
parking=eval(input())
prefarea=eval(input())
furnishingstatus=eval(input())
val=[[area,bedrooms,bathrooms,stories,mainroad,guestroom,basement,hotwaterheating,airconditioning,parking,prefarea,furnishingstatus]]
result=model.predict(val)
result


# In[16]:


df=pd.DataFrame({'Atcual': y_test,'predicted': y_pred})
df


# In[17]:


y_pred=model.predict(x_test)
y_pred=y_pred.round()
y_pred


# In[18]:


from sklearn.metrics import r2_score
r2=r2_score(y_test,y_pred)
r2


# In[19]:


from sklearn.metrics import mean_squared_error
mse=mean_squared_error(y_test, y_pred)
mse


# In[21]:


sns.lineplot(x=y_test,y=y_pred)

