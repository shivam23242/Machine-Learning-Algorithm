#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


#firtly will read our dataset in our jupyter notebook with the help of pandas
dataset=pd.read_csv("C:/Users/shiva/Downloads/titanic.csv")


# In[3]:


#lest check the no. of rows and column present in our data
dataset.shape


# In[4]:


#now we will see first five rows of our data
dataset.head(5)


# In[22]:


#iloc- it helps us to select a value that belongs to a particlar row or column
#syntax:x=dataset.iloc[:,:-1].values  , we will take all colums excpet last one
        #y=dataset.iloc[:,-1].values  , we will access only last coloumn
x=dataset.iloc[:,[1,4,5,6,7]].values
y=dataset.iloc[:,1].values


# In[24]:


#splitting dataset to train and test
#train_test_split(X,Y,test_size=0.25,random_state=0)
#here 25% data is given to testing and 75% for trainingbb
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)


# In[25]:


#feature scaling(convert imbalanced scale data to balanced scale)
#standardization x'=x-miu/sigma
#normalization x'=x-xmin/xmax-xmin
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.fit_transform(x_test)
print(x_train)


# In[26]:


#algorithm=logistic regression
from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(x_train,y_train)


# In[27]:


#predicting whether passenger will survive or not
y_pred=model.predict(x_test)
y_pred


# In[31]:


#finding passenger will survive or not
places=int(input("enter the places:"))
age=eval(input("enter the age:"))
Spouses_Aboard=int(input("enter the spouses_abroad:"))
childern_abroad=int(input("enter the childern_abroad:"))
fare=eval(input("enter the fare:"))
newval=[[places,age,Spouses_Aboard,childern_abroad,fare]]
result=model.predict(sc.transform(newval))
print(result)
if result==1:
    print("passenger will survive")
else:
    print("passenger will not survive")


# In[33]:


print(model.intercept_)
print(model.coef_)


# In[34]:


y_pred = model.predict(x_test)
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
df

