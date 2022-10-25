#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing the libraries we will use
#numpy and pandas are use tu creating array and dataframe and do many operations on dataset
import numpy as np
import pandas as pd
#for visualizing our data we import pyplot from matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
#train and test split use to spli our data into training and test set
from sklearn.model_selection import train_test_split
#LogisticRegression is used to buid our model
from sklearn.ensemble import RandomForestClassifier
#accuracy_score is used to find the accuracy of our model
from sklearn.metrics import accuracy_score


# data colection and preprocessing

# In[3]:


#loading our dataset with the help of pandas dataframes
dataset=pd.read_csv('E:/Ml_projects/wine quality prediction/winequality-red.csv')


# In[4]:


#now check no. of rows and coulums in dataset
dataset.shape


# In[5]:


#now print first five row of the dataset
dataset.head()


# In[6]:


#now check for missing values in dataset
dataset.isnull().sum()


# In[ ]:


#as we see there is no null value present then we dont need to handle missing value for our data


# data analysis and visualization

# In[7]:


#get some statistical measures of the data
dataset.describe()


# In[8]:


#number of value for each quality
sns.catplot(x='quality',data=dataset,kind='count')


# In[10]:


#volatile acidity vs quality 
plot=plt.figure(figsize=(5,5))
sns.barplot(x='quality',y='volatile acidity',data=dataset)


# In[11]:


#citric acid vs quality 
plot=plt.figure(figsize=(5,5))
sns.barplot(x='quality',y='citric acid',data=dataset)


# In[12]:


#correlation
corr=dataset.corr()


# In[13]:


#constructing heatmap to understand the correlation
plt.figure(figsize=(10,10))
sns.heatmap(corr,cbar=True,square=True,fmt='.1f',annot=True,annot_kws={'size':8},cmap='Blues')


# In[17]:


#splitting data into features and target
x=dataset.drop(columns='quality',axis=1).values


# In[18]:


x


# label binarizing

# In[21]:


y=dataset['quality'].apply(lambda y: 1 if y>=7 else 0)


# In[22]:


y


# In[23]:


#now split data into x and y train
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=2)


# In[24]:


print(x.shape,x_train.shape,x_test.shape)


# In[25]:


#now training our model using logistic regression
model=RandomForestClassifier()
model.fit(x_train,y_train)


# model evaluation

# In[26]:


#accuracy score of our model on training data
x_train_prediction=model.predict(x_train)
training_data_accuracy=accuracy_score(x_train_prediction,y_train)
print("accuracy score of our model on training data: ",training_data_accuracy)


# In[27]:


#accuracy score of our model on training data
x_test_prediction=model.predict(x_test)
test_data_accuracy=accuracy_score(x_test_prediction,y_test)
print("accuracy score of our model on training data: ",test_data_accuracy)


# building a predictive system

# In[29]:


x=(7.4,0.7,0,1.9,0.076,11,34,0.9978,3.51,0.56,9.4)
input_data=np.asarray(x) #chaning data to a numpy array
#reshaping the np array as we are predicting for one instance
input_data=input_data.reshape(1,-1)
  
pred=model.predict(input_data)
print(pred)

if(pred==1):
    print('good quality wine')
else:
    print('bad quality wine')

