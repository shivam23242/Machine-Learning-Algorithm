#!/usr/bin/env python
# coding: utf-8

# In[2]:


#importing the libraries we will use
#numpy and pandas are use tu creating array and dataframe and do many operations on dataset
import numpy as np
import pandas as pd
#regular expression is used to searching word in a paragraph
import re
#stopwords will remove all the words which doesnt add much meaning to sentence
from nltk.corpus import stopwords
#porter stemmer will provide root word for a given word
from nltk.stem.porter import PorterStemmer
#tfidfvectorizer will convert text to feature vector
from sklearn.feature_extraction.text import TfidfVectorizer
#train and test split use to spli our data into training and test set
from sklearn.model_selection import train_test_split
#LogisticRegression is used to buid our model
from sklearn.linear_model import LogisticRegression
#accuracy_score is used to find the accuracy of our model
from sklearn.metrics import accuracy_score


# In[3]:


#now lets see what are stopwords in english
print(stopwords.words('english'))


# data colection and preprocessing

# In[5]:


#loading our dataset with the help of pandas dataframes
dataset=pd.read_csv('E:/Ml_projects/Fake news prediction/train.csv/train.csv')


# In[6]:


#now check no. of rows and coulums in dataset
dataset.shape


# In[7]:


#now print first five row of the dataset
dataset.head()


# In[8]:


#now check for missing values in dataset
dataset.isnull().sum()


# In[9]:


#as we see some values are missing in title , author and text columns
#these are very small as compare to size of our dataset so we can replace them by null strings
dataset=dataset.fillna('')


# In[10]:


#lets check missing value after filling null valuesb
dataset.isnull().sum()


# In[30]:


#merging the author name and news title
dataset['content']=dataset['author']+' '+dataset['title']
print(dataset['content'])


# In[31]:


#now seprating the data into features and target
x=dataset.drop(columns='label',axis=1)
y= dataset['label']


# In[32]:


x


# In[33]:


y


# In[34]:


#now we will do stemming that is process of reducing a word into root word
#ex: actor,actress,acting------>> act 
port_stem=PorterStemmer()


# In[50]:


def stemming(content):
    stm_content=re.sub('[^a-zA-Z]',' ',content)
    stm_content=stm_content.lower()
    stm_content=stm_content.split()
    stm_content=[port_stem.stem(word) for word in stm_content if not word in stopwords.words('english')]
    stm_content=' '.join(stm_content)
    return stm_content


# In[53]:


dataset['content']=dataset['content'].apply(stemming)


# In[54]:


print(dataset['content'])


# In[62]:


X=dataset['content'].values
Y=dataset['label'].values


# In[63]:


X


# In[64]:


Y


# In[68]:


#now converting text data into numeric data
vectorizer=TfidfVectorizer()
vectorizer.fit(X) #here x should be capital

X=vectorizer.transform(X) 


# In[69]:


print(X)


# In[70]:


#spliting our data into training and testing
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.25,random_state=2)


# In[71]:


#now training our model using logistic regression
model=LogisticRegression()
model.fit(X_train,Y_train)


# In[73]:


#accuracy score of our model on training data
X_train_prediction=model.predict(X_train)
training_data_accuracy=accuracy_score(X_train_prediction,Y_train)
print("accuracy score of our model on training data: ",training_data_accuracy)


# In[75]:


#accuracy score of our model on training data
X_test_prediction=model.predict(X_test)
test_data_accuracy=accuracy_score(X_test_prediction,Y_test)
print("accuracy score of our model on training data: ",test_data_accuracy)


# In[80]:


X_new=X_test[7] #7th testing input
pred=model.predict(X_new)
print(pred)

if(pred[0]==0):
    print('news is real')
else:
    print('news is fake')

