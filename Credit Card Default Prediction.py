#!/usr/bin/env python
# coding: utf-8

# In[2]:


#importing libraries which will be used
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', "inline #sets the backend of matplotlib to the 'inline' backend")


# In[3]:


#now we will read data from its directory
dataset=pd.read_csv('D:/Datasets/credit_card.csv')


# In[4]:


#head is use to show fist 5 rows of the data
dataset.head()


# In[5]:


#The info() method prints information about the DataFrame
dataset.info()


# In[6]:


#Scale features using statistics that are robust to outliers.
from sklearn.preprocessing import RobustScaler


# In[7]:


#asingnment of features to x and target to y 
x=dataset.iloc[:,:-1]
rb=RobustScaler()
x=rb.fit_transform(x)
y=dataset.iloc[:,-1]


# In[8]:


#features after scalling
x


# In[9]:


#target
y


# In[10]:


#visualization of data with heat map
#annot :bool or rectangular dataset, optional
#If True, write the data value in each cell. If an array-like with the same shape as data, then use this to annotate the heatmap instead of the data. Note that DataFrames will match on position, not index.
#cmap :matplotlib colormap name or object, or list of colors, optional
#The mapping from data values to color space. If not provided, the default will depend on whether center is set.#vmin, vmaxfloats, optional
#Values to anchor the colormap, otherwise they are inferred from the data and other keyword arguments.
corr=dataset.corr()
plt.figure(figsize=(18,15))
sns.heatmap(corr,annot=True,vmin=-1.0,cmap='mako')
plt.title("correlation heatmap")
plt.show()


# In[11]:


#importing train test split from sklearn
#and spliting taget and features into train and test set
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=123)


# In[12]:



def CMatrix(CM,labels=['pay','deafault']):
    df=pd.DataFrame(data=CM,index=labels,columns=labels)
    df.index.name='TRUE'
    df.columns.name='Prediction'
    df.loc['Total']=df.sum()
    df['Total']=df.sum(axis=1)
    return df


# In[13]:


#preparing a dataframe for model analysis
metrics=pd.DataFrame(index=['accuracy','precision','recall'],columns=['NULL','LogisticReg','ClassTree','Naivebaiyes'])

accuracy: the proportion of the total number of prediction that are correct
precision: the proportion of positive prediction that are actually correct
recall: the proportion of positive observed values correctly predicted as such 
    
in the application

accuracy: overall how often the model predicts correctly defaulters and non-defaulters
precision: when the model predicts default:how often is correct?
recall: the proportion of actual defaulters that the model will correctly predict as such 
    features discription:

ID: ID of each client
LIMIT_BAL: Amount of given credit in NT dollars (includes individual and family/supplementary credit
SEX: Gender (1=male, 2=female)
EDUCATION: (1=graduate school, 2=university, 3=high school, 4=others, 5=unknown, 6=unknown)
MARRIAGE: Marital status (1=married, 2=single, 3=others)
AGE: Age in years
PAY_0: Repayment status in September, 2005 (-1=pay duly, 1=payment delay for one month, 2=payment delay for two months, … 8=payment delay for eight months, 9=payment delay for nine months and above)
PAY_2: Repayment status in August, 2005 (scale same as above)
PAY_3: Repayment status in July, 2005 (scale same as above)
PAY_4: Repayment status in June, 2005 (scale same as above)
PAY_5: Repayment status in May, 2005 (scale same as above)
PAY_6: Repayment status in April, 2005 (scale same as above)
BILL_AMT1: Amount of bill statement in September, 2005 (NT dollar)
BILL_AMT2: Amount of bill statement in August, 2005 (NT dollar)
BILL_AMT3: Amount of bill statement in July, 2005 (NT dollar)
BILL_AMT4: Amount of bill statement in June, 2005 (NT dollar)
BILL_AMT5: Amount of bill statement in May, 2005 (NT dollar)
BILL_AMT6: Amount of bill statement in April, 2005 (NT dollar)
PAY_AMT1: Amount of previous payment in September, 2005 (NT dollar)
PAY_AMT2: Amount of previous payment in August, 2005 (NT dollar)
PAY_AMT3: Amount of previous payment in July, 2005 (NT dollar)
PAY_AMT4: Amount of previous payment in June, 2005 (NT dollar)
PAY_AMT5: Amount of previous payment in May, 2005 (NT dollar)
PAY_AMT6: Amount of previous payment in April, 2005 (NT dollar)
default.payment.next.month: Default payment (1=yes, 0=no)
# In[14]:


from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,precision_recall_curve,recall_score


# In[15]:


#the null model: always predict the most common category
y_pred_test=np.repeat(y_train.value_counts().idxmax(),y_test.size)
metrics.loc['accuracy','NULL']=accuracy_score(y_pred=y_pred_test,y_true=y_test)
metrics.loc['precision','NULL']=precision_score(y_pred=y_pred_test,y_true=y_test)
metrics.loc['recall','NULL']=recall_score(y_pred=y_pred_test,y_true=y_test)

CM=confusion_matrix(y_pred=y_pred_test,y_true=y_test)
CMatrix(CM)


# In[20]:


#import the logistic model from sklearn/estimator object
from sklearn.linear_model import LogisticRegression
#create an instance of the estimator
LR=LogisticRegression(n_jobs=-1,random_state=15)
#use the training data to train the estimator
LR.fit(x_train,y_train)
#evaluate the model
y_pred_test=LR.predict(x_test)
metrics.loc['accuracy','LogisticReg']=accuracy_score(y_pred=y_pred_test,y_true=y_test)
metrics.loc['precision','LogisticReg']=precision_score(y_pred=y_pred_test,y_true=y_test)
metrics.loc['recall','LogisticReg']=recall_score(y_pred=y_pred_test,y_true=y_test)
#confusion matrix
CM=confusion_matrix(y_pred=y_pred_test,y_true=y_test)
CMatrix(CM)


# In[21]:


#import the Decisiontreeclassifier from sklearn/estimator object
from sklearn.tree import DecisionTreeClassifier
#create an instance of the estimator
class_tree=DecisionTreeClassifier(min_samples_split=30,min_samples_leaf=10,random_state=10)
#use the training data to train the estimator
class_tree.fit(x_train,y_train)
#evaluate the model
y_pred_test=class_tree.predict(x_test)
metrics.loc['accuracy','ClassTree']=accuracy_score(y_pred=y_pred_test,y_true=y_test)
metrics.loc['precision','ClassTree']=precision_score(y_pred=y_pred_test,y_true=y_test)
metrics.loc['recall','ClassTree']=recall_score(y_pred=y_pred_test,y_true=y_test)
#confusion matrix
CM=confusion_matrix(y_pred=y_pred_test,y_true=y_test)
CMatrix(CM)


# In[22]:


#import gaussionNB fron sklearn.naive_bayes/estimator object
from sklearn.naive_bayes import GaussianNB
#create an instance of the estimator
NBC=GaussianNB()
#use the training data to train the estimator
NBC.fit(x_train,y_train)
#evaluate the model
y_pred_test=NBC.predict(x_test)
metrics.loc['accuracy','Naivebaiyes']=accuracy_score(y_pred=y_pred_test,y_true=y_test)
metrics.loc['precision','Naivebaiyes']=precision_score(y_pred=y_pred_test,y_true=y_test)
metrics.loc['recall','Naivebaiyes']=recall_score(y_pred=y_pred_test,y_true=y_test)
#confusion matrix
CM=confusion_matrix(y_pred=y_pred_test,y_true=y_test)
CMatrix(CM)


# In[23]:


#no lets see in percentage
100*metrics


# In[24]:


fig,ax=plt.subplots(figsize=(8,5))
metrics.plot(kind='barh',ax=ax)
ax.grid();

