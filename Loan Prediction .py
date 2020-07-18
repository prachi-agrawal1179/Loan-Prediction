#!/usr/bin/env python
# coding: utf-8

# In[53]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[54]:


traindf = pd.read_csv(r"C:\Users\mypc\Downloads\train.csv")
testdf = pd.read_csv(r"C:\Users\mypc\Downloads\test.csv")


# In[55]:


traindf.isnull().sum()
testdf.isnull().sum()


# In[56]:


traindf.info()


# In[57]:


traindf.fillna(traindf.mean(),inplace=True) 
traindf.isnull().sum() 


# In[58]:


testdf.fillna(testdf.mean(),inplace=True) 
testdf.isnull().sum()


# In[59]:


traindf.Gender.fillna(traindf.Gender.mode()[0],inplace=True)
traindf.Married.fillna(traindf.Married.mode()[0],inplace=True)
traindf.Dependents.fillna(traindf.Dependents.mode()[0],inplace=True) 
traindf.Self_Employed.fillna(traindf.Self_Employed.mode()[0],inplace=True)  
traindf.isnull().sum() 


# In[60]:


testdf.Gender.fillna(testdf.Gender.mode()[0],inplace=True)
testdf.Dependents.fillna(testdf.Dependents.mode()[0],inplace=True) 
testdf.Self_Employed.fillna(testdf.Self_Employed.mode()[0],inplace=True)  
testdf.isnull().sum()


# In[61]:


traindf.Loan_Amount_Term=np.log(traindf.Loan_Amount_Term)
finaldf=testdf.drop(['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History', 'Property_Area'],axis=1)
traindf=traindf.drop('Loan_ID',axis=1)
testdf=testdf.drop('Loan_ID',axis=1)


# In[62]:


X=traindf.drop('Loan_Status',1)
y=traindf.Loan_Status


# In[63]:


X=pd.get_dummies(X)
traindf=pd.get_dummies(traindf)
testdf=pd.get_dummies(testdf)


# In[64]:


from sklearn.model_selection import train_test_split
x_train,x_cv,y_train,y_cv = train_test_split(X,y,test_size=0.2)


# In[65]:


from sklearn import model_selection
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split 
from sklearn import metrics


# In[66]:


clf = DecisionTreeClassifier()
clf = clf.fit(x_train,y_train)
y_pred = clf.predict(x_cv)


# In[67]:


print("Accuracy:",metrics.accuracy_score(y_cv, y_pred))


# In[68]:


model=LogisticRegression()
model.fit(x_train,y_train)


# In[69]:


pred_cv=model.predict(x_cv)


# In[70]:


print("Accuracy:",metrics.accuracy_score(y_cv, pred_cv))


# In[71]:


from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
accuracy_score(y_cv,pred_cv)
matrix=confusion_matrix(y_cv,pred_cv)
print(matrix)


# In[72]:


from sklearn.naive_bayes import GaussianNB 
nb=GaussianNB()
nb.fit(x_train,y_train)


# In[73]:


pred_cv4=nb.predict(x_cv)


# In[74]:


print("Accuracy:",metrics.accuracy_score(y_cv, pred_cv4))


# In[75]:


pred_test=nb.predict(testdf)


# In[85]:



finaldf['Loan_Status']=pred_test
finaldf.head()


# In[86]:


testdf.head()


# In[87]:


finaldf.to_csv('result.csv', index=False)


# In[88]:


finaldf.head()

