#!/usr/bin/env python
# coding: utf-8

# # THE SPARKS FOUNDATION            @ GRIPNOVEMBER2021

# ## TASK 1 - Prediction using Supervised ML

# ### PROBLEM STATEMENT - To predict the percentage of marks of students based on the number of study hours

# ### Author - Sparsh Lakhani

# In[1]:


# importing the required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error


# In[3]:


# reading the data
data = pd.read_csv('http://bit.ly/w-data')
data.head(5)


# In[11]:


#check if there any null value in the dataset
data.isnull == True


# #### There is no null value in the Dataset 

# In[4]:


sns.set_style('darkgrid')
sns.scatterplot(y= data['Scores'], x= data['Hours'])
plt.title('Marks Vs Study Hours',size=20)
plt.ylabel('Marks Percentage', size=12)
plt.xlabel('Hours Studied', size=12)
plt.show()


# In[5]:


sns.regplot(x= data['Hours'], y= data['Scores'])
plt.title('Regression Plot',size=20)
plt.ylabel('Marks Percentage', size=12)
plt.xlabel('Hours Studied', size=12)
plt.show()
print(data.corr())


# #### The variables are positively correlated

# ## Training the Model

# ### 1) splitting the Data

# In[7]:


# Defining X and y from the Data
X = data.iloc[:, :-1].values  
y = data.iloc[:, 1].values

# Spliting the Data in two
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)


# ### 2) Fitting the Data into model

# In[8]:


regression = LinearRegression()
regression.fit(train_X, train_y)
print(" The Model is Trained")


# ### 3) predicting the percentage of marks

# In[9]:


pred_y = regression.predict(val_X)
prediction = pd.DataFrame({'Hours': [i[0] for i in val_X], 'Predicted Marks': [k for k in pred_y]})
prediction


# In[10]:


compare_scores = pd.DataFrame({'Actual Marks': val_y, 'Predicted Marks': pred_y})
print('Actual Marks v/s Predicted Marks')
compare_scores


# ### 4)  Visually Comparing the Predicted Marks with the Actual Marks

# In[12]:


plt.scatter(x=val_X, y=val_y, color='Red')
plt.plot(val_X, pred_y, color='Black')
plt.title('Actual Marks vs Predicted Marks', size=20)
plt.ylabel('Marks Percentage', size=12)
plt.xlabel('Hours Studied', size=12)
plt.show()


# ### 5) Evaluating the model

# In[13]:


# Calculating the accuracy of the model
print('Mean absolute error: ',mean_absolute_error(val_y,pred_y))


# ## Problem statement - What will be the predicted score if a student studies for 9.25 hrs/ day?

# In[14]:


hours = [9.25]
answer = regression.predict([hours])
print("Score = {}".format(round(answer[0],3)))


# ### The student will score 93.893 marks if they study for 9.25hrs/day
