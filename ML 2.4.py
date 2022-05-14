#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd


# In[11]:


house_data = pd.read_csv('https://raw.githubusercontent.com/zekelabs/data-science-complete-tutorial/master/Data/house_rental_data.csv.txt',index_col='Unnamed: 0')
house_data.head()


# In[12]:


house_data.rename(columns={'Living.Room':'Livingroom'}, inplace=True)
house_data.head()


# In[20]:


columns = house_data.columns.tolist()
columns


# In[21]:



columns.remove('Price')
columns


# In[22]:



feature_data = house_data[columns]
feature_data


# In[23]:



target_data = house_data.Price
target_data


# In[17]:



from sklearn.model_selection import train_test_split
trainX,testX, trainY,testY = train_test_split(feature_data, target_data)


# In[18]:



trainX.shape


# In[19]:


testX.shape


# In[25]:


from sklearn.linear_model import LinearRegression
lr = LinearRegression()


# In[26]:


lr = LinearRegression(normalize=True)
lr.fit(trainX,trainY)


# In[27]:


lr.coef_


# In[28]:


testX[:5]


# In[29]:


testY[:5]


# In[30]:


lr.predict(testX[:5])


# In[31]:


from sklearn.metrics import mean_squared_error, mean_absolute_error


# In[32]:


pred = lr.predict(testX)
mean_absolute_error(y_pred=pred, y_true=testY)


# In[33]:


from sklearn.linear_model import Ridge,Lasso


# In[34]:


ridge = Ridge(alpha=1000)
lasso = Lasso(alpha=1000)


# In[35]:


ridge.fit(trainX,trainY)
lasso.fit(trainX,trainY)


# In[36]:


pred = ridge.predict(testX)
pred = lasso.predict(testX)


# In[37]:


mean_absolute_error(y_pred=pred, y_true=testY)


# In[ ]:




