#!/usr/bin/env python
# coding: utf-8

# In[1]:


import keras


# In[2]:


import pandas as pd


# In[3]:


weather = pd.read_csv("weather-random.csv")


# In[4]:


weather.head(5)


# In[5]:


weather.drop(["(Inverters)"], axis = 1, inplace = True) 


# In[6]:


df = pd.DataFrame(weather, columns = ['Date', 'Cloud coverage', 'Visibility', 'Temperature', 'Dew point',
       'Relative humidity', 'Wind speed', 'Station pressure', 'Altimeter',
       'Solar energy']) 


# #seperating day, month and year
# df['Day'] = pd.DatetimeIndex(df['Date']).day
# df['Month'] = pd.DatetimeIndex(df['Date']).month
# df['Year'] = pd.DatetimeIndex(df['Date']).year

# In[7]:


df.shape


# In[8]:


df.columns


# In[9]:


df.head(5)


# In[10]:


print(df["Solar energy"].min())
print(df["Solar energy"].max())
print(df["Solar energy"].mean())


# In[11]:


#ensuring no null value
print(df.count(0))


# In[12]:


X= df.loc[:, df.columns != 'Solar energy']


# In[13]:


X.drop(["Date"], axis = 1, inplace = True) 


# In[14]:


y= df.iloc[:,9].values


# In[15]:


y


# In[16]:


# train and test data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[24]:


model = keras.Sequential()


# In[25]:


X.shape


# In[51]:


from keras.layers import Dense, Dropout, Activation

model.add(Dense(4, activation='tanh', input_dim=8))
#model.add(Dropout(0.5))
model.add(Dense(4, activation='tanh'))
#model.add(Dropout(0.5))
model.add(Dense(1, activation='softmax'))


# In[52]:


# Compile model
model.compile(loss='mean_squared_error', optimizer='adam')


# In[53]:


X_train.shape


# In[54]:


model.fit(X_train, y_train, batch_size=100, epochs=150, verbose=2, callbacks=None, validation_split=0.5, validation_data=(X_test,y_test), shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0, steps_per_epoch=None, validation_steps=None)


# In[59]:


from sklearn.externals import joblib


# In[60]:


joblib.dump(model, 'model.pkl')


# In[61]:


df.shape


# In[62]:


# Saving the data columns from training
model_columns = list(df.columns)
joblib.dump(model_columns, 'model_columns.pkl')
print("Models columns dumped!")


# In[64]:


X.columns


# In[ ]:




