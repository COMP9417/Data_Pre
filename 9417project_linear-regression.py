#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


train = pd.read_csv('C:\\Users\\Neo Zhan\\Desktop\\ashrae-energy-prediction\\train.csv')
building_metadata = pd.read_csv('C:\\Users\\Neo Zhan\\Desktop\\ashrae-energy-prediction\\building_metadata.csv')


# In[3]:


train


# In[4]:


building_metadata


# In[5]:


weather_train = pd.read_csv('C:\\Users\\Neo Zhan\\Desktop\\ashrae-energy-prediction\\weather_train.csv')


# In[6]:


null_percentages_train = train.isnull().mean() * 100


# In[7]:


null_percentages_train


# In[8]:


null_percentages_building_metadata = building_metadata.isnull().mean() * 100


# In[9]:


null_percentages_building_metadata


# In[10]:


null_percentages_weather_train = weather_train.isnull().mean() * 100


# In[11]:


null_percentages_weather_train


# In[12]:


weather_train


# In[13]:


merged_df = pd.merge(weather_train, building_metadata, on='site_id', how='inner')


# In[14]:


merged_df


# In[15]:


Final_merged_df = pd.merge(merged_df, train, on=['building_id', 'timestamp'], how='inner')


# In[16]:


Final_merged_df


# In[17]:


null_percentages = Final_merged_df.isnull().mean() * 100


# In[18]:


null_percentages


# In[19]:


from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


# In[20]:


Final_merged_df.dtypes


# In[21]:


Final_merged_df['timestamp'] = pd.to_datetime(Final_merged_df['timestamp'])


# In[22]:


Final_merged_df.dtypes


# In[23]:


from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()


# In[24]:


encoder.fit(Final_merged_df['primary_use'])


# In[25]:


encoded_col = encoder.transform(Final_merged_df['primary_use'])


# In[26]:


Final_merged_df['primary_use'] = encoded_col


# In[27]:


imp = IterativeImputer(max_iter=10, random_state=0)


# In[28]:


Final_merged_df['timestamp']


# In[29]:


import numpy as np
epoch = np.datetime64('1970-01-01T00:00:00')


# In[30]:


float_array = (Final_merged_df['timestamp'] - epoch) / np.timedelta64(1, 's')


# In[31]:


Final_merged_df['timestamp']=float_array


# In[32]:


Final_merged_df.dtypes


# In[33]:


null_percentages = Final_merged_df.isnull().mean() * 100
null_percentages


# In[52]:


Final_merged_df_sample = Final_merged_df.sample(frac=0.2)


# In[53]:


Final_merged_df_sample


# In[54]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


# In[55]:


X = Final_merged_df_sample[['timestamp', 'air_temperature', 'cloud_coverage', 'dew_temperature', 'precip_depth_1_hr', 'sea_level_pressure', 'wind_direction', 'wind_speed', 'primary_use', 'square_feet','year_built', 'floor_count', 'meter']]


# In[56]:


y = Final_merged_df_sample['meter_reading']


# In[57]:


# Initialize the imputer MICE
imputer = IterativeImputer()

# Impute the missing values
X_imputed = pd.DataFrame(imputer.fit_transform(X))


# In[58]:


X_imputed


# In[59]:


X_imputed.columns = X.columns


# In[60]:


X_imputed


# In[61]:


# X.dtypes


# In[62]:


# X_imputed.dtypes


# In[63]:


X_imputed['primary_use'] = X_imputed['primary_use'].astype(int)
X_imputed['square_feet'] = X_imputed['square_feet'].astype('int64')
X_imputed['meter'] = X_imputed['meter'].astype('int64')


# In[64]:


# X_imputed.dtypes


# In[65]:


X_imputed_train, X_imputed_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)


# In[66]:


model = LinearRegression()

# Train the model on the training data
model.fit(X_imputed_train, y_train)

# Use the model to make predictions on the testing data
y_pred = model.predict(X_imputed_test)


# In[67]:


from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, y_pred)
print('Mean squared error:', mse)


# In[68]:


r2 = r2_score(y_test, y_pred)
print('R-squared:', r2)


# In[69]:


rmse = mean_squared_error(y_test, y_pred, squared=False)

print('Mean squared error:', mse)
print('R-squared:', r2)
print('Root mean squared error:', rmse)


# In[ ]:




