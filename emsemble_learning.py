"""# MINI Dataset"""

import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.model_selection import train_test_split

building_meta = pd.read_csv('/content/drive/MyDrive/Energy Predictor/building_metadata.csv')
primary_use_categories = building_meta['primary_use'].unique()
p_u_code = []
for c in enumerate(primary_use_categories):
  p_u_code.append([c[0],c[1]])
p_u_df = pd.DataFrame(p_u_code, columns=['primary_use_code', 'primary_use'])

def convert_timestamp(df):
  df['year'] = pd.to_datetime(df['timestamp']).dt.year
  df['month'] = pd.to_datetime(df['timestamp']).dt.month
  df['day'] = pd.to_datetime(df['timestamp']).dt.day
  df['hour'] = pd.to_datetime(df['timestamp']).dt.hour

  return df

full_data = pd.read_csv('/content/drive/MyDrive/Energy Predictor/mini_train.csv')
full_data = convert_timestamp(full_data)
full_data = pd.merge(full_data, p_u_df, on='primary_use', how='left')
full_data = full_data.drop(['building_id','timestamp','site_id','primary_use'], axis=1)
Final_merged_df = full_data.drop(['floor_count','year_built','cloud_coverage'], axis=1)
#train = train.drop(['year','day', 'hour', 'precip_depth_1_hr', 'wind_direction'], axis=1)

Final_merged_df.info()

X = Final_merged_df.drop(['meter_reading'], axis=1)
y = Final_merged_df['meter_reading']

# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler()
# X[X.columns] = scaler.fit_transform(X[X.columns])

# Initialize the imputer MICE
imputer = IterativeImputer()

# Impute the missing values
X_imputed = pd.DataFrame(imputer.fit_transform(X))
X_imputed.columns = X.columns

X_imputed['primary_use_code'] = X_imputed['primary_use_code'].astype(int)
X_imputed['square_feet'] = X_imputed['square_feet'].astype('int64')
X_imputed['meter'] = X_imputed['meter'].astype('int64')

X_imputed_train, X_imputed_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

"""# RamdonForest"""

# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_squared_error

# regressor = RandomForestRegressor(max_depth=2, random_state=0)
# regressor.fit(X_imputed_train, y_train)
# y_pred = regressor.predict(X_imputed_test)

# # RMSE
# rmse = mean_squared_error(y_test, y_pred, squared=False)
# print(rmse)

"""# Bagging of Decision Tree"""

from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR

bag_reg = BaggingRegressor(
    DecisionTreeRegressor(max_depth=10), n_estimators=6
)
bag_reg.fit(X_imputed_train, y_train)
y_pred = bag_reg.predict(X_imputed_test)

# RMSE
rmse = mean_squared_error(y_test, y_pred, squared=False)
print(rmse)

"""# Gradirnt Boosting"""

# from sklearn.ensemble import GradientBoostingRegressor
# from sklearn.metrics import mean_squared_error

# gb_reg = GradientBoostingRegressor(random_state=0)
# gb_reg.fit(X_imputed_train, y_train)
# y_pred = gb_reg.predict(X_imputed_test)

# # RMSE
# rmse = mean_squared_error(y_test, y_pred, squared=False)
# print(rmse)