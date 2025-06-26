#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.ensemble import GradientBoostingRegressor


from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import Ridge

from sklearn.feature_selection import RFE


from sklearn.metrics import mean_squared_error 
from sklearn.metrics import mean_absolute_error 


# Databse
dataset = pd.read_csv("PLA_LOI.csv")
#dataset = pd.read_csv("PLA_PROPERTIE_Tensile.csv")
#dataset = pd.read_csv("PLA_PHRR.csv")
#dataset = pd.read_csv("PLA_Tg.csv")

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

dataset_generated_structures = pd.read_csv("EDP.csv")
X_generated_structures = dataset_generated_structures

# split data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=18)
# tensile, X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=14)
# PHRR, X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=17)
# Tg, X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=16)

# Data processing
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
X_generated_structures_scaler = scaler.transform(X_generated_structures)

# Feature Selection
estimator = Ridge(alpha=1, random_state=1, max_iter=100000)
# tensile, estimator = Ridge(alpha=2, random_state=1, max_iter=100000)
# PHRR, estimator = Ridge(alpha=1, random_state=1, max_iter=100000)
# Tg, estimator = Ridge(alpha=1, random_state=1, max_iter=100000)
selector = RFE(estimator, n_features_to_select = 282)
# tensile, selector = RFE(estimator, n_features_to_select =137)
# PHRR, selector = RFE(estimator, n_features_to_select =95)
# selector = RFE(estimator, n_features_to_select =80)


selector.fit(X_train, y_train)
X_train_selector = selector.transform(X_train)
X_test_selector =selector.transform(X_test) 
X_generated_structures_selector = selector.transform(X_generated_structures_scaler)
                 
# Output Selected Feature
feature_selected = selector.get_support()
logf = open("logfile.log", "a+")
np.set_printoptions(threshold=np.inf)
print(f"{feature_selected}", file=logf, flush=True)

# Regression
regressor = GradientBoostingRegressor(n_estimators=520, max_depth=3, random_state=42)
# tensile, regressor = GradientBoostingRegressor(n_estimators=270, max_depth=3, random_state=42)
# PHRR, regressor = GradientBoostingRegressor(n_estimators=230, max_depth=3, random_state=42)
# Tg, regressor = GradientBoostingRegressor(n_estimators=150, max_depth=3, random_state=42)


 
regressor.fit(X_train_selector, y_train)

# Output feature importance
feature_importances_ = regressor.feature_importances_

# Model Performance

y_train_predict = regressor.predict(X_train_selector)
y_predict = regressor.predict(X_test_selector)
mean_squared_error = mean_squared_error(y_test, y_predict)
root_mean_squard_error = mean_squared_error**0.5
mean_absolute_error = mean_absolute_error(y_test, y_predict)

                                                                        
print(f"train R2: {regressor.score(X_train_selector, y_train):.3f}")
print(f"test R2: {regressor.score(X_test_selector, y_test):.3f}")

# Predict PLA/EDP
y_generated_structures_predict = regressor.predict(X_generated_structures_selector)

