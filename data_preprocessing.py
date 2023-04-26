# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

flights_data = pd.read_csv("flight_dataset.csv")
flights_data.drop(columns=["Unnamed: 0"],inplace=True)

flights_data.isna().sum()
flights_data.duplicated().sum()

corr_df = flights_data.corr()
fig, ax = plt.subplots()
sns.heatmap(corr_df, annot=True, cmap=sns.color_palette('ch:s=.25,rot=-.25', as_cmap=True), ax=ax)
plt.show()

sns.histplot(data=flights_data['duration'])
sns.histplot(data=flights_data['days_left'])
sns.histplot(data=flights_data['price'])

categorical_cols = ['airline','source_city','departure_time','arrival_time','destination_city','class']
for col in categorical_cols:
    print(f"\n\n {col} \n\n",flights_data[f"{col}"].value_counts())

flights_data.nunique()
flights_data.drop('flight',axis=1,inplace=True)

X = flights_data.drop('price',axis=1)
y = flights_data[['price']]

stratify_cols = ['airline','departure_time','stops','class' ]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=flights_data[stratify_cols],random_state=42)

X_train_transformed = pd.get_dummies(X_train, columns = categorical_cols,drop_first=True)
X_test_transformed = pd.get_dummies(X_test, columns = categorical_cols,drop_first=True)

X_train_transformed['stops'] = X_train_transformed['stops'].replace({'zero':0,'one':1,'two_or_more':2})
X_test_transformed['stops'] = X_test_transformed['stops'].replace({'zero':0,'one':1,'two_or_more':2})

num_cols = ['duration','days_left']
ct = ColumnTransformer(
    [('scale', StandardScaler(), num_cols)], 
    remainder='passthrough')

num_scaled_train = ct.fit_transform(X_train_transformed[['duration','days_left']])
num_scaled_test = ct.transform(X_test_transformed[['duration','days_left']])

X_train_transformed.iloc[:, 1:3] = num_scaled_train
X_test_transformed.iloc[:, 1:3] = num_scaled_test

scaler_target = StandardScaler()

y_train['price'] = scaler_target.fit_transform(y_train[['price']].values)
y_test['price'] = scaler_target.transform(y_test[['price']].values)
