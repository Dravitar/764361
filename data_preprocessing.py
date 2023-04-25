# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

flights_data.nunique()

time_categories = {"Early_Morning":0, "Morning":1, "Afternoon":2, "Evening":3, "Night":4, "Late_Night":5}

flights_data = flights_data.replace(time_categories)

flights_data['stops'] = flights_data['stops'].replace({'zero': 0, 'one': 1, 'two_or_more': 2})

flights_sourcecity = flights_data[["source_city"]]
flights_destinationcity = flights_data[["destination_city"]]
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
Encoder = OneHotEncoder(drop='first',sparse=False)

encoding_sourcecities = Encoder.fit_transform(flights_sourcecity)
encoding_destinationcities = Encoder.transform(flights_destinationcity)

df_new = pd.concat([flights_data, pd.DataFrame(encoding_sourcecities)], axis=1)
df_new.rename(columns={
    0:"source_city1",
    1:"source_city2",
    2:"source_city3",
    3:"source_city4",
    4:"source_city5"},
    inplace=True)
df_new = pd.concat([df_new, pd.DataFrame(encoding_destinationcities)], axis=1)
df_new.rename(columns={
    0:"destination_city1",
    1:"destination_city2",
    2:"destination_city3",
    3:"destination_city4",
    4:"destination_city5"},
    inplace=True)

df_new.drop(columns=["source_city","destination_city"],inplace=True)

flights_airlines = flights_data[['airline']]
encoding_airlines = Encoder.fit_transform(flights_airlines)
df_new = pd.concat([df_new, pd.DataFrame(encoding_airlines)], axis=1)
df_new.rename(columns={
    0:"airlines1",
    1:"airlines2",
    2:"airlines3",
    3:"airlines4",
    4:"airlines5"},
    inplace=True)

df_new.drop(columns=["airline","flight"],inplace=True)
df_new = df_new.replace({'Economy':0,'Business':1})

del encoding_airlines, encoding_destinationcities, encoding_sourcecities,flights_destinationcity,flights_sourcecity,flights_airlines

scaler = MinMaxScaler(feature_range=(0,1))

