import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# uploading the data
flights_data = pd.read_csv("flight_dataset.csv")
flights_data.drop(columns=["Unnamed: 0"],inplace=True)

# checking for null and duplicated values
flights_data.isna().sum()
flights_data.duplicated().sum()

# creating the correlation matrix
corr_df = flights_data.corr()
fig, ax = plt.subplots()
sns.heatmap(corr_df, annot=True, cmap=sns.color_palette('ch:s=.25,rot=-.25', as_cmap=True), ax=ax)
plt.savefig("Flight corr.png", dpi=300, bbox_inches='tight')
plt.show()

# now let's consider the distribution of our target variable
flights_data['price']=flights_data['price'].astype(float)

fig, ax = plt.subplots()
sns.histplot(data=flights_data, x="price", palette="Set1", ax=ax)
plt.xlabel('Price')
fig.tight_layout()
plt.savefig("Target distribution.png", dpi=300, bbox_inches='tight')
fig.show()

    # we can calculate the imbalance:
low_p = flights_data['price'].loc[flights_data['price'] < 30000]
high_p = flights_data['price'].loc[flights_data['price'] >= 30000]
c_ratio = high_p.count()/low_p.count()
print("Minority class ratio:", c_ratio)

sns.histplot(data=flights_data['duration'])
sns.histplot(data=flights_data['days_left'])
sns.histplot(data=flights_data['price'])


# now let's take a look at the categorical values to know what is the best way to encode them
categorical_cols = ['airline','source_city','departure_time','arrival_time','destination_city','class','stops']
for col in categorical_cols:
    print(f"\n\n{col} \n\n",flights_data[f"{col}"].value_counts())

flights_data.nunique()

#the flight number column is not relevant for our analysis, so we drop it
flights_data.drop('flight',axis=1,inplace=True)


target = 'price'
features = ['airline', 'source_city', 'departure_time', 'stops', 'arrival_time',
       'destination_city', 'class', 'duration', 'days_left','bins']

# Calculate the cut values for the stratification bins
percentiles = [25, 50, 75]
cut_values = flights_data[target].quantile([p/100 for p in percentiles])

# Create a new column with the bin labels
flights_data['bins'] = pd.cut(flights_data[target], bins=[-float("inf")] + list(cut_values) + [float("inf")])

# Split the data into training and testing sets, stratified by the bins
X_train, X_test, y_train, y_test = train_test_split(flights_data[features], flights_data[target], test_size=0.2, stratify=flights_data['bins'], random_state=42)

y_test = pd.DataFrame(y_test)
y_train = pd.DataFrame(y_train)

# Drop the bin column
X_train.drop('bins',axis=1,inplace=True)
X_test.drop('bins',axis=1,inplace=True)



#we transform the categorical variables into dummy variables
X_train_transformed = pd.get_dummies(X_train, columns = categorical_cols[0:6],drop_first=True)
X_test_transformed = pd.get_dummies(X_test, columns = categorical_cols[0:6],drop_first=True)

X_train_transformed['stops'] = X_train_transformed['stops'].replace({'zero':0,'one':1,'two_or_more':2})
X_test_transformed['stops'] = X_test_transformed['stops'].replace({'zero':0,'one':1,'two_or_more':2})

#creating the box plots for the categorical variables
df=X_train_transformed.drop(columns=(['duration','days_left']))
sns.boxplot(data=df)

for col in categorical_cols:
    counts = pd.DataFrame(flights_data[col].value_counts())
    counts.reset_index(inplace=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax = sns.barplot(x=counts["index"], y=counts[col], data=counts, palette="bright",)
    # set the title and axis labels for the plot
    plt.xlabel(col)
    plt.ylabel('Frequency')
    # show the plot
    plt.show()
    ax.figure.savefig(f"{col}.png",dpi=300)



#then we scale the numerical variables
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


###############################################################################
# here we test different models

### LINEAR REGRESSION ###
from sklearn.linear_model import LinearRegression

# Set up a regression model
lin_reg = LinearRegression()

# Fit the model on the training data
lin_reg.fit(X_train_transformed, y_train)

# Use the trained model to make predictions on the test set
y_pred = lin_reg.predict(X_test_transformed)

# Compute the mean absolute error and R-squared on the test set
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean absolute error:", mae)
print("Mean squared root error:", np.sqrt(mse))
print("R-squared:", r2)


### DECISION TREE ###
from sklearn.tree import DecisionTreeRegressor

# Set up a regression model
tree_reg = DecisionTreeRegressor()

# Fit the model on the training data
tree_reg.fit(X_train_transformed, y_train)

# Use the trained model to make predictions on the test set
y_pred = tree_reg.predict(X_test_transformed)

# Compute the mean absolute error and R-squared on the test set
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean absolute error:", mae)
print("Mean squared root error:", np.sqrt(mse))
print("R-squared:", r2)


### RANDOM FOREST ###
from sklearn.ensemble import RandomForestRegressor

# Create a random forest regressor
rf = RandomForestRegressor()

# Fit the model on the training data
rf.fit(X_train_transformed, y_train)

# Predict the target values for the test data
y_pred = rf.predict(X_test_transformed)

# Compute the mean absolute error and R-squared on the test set
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean absolute error:", mae)
print("Mean squared root error:", np.sqrt(mse))
print("R-squared:", r2)


'''
### Lasso ###
from sklearn.linear_model import Lasso
# Set up a regression model
lasso = Lasso(alpha=33)
# Fit the model on the training data
lasso.fit(X_train_transformed, y_train)
# Use the trained model to make predictions on the test set
y_pred = lasso.predict(X_test_transformed)
# Compute the mean absolute error and R-squared on the test set
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Mean absolute error:", mae)
print("Mean squared root error:", np.sqrt(mse))
print("R-squared:", r2)
'''

### KNN ###
from sklearn.neighbors import KNeighborsRegressor

knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(X_train_transformed, y_train)

y_pred = knn.predict(X_test_transformed)

# Compute the mean absolute error and R-squared on the test set
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean absolute error:", mae)
print("Mean squared root error:", np.sqrt(mse))
print("R-squared:", r2)


### SVR ###
from sklearn.svm import SVR

y_train_numpy = y_train.to_numpy()
y_train_numpy = np.ravel(y_train_numpy)

# Create a Support Vector Regression model and fit it on the training data
svr = SVR(kernel='rbf', C=1, gamma=0.1)
svr.fit(X_train_transformed, y_train_numpy)

y_pred = svr.predict(X_test_transformed)

# Compute the mean absolute error and R-squared on the test set
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean absolute error:", mae)
print("Mean squared root error:", np.sqrt(mse))
print("R-squared:", r2)

from sklearn.model_selection import GridSearchCV

# Define the parameter grid
param_grid = {'C': [0.1, 1, 10, 100],
              'epsilon': [0.01, 0.1, 1, 10],
              'gamma': [0.001, 0.01, 0.1, 1, 'scale', 'auto']}

# Define the SVR model
svr = SVR(kernel = 'rbf')

# Define the grid search object
grid_search = GridSearchCV(svr, param_grid, cv=5, scoring='neg_mean_squared_error')

# Fit the grid search object to the data
grid_search.fit(X_train_transformed, y_train_numpy)

# Print the best parameters and score
print("Best parameters: {}".format(grid_search.best_params_))
print("Best cross-validation score: {:.2f}".format(np.sqrt(-grid_search.best_score_)))