# Air Transportation Fare Prediction
**Group 13:
Arianna Bucca (764361), Angela Jane Salazar Hernandez (766411), Nathan Alexander Henderson (766161)**

## Introduction
In an ever-evolving travel industry, where flight prices fluctuate frequently, it becomes crucial for both customers and airlines to have access to reliable tools that can predict flight prices accurately. This report outlines a comprehensive analysis conducted for one of the top-earning travel companies, aiming to leverage a provided dataset to predict flight prices for different airlines and routes. By using different regression algorithmns, this project seeks to empower customers with the ability to make well-informed decisions when booking flights and assist airlines in understanding the factors that influence flight prices.


## Dataset
The dataset provided by the company serves as the foundation for our analysis. It includes several variables describing flights tickets for various airlines concentrated in Asia. There is information on the name of the airline, the flight number, the price of the ticket, the city and time of departure, the city and time of arrival, the number of stops on the flight, the class of the ticket, the duration of the flight, and the number of days between the date of data collection and the date of departure. It is also necessary to note that the price of the ticket is in rupees, since the flights are all connecting Indian cities.
 
 
## Exploratory Data Analysis
After making sure the dataset did not have any null or duplicated variables, we moved on to analyzing the correlation matrix.
  
*Fig 1*: Correlation Matrix| 
:-------------------------:|
![](https://github.com/Dravitar/764361/blob/main/Flight%20corr.png)  |

As you can see from *Fig 1*, the numerical variable most related to our target variable Price is the one concerning the duration of the flight. In general, however, the values on the correlation matrix are not very strong.

We then moved on to analyzing the distribution of the target variables itself, as shown in *Fig 2* and *Fig 3*.

*Fig 2*: Price distribution - Histogram|*Fig 5*: Price distribution - Boxplot
:-------------------------:|:-------------------------:
![](https://github.com/Dravitar/764361/blob/main/Target%20distribution.png)  |  ![](https://github.com/Dravitar/764361/blob/main/Class_boxplots.png)

From this, we were able to identify two important characteristics of our data. To begin with, there is a sharp division between the price of tickets in economy class and business class. This is justified as higher classes include more benefits and require more expensive tickets. Moreover, we can clearly see from the image how the Price variable is not normally distributed. This makes the dataset imbalanced, and increases the probability of business class tickets being underepresented in our model. Therfore, we will have to take this into consideration before running our models.

We then moved on to visualizing the distribution of the other numerical variables.

*Fig 4*: Distribution of days left data|*Fig 5*: Distribution of flight duration data
:-------------------------:|:-------------------------:
![](https://github.com/Dravitar/764361/blob/main/pics/daysleft_distr.png)  |  ![](https://github.com/Dravitar/764361/blob/main/pics/duration_distr.png)

These graphs confirm what we had already seen in the correlation matrix. In particular, we see a strong resemblance between the Duration and the Price variable. This can be deriving from a variety of factors. Flights that have higher duration times are going to have to cover higher costs, that are going to show up in the form of higher prices for the customers. In fact, flights with higher duration - and hence longer routes - require more fuel for the plane. Moreover, longer flights require higher wages for the entire crew. A final aspect to consider is the need for refreshments for passengers spending more hours on the plane. All these are costs that increase the final price of the flight.

We proceeded to visualize the relationship between our target variable and our numerical variables (*Fig 6* and *Fig 7*).

*Fig 6*: Price according to the days left until the flight|*Fig 7*: Price according to the duration of the flight
:-------------------------:|:-------------------------:
![](https://github.com/Dravitar/764361/blob/main/pics/price_daysleft.png)  |  ![](https://github.com/Dravitar/764361/blob/main/pics/price_duration.png)

We can identify the negative reletionship between the price of the ticket and the number of days left before the flight. This can easily be attributed to the increase in demand for the ticket, and lack of supply for flights (we can imagine that as time passes, there is going to be less and less expty seats on the flights). Moreover, we can see that the price initually increases for flights lasting up until 25 hours. Afterwards, the price starts decreasing (although it retains a higher variance). This can be explained by the fact that flights lasting too long are not convenient and not appealing enough to customers, who are more likely to choose other modes of transport, or closer destinations.

  
## Methods
In order to meet the goals of our project, we tested and evaluated different regression algorithms. However, before doing that we took some measures to help with the imbalance in our target variable. In particular, we decided to do a stratified sampling on our data, splitting the "bins" into batches each containing 20% of the total sample. This helped us better divide the data into training and test sets, providing a more robust model. Stratified sampling allowed us to avoid underrepresenting or overrpepreseting outliers, while still maintaining the original distribution of our data, natural for the airfare market.
  We then proceeded to test different algorithms, namely Linear Regression, Decision Tree, Random Forest, KNN, and SVR. In order to achieve this, we imported the following libraries into our Jupyter notebook:
* import numpy as np
* import pandas as pd
* import matplotlib.pyplot as plt
* import seaborn as sns
* from sklearn.preprocessing import StandardScaler
* from sklearn.compose import ColumnTransformer
* from sklearn.model_selection import train_test_split
* from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
* from sklearn.linear_model import LinearRegression
* from sklearn.tree import DecisionTreeRegressor
* from sklearn.ensemble import RandomForestRegressor
* from sklearn.neighbors import KNeighborsRegressor
* from sklearn.svm import SVR
* from sklearn.model_selection import RandomizedSearchCV
* from sklearn.model_selection import GridSearchCV
  
Another aspect to note is that we decided to discard SVR throughout the testing process. This is due to the extremely long run time and lower performance metrics compared to the other algorithms tried.

After testing all the algorithms, we decided the best performing one was Random Forest (we will explain more about this decision in the following paragraph). To conclude the model training, we proceeded to tune our hyperparameters. We did this by testing both a randomized search on hyperparameters, and grid search. After testing different options, we selected the following parameters for the final model: "max_depth": 30, "min_samples_leaf": 8, "min_samples_split": 15, "n_estimators": 300.

Finally, you can observe in *Fig 8* the flowchart illustrating the path in algorithm testing.

*Fig 8*: Flowchart| 
:-------------------------:|
![](LINK)  |


## Experimental Design
After running the different algorithms, we elected Random Forest as the best performing one. This decision was supported by a variety of reasons. To begin with, we compared different evaluation metrics (*Fig 9*).

*Fig 9*: Evaluation Metrics| 
:-------------------------:|
![](https://github.com/Dravitar/764361/blob/main/pics/Evaluation%20metrics.png)  |

We considered the Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R squared. The R squared produced high results for all models, but due to the high number of variables (deriving from encoding the categorical variables) we considered the MAE and RMSE to be more reliable metrics. Regardless of the choice, it is clear from *Fig 9* that the best performing algorithm is Random Forest.
We also compared the metrics after tuning the hyperparameters. Even though the MAE of the Random Forest Regressor model with default parameters is lower, the minimum samples in leaf nodes is 1. This means the model is susceptible to noise and the max depth is much higher. This could lead to overfitting and may cause problems with new data. The RMSE and the R-squared values are slighly improved. Either way, in both cases the models are robust. Therefore, in our final choice we prioritized having a better generelazition.

Moreover, we decided to use the Linear Regression model as our baseline. This is a classic model for regression problems, but it is also very simple. 

In addition to the better evaluation metrics, we noted other advantages of Random Forest. In particular, Random Forest is an algorithm able to perform well with a large number of input variables and to identify the most important variables for prediction. Moreover, it is less prone to overfitting than Decision Trees. Finally, the training times for Random Forest are much faster than in other algorithms (namely KNN and SVR). This makes Random Forest a more efficient and beneficial choice.



## Results 
Describe the following:
i. Main finding(s): report your final results and what you might conclude from your work
ii. Include at least one placeholder figure and/or table for communicating your findings
iii. All the figures containing results should be generated from the code.


## Conclusions
This project aimed to predict flight prices using a provided dataset for the benefit of both customers and airlines. The Random Forest algorithm emerged as the best-performing model, demonstrating its effectiveness in predicting flight prices accurately. The evaluation metrics, including MAE, RMSE, and R-squared, supported this conclusion.
The project also highlighted the importance of considering the characteristics of the dataset, such as the imbalanced nature of the price variable and the relationships between different numerical variables. Exploratory data analysis revealed insights into factors influencing flight prices, such as flight duration, days left until the flight, and ticket class.

However, it is important to note that this work may not fully address all questions related to flight price prediction. Further research could explore additional factors influencing prices, such as airline reputation, seasonal variations, and external factors like economic conditions or fuel prices. Additionally, incorporating customer-specific variables, such as travel preferences and loyalty programs, could enhance the accuracy of the predictions.

For future work, it would be beneficial to investigate ensemble models or advanced techniques like gradient boosting to further improve the prediction accuracy. Additionally, integrating real-time data updates and continuous model retraining can enhance the model's performance in dynamic market conditions.

Overall, this project provides valuable insights and a strong foundation for predicting flight prices, enabling customers and airlines to make informed decisions.
