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


  6. relationship between price and: days left , duration
  8. range of price values for each of the class labels (box plots)
  9. other box plots maybe???? if we can find something good to comment for them

  
# NEXT TO WRITE: 
## Methods
Describe your proposed ideas (e.g., features, algorithm(s), training overview, design choices, etc.) and your environment so that:
i. A reader can understand why you made your design decisions and the reasons behind any other choice related to the project
ii. A reader should be able to recreate your environment (e.g., conda list, conda env export, etc.)
iii. It may help to include a figure illustrating your ideas, e.g., a flowchart illustrating the steps in your machine learning system(s)

1. stratification
2. benefits of random forest

## Experimental Design
Describe any experiments conducted to validate the target contribution(s) of the project. Indicate the main purpose of each experiment, in particular:
i. The main purpose: 1-2 sentence high-level explanation
ii. Baseline(s): describe the method(s) that you used to compare your work to
iii. Evaluation Metrics(s): which ones did you use and why?

## Results 
Describe the following:
i. Main finding(s): report your final results and what you might conclude from your work
ii. Include at least one placeholder figure and/or table for communicating your findings
iii. All the figures containing results should be generated from the code.

## Conclusions
List some concluding remarks. In particular:
i. Summarize in one paragraph the take-away point from your work.
Include one paragraph to explain what questions may not be fully answered by your work as well as natural next steps for this direction of future work
