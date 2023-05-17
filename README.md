# Air Transportation Fare Prediction
Group 13:
Arianna Bucca (764361), Angela Jane Salazar Hernandez (766411), Nathan Alexander Henderson (766161)

## Introduction
  In an ever-evolving travel industry, where flight prices fluctuate frequently, it becomes crucial for both customers and airlines to have access to reliable tools that can predict flight prices accurately. This report outlines a comprehensive analysis conducted for one of the top-earning travel companies, aiming to leverage a provided dataset to predict flight prices for different airlines and routes. By using different regression algorithmns, this project seeks to empower customers with the ability to make well-informed decisions when booking flights and assist airlines in understanding the factors that influence flight prices.

## Dataset
  The dataset provided by the company serves as the foundation for our analysis. It includes several variables describing flights tickets for various airlines concentrated in Asia. There is information on the name of the airline, the flight number, the price of the ticket, the city and time of departure, the city and time of arrival, the number of stops on the flight, the class of the ticket, the duration of the flight, and the number of days between the date of data collection and the date of departure. It is also necessary to note that the price of the ticket is in rupees, since the flights are all connecting Indian cities.
  
## Exploratory Data Analysis
  After making sure the dataset did not have any null or duplicated variables, we moved on to analyzing the correlation matrix.
  ![Alt text](https://github.com/Dravitar/764361/blob/main/Flight%20corr.png)


  2. numerical variables distribution
  4. relationship between price and: days left , duration
  5. target variable distribution
  6. range of price values for each of the class labels (box plots)
  7. other box plots maybe???? if we can find something good to comment for them

  
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
