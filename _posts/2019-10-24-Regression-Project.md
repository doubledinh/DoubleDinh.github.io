---
layout: post
title: Regression Project
---
#### Multivariable Regression Model to Predict Prices of Rideshare Companies

My project revolved around predicting the prices of rideshare companies, particularly Lyft and Uber. Current both Lyft and Uber offer several options regarding which type of vehicle/ride the user wants to drive in. As a result, my data largely depended on the categorical data that was given. However, there were also quantitative values such as the epoch time the ride was taken as well as the distance in miles that the user drove. What proved to be difficult during this project was the amount of data that was provided. Before cleaning and cropping the data, there were 693071 rows which lengthened the time of the SNS pear plot as well as the polynomial fit algorithm. Consequently, I resorted to scrambling the original data set and strictly selecting the first 20,000 data points. From there, I removed the rows with missing values, which brought down the total number of rows to just 18416 (something my puny computer could handle). I then one-hotted the categorical data which added 22 extra columns due to the amount of types of rides the user could select. To determine the correlation between the quantitative explanatory values and my responses variable (price), I ran a .coor. However, because my dataset only had 3 quantitative variables, I chose to keep all of them in my regression model. What was interesting though was how low the timestamp correlated with price. I soon discovered after running an SNS plot that the timestamp data was bi-model, a clear reason why it produced such a poor correlation coefficient. I then ran a polynomial fit on the quantitative data and again, the mean squared error was significantly large, most likely due to the bi-model distribution curves. The result of the poly fit demonstrated that the best fit curve was linear so that saved me the trouble of transforming the data. To make the model, I pipelined a polyfit of 1 (I know that this is unnecessary) and standard scaled my quantitative data. Out of curiosity, I ran a normal regression which resulted in an R squared value of 0.926. Then, I pipelined a ridge regression model, testing 4 different alphas: 0.0001, 0.1, 1, 10. The resulting R squared from best alpha, 0.001 was 0.927, slightly higher than the regular regression. The adjusted R squared turned out to be 0.9268, a number very similar to the R squared value since no new variable was added. Overall, it was rather difficult for a model to be made from my dataset simply because most of the explanatory variables were bi-model. Because of that, the polynomial fit turned out the be 1, not allowing me to introduce over fitting into the data. I do not believe that my model was biased, because most of the explanatory variables has some correlation with the response. I also do not believe that my model had high variance because I was unable to overfit the data with a polynomial fit. Ultimately, if I were to do something differently, I would split the data based on the time the ride was ordered. As a result, I think the model would fit better since there would be no bi-model distribution. 


Below, I inserted the way I trimmed the dataset
```javascript
#Splitting df and strictly selecting 20000 rows
df = df.iloc[0: 20000]
df = df.dropna()
df = df.reset_index()
``` 

Below, I included the coefficients of my model:
-8.21148765e-02 -3.28881410e-01  1.75447401e-01 -2.77632290e-01
  4.45172150e-01  1.08060848e-01  3.33448560e-02  3.13941517e-02
  3.98346812e-02 -1.70080141e-01  1.13472264e-01 -8.80176306e-02
  4.58556963e+00  1.44274421e+01  2.81797572e-01  5.77794982e+00
  1.50271100e+01 -7.88912639e+00 -2.10718633e+00 -1.06561204e+01
 -7.02340736e+00 -6.05354767e+00 -3.01134429e-01 -6.06934658e+00
  4.34424309e-01 -4.34424309e-01  0.00000000e+00  3.19080954e+00
 -4.19487897e-03  1.88847413e+00

You can find my code on my [repository](https://github.com/doubledinh/Regression_Project)