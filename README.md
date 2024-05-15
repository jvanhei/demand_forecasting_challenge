Food Demand Forecasting solution 

Forecasting demand is important in many practial applications including food, retail, energy and finance. This repository presents a solution to the problem of predicting how many food items (num_orders) will be ordered from different center (center_id) locations serving different meals (meal_id). The objective is to predict the number of orders (num_orders) for the next 10 time-steps (week) minimizing the total RMSE. More information about the dataset and problem can be found here:
https://datahack.analyticsvidhya.com/contest/genpact-machine-learning-hackathon-1/

Gradient boosting machine (LightGBM) was used. Various aspects including 'exploratory data analysis, feature engineering, feature importance and hyperparameter optimization relevant to time-series forecasting are presented. The final RMSE obtained for the submitted answers after only 4 attempts was 50.8:
https://datahack.analyticsvidhya.com/contest/genpact-machine-learning-hackathon-1/#LeaderBoard
