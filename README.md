Food Demand Forecasting 

Forecasting demand is important in many practial applications including food, retail, energy and finance. This repository presents a solution to the problem of predicting how many food items (num_orders) will be ordered from different center (center_id) locations serving different meals (meal_id). The objective is to predict the number of orders (num_orders) for the next 10 time-steps (week) minimizing the total RMSE. Thanks to Analytics Vidhya for providing this dataset. More information can be found here: 
https://datahack.analyticsvidhya.com/contest/genpact-machine-learning-hackathon-1/

Gradient boosting machine (LightGBM: https://lightgbm.readthedocs.io/en/stable/Python-Intro.html) was used. Various aspects including: exploratory data analysis, visualization, feature engineering, feature importance/selection and hyperparameter optimization relevant to time-series forecasting with LightGBM are presented. The RMSE obtained for the submitted answers after only 4 attempts was 50.8:
https://datahack.analyticsvidhya.com/contest/genpact-machine-learning-hackathon-1/#LeaderBoard
