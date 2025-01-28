import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import joblib
import os

#load data file
data = pd.read_csv(r"C:\Users\manos\OneDrive\Desktop\MSU\Spring 2025\STT 890 ML-Ops\HW\1\STT890-MLOps\HW1\data\sampregdata.csv")
print(data.shape)
data = data.drop(columns = ['Unnamed: 0'])

#check for missing values 
total = data.isnull().sum().sort_values(ascending = False)
percent = (data.isnull().sum()/data.isnull().count()*100).sort_values(ascending = False)
pd.concat([total, percent], axis=1, keys=['Total', 'Percent']).transpose()

#find best fitting X
print(data.corr())
# x4 has the highest absolute correlation with y, followed by x2 

## Q1-b
#split test and train data
x = data[['x4']]
y = data[['y']]

#train on 70% of the data, test on the remaining 30%
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=50)
lr = LinearRegression()
lr.fit(x_train, y_train)
y_pred = lr.predict(x_test)

#results
print("MSE model 1 =", mean_squared_error(y_test, y_pred))
print("R^2 Score model 1 =", r2_score(y_test, y_pred))
# MSE = 85.07018698169895
# R^2 Score = 0.2752714488833352

#organize work
os.makedirs("HW1/model", exist_ok=True)
joblib.dump(lr, "HW1/model/lr_model1.pkl")
x_train.to_csv("HW1/data/x_train1.csv", index=False)
x_test.to_csv("HW1/data/x_test1.csv", index=False)
y_train.to_csv("HW1/data/y_train.csv", index=False)
y_test.to_csv("HW1/data/y_test.csv", index=False)

## Q1-c
x = data[['x2', 'x4']]
y = data[['y']]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=50)
lr = LinearRegression()
lr.fit(x_train, y_train)
y_pred = lr.predict(x_test)

print("MSE model 2 =", mean_squared_error(y_test, y_pred))
print("R^2 Score model 2 =", r2_score(y_test, y_pred))
# MSE = 56.49218958495245
# R^2 Score = 0.5187326587618971

joblib.dump(lr, "HW1/model/lr_model2.pkl")
x_train.to_csv("HW1/data/x_train2.csv", index=False)
x_test.to_csv("HW1/data/x_test2.csv", index=False)