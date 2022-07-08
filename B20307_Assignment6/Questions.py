# -*- coding: utf-8 -*-
"""
author: 
    Name : priyanka kumari
    Roll No: B20307
    Mob: 8328354314
"""
import math
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_percentage_error as mape
from statsmodels.tsa.ar_model import AutoReg

# Q1 (a)
# import the csv file
df = pd.read_csv('daily_covid_cases.csv')
df['Date'] = pd.to_datetime(df['Date'], infer_datetime_format=True)
x = df['Date']
y = df['new_cases']
# line plot
plt.plot(x, y)
plt.xlabel('Month-year')
plt.ylabel('New confirmed cases')
plt.title('Line plot of Q1')
plt.show()

# Q1 (b)
print("Autocorrelation coefficient: ", y.autocorr())

# Q1 (c)
plt.scatter(y[:-1], y[1:])
plt.xlabel("Predicted values")
plt.ylabel("Original values")
plt.show()

# Q1 (d)
lag1 = range(1, 7)
corr = []
# loop to calculate autocorrelation for 6 time lags
for i in lag1:
    corr.append(y.autocorr(lag=i))
j = 1
# print the values of autocorrelation
for i in corr:
    print(f'Autocorrelation of lag {j}: ', i)
    j += 1
# line plot between obtained autocorrelation and lagged values
plt.plot(lag1, corr)
plt.xlabel("Lags")
plt.ylabel("Auto-correlation")
plt.show()

# Q1 (e)
plot_acf(y, lags=6)
plt.xlabel("Lag")
plt.show()

# Q2
# splitting data into train and test data
df2 = pd.read_csv('C:/Users/PC/Downloads/daily_covid_cases.csv', parse_dates=['Date'], index_col=['Date'], sep=',')
test_size = 0.35  # 35% for testing data
X = df2.values
tst_sz = math.ceil(len(X) * test_size)
train, test = X[:len(X) - tst_sz], X[len(X) - tst_sz:]

# train AR mode l and predict using the coefficients.

p = 5  # The lag=5
model = AutoReg(train, lags=p, old_names=False)
model_fit = model.fit()  # fit/train the model
coeff = model_fit.params  # Get the coefficients of AR model
print(coeff)

# Q2 (b)
# using these coefficients walk forward over time steps in test, one step each time
hist = train[len(train) - p:]
hist = [hist[i] for i in range(len(hist))]
predictions = list()  # List to hold the predictions, 1 step at a time
for t in range(len(test)):
    length = len(hist)
    lag = [hist[i] for i in range(length - p, length)]
    yhat = coeff[0]  # Initialize to w0
    for d in range(p):
        yhat += coeff[d + 1] * lag[p - d - 1]  # Add other values of coefficient
    obs = test[t]
    predictions.append(yhat)  # Append predictions to compute RMSE later
    hist.append(obs)  # Append actual test value to history, to be used in next step.

# scatter plot
plt.scatter(predictions, test)
plt.xlabel("predicted values")
plt.ylabel("original values")
plt.title("scatter plot between predicted and original values")
plt.show()

# line plot
plt.plot(predictions)
plt.plot(test)
plt.xlabel("predicted values")
plt.ylabel("original values")
plt.title("Line plot between predicted and original values")
plt.show()
# plt.legend()

# RMSE values
RMSE = mse(test, predictions, squared=False)
rmse_5 = (RMSE / (test.mean())) * 100
ms = mape(test, predictions)
print("RMSE(%) with change in lagged values: ", rmse_5)
print("MAPE(%): ", ms * 100)

# Q3
lags = [1, 5, 10, 15, 25]
rmse = []
mape1 = []
for i in lags:
    model = AutoReg(train, lags=i, old_names=False)
    model_fit = model.fit()
    coeff = model_fit.params  # Get the coefficients of AR model
    # print(coeff)
    # using these coefficients walk forward over time steps in test, one step each time
    hist = train[len(train) - i:]
    hist = [hist[k] for k in range(len(hist))]
    predictions = list()  # List to hold the predictions, 1 step at a time
    for t in range(len(test)):
        length = len(hist)
        lag = [hist[j] for j in range(length - i, length)]
        yhat = coeff[0]  # Initialize to w0
        for d in range(i):
            yhat += coeff[d + 1] * lag[i - d - 1]  # Add other values of coefficient
        obs = test[t]
        predictions.append(yhat)  # Append predictions to compute RMSE later
        hist.append(obs)  # Append actual test value to history, to be used in next step.
    RMSE = mse(test, predictions, squared=False)
    rmse.append((RMSE / test.mean() * 100))
    MAPE = mape(test, predictions)
    mape1.append(MAPE * 100)

print("Lags: \t RMSE(%): \t \t MAPE(%):")
for i in range(5):
    print(lags[i], '\t', rmse[i], '\t \t', mape1[i])
plt.bar(lags, rmse)
plt.xlabel('Lags')
plt.ylabel('RMSE(%)')
plt.title("RMSE(%) vs Time lag")
plt.show()
plt.bar(lags, mape1)
plt.xlabel('Lags')
plt.ylabel('MAPE')
plt.title("MAPE vs Time lag")
plt.show()
# Q4
b = []
for i in range(len(train)):
    b.append(train[i][0])
b = pd.Series(b)
i = 1
rmse_2 = []
while (i > 0):
    a = b.autocorr(lag=i)
    if abs(a) <= 2 / (len(train) ** 0.5):
        break
    i = i + 1
hval = i - 1
p = hval  # The lag=hval
model = AutoReg(train, lags=p, old_names=False)
model_fit = model.fit()  # fit/train the model
coeff = model_fit.params  # Get the coefficients of AR model
# print(coeff)
# using these coefficients walk forward over time steps in test, one step each time
hist = train[len(train) - p:]
hist = [hist[i] for i in range(len(hist))]
predictions = list()  # List to hold the predictions, 1 step at a time
for t in range(len(test)):
    length = len(hist)
    lag = [hist[i] for i in range(length - p, length)]
    yhat = coeff[0]  # Initialize to w0
    for d in range(p):
        yhat += coeff[d + 1] * lag[p - d - 1]  # Add other values of coefficient
    obs = test[t]
    predictions.append(yhat)  # Append predictions to compute RMSE later
    hist.append(obs)  # Append actual test value to history, to be used in next step.

print("The heuiristic value for the optical number of lags is", hval)
# RMSE values
RMSE = mse(test, predictions, squared=False)
rmse = (RMSE / (test.mean())) * 100
mape = mape(test, predictions)
print("RMSE(%) and MAP with change in lagged values: ", rmse)
print("MAPE = ", mape * 100)
