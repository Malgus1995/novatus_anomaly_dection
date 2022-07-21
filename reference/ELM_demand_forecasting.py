import pandas as pd
import numpy as np

from pandas import datetime

import matplotlib.pyplot as plt

import math

import scipy.stats as stats
from scipy.linalg import pinv2

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler

def parser(x):
    return datetime.strptime(x, "%Y-%m-%d")

df = pd.read_csv('D:/ml/input/ELM_수요 예측/Sales_Data.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)

df = df.astype(float)
df.plot(style='k', ylabel='Sales')
plt.show()

df = np.array(df)
df = np.reshape(df, (-1, 1))
print(df.shape) # 1826 x 1
    
m = 14                    # lag size 
per = (1736 - m) / 1826
size = int(len(df) * per) # 1722 
d_train, d_test = df[0:size], df[size:len(df)]
print(d_train.shape, d_test.shape) # (1722, 1), (104, 1)

# normalization 

mean_train = np.mean(d_train)
sd_train = np.std(d_train)
d_train = (d_train - mean_train) / sd_train
d_test = (d_test - mean_train) / sd_train

# train test split 

x_train = np.array([d_train[i][0] for i in range(m)]) # (14,)
y_train = np.array(d_train[m][0])

for i in range(1, (d_train.shape[0]-m)):
    l = np.array([d_train[j][0] for j in range(i, i+m)])
    x_train = np.vstack([x_train, l])
    y_train = np.vstack([y_train, d_train[i+m]])
    
x_test = np.array([d_test[i][0] for i in range(m)]) # (14,)
y_test = np.array(d_test[m][0])

for i in range(1, (d_test.shape[0]-m)):
    l = np.array([d_test[j][0] for j in range(i, i+m)])
    x_test = np.vstack([x_test, l])
    y_test = np.vstack([y_test, d_test[i+m]])

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

input_size = x_train.shape[1] # 14
hidden_size = 110 # no.of hidden neurons (100, MAPE 3.083 / 200, 3.214 / 300, 3.574 / 400, 3.685 / 4.017)

mu, sigma = 0, 1
w_lo = -1
w_hi = 1
b_lo = -1
b_hi = 1

input_weights = stats.truncnorm.rvs((w_lo - mu) / sigma, (w_hi - mu) / sigma, loc=mu, scale=sigma, size=[input_size, hidden_size])
biases = stats.truncnorm.rvs((b_lo - mu) / sigma, (b_hi - mu) / sigma, loc=mu, scale=sigma, size=[hidden_size])

print(biases)

def relu(x):
    return np.maximum(x, 0, x)

def hidden_nodes(X):
    G = np.dot(X, input_weights)
    G = G + biases
    H = relu(G)
    
    return H

output_weights = np.dot(pinv2(hidden_nodes(x_train)), y_train)

def predict(X):
    out = hidden_nodes(X)
    out = np.dot(out, output_weights)
    
    return out 

pred = predict(x_test)
correct = 0
total = x_test.shape[0]

y_test = (y_test * sd_train) + mean_train
pred = (pred * sd_train) + mean_train

# evaluate forecast

rmse = math.sqrt(mean_squared_error(y_test, pred))
print('TEST RMSE: %.3f' % rmse)

mape_sum = 0

for i, j in zip(y_test, pred):
    mape_sum = mape_sum + (abs((i - j) / i))

mape = (mape_sum / total) * 100

mpe_sum = 0

for i, j in zip(y_test, pred):
    mpe_sum = mpe_sum + ((i - j) / i)

mpe = (mpe_sum / total) * 100

print('TEST MAPE: %.3f' % mape)
print('TEST MPE: %.3f' % mpe)

# hyperparams : lag_size, activation function, no.of hidden nodes

# plot forecasts against actual outcomes 

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(y_test, label='Actual')
ax.plot(pred, color='red', label='Prediction')
ax.legend(loc='upper right', frameon=False)

plt.xlabel('Days', fontname='Arial', fontsize=24, style='italic', fontweight='bold')
plt.ylabel('Sales Data', fontname='Arial', fontsize=24, style='italic', fontweight='bold')
plt.title('Forecasting for last 3 months with ELM (110 hidden nodes)', fontname='Arial', fontsize=24, style='italic', fontweight='bold')
plt.xticks([0, 20, 40, 60, 80], ['2017-10-02','2017-10-22','2017-11-11','2017-12-01','2017-12-21'], fontname='Arial', fontsize=20, style='italic')
plt.yticks(fontname='Arial', fontsize=22, style='italic')



























