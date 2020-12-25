#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import sys


# In[2]:


import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader
from scipy import signal
from datetime import date
import calendar


# In[3]:


print("Num GPUs Available: ", tf.config.experimental.list_physical_devices('GPU'))


# In[4]:


PATH = r"C:\Users\tanus\Deep Learning\Current Datasets\Yahoo Finance Historical Data"


# In[5]:


def load_data(path):
    stocks = os.listdir(path)
    stock_data = []
    for ticker in stocks:
        stock_path = path + "\{}".format(ticker)
        data = pd.DataFrame(pd.read_csv(stock_path))
        data['Date'] = pd.to_datetime(data['Date'])
        stock_data.append(data)
    return stock_data


# In[6]:


def retrieve_day(date):
    return date.weekday()+1


# In[7]:


def retrieve_month(date):
    return date.month


# In[8]:


def retrieve_year(date):
    return date.year


# In[9]:


def preprocess(stocks):
    preprocessed_stocks = []
    for stock in stocks:
        stock['Weekday'] = stock['Date'].apply(lambda x: retrieve_day(x))
        stock['Month'] = stock['Date'].apply(lambda x: retrieve_month(x))
        stock['Year'] = stock['Date'].apply(lambda x: retrieve_year(x))
        stock.drop(labels = ['Date', 'Low', 'Open', 'Close', 'Adj Close', 'Volume'], axis = 1, inplace = True)
        
        preprocessed_stocks.append(stock)
    return preprocessed_stocks


# In[10]:


def create_dataset(stocks, input_length, output_length):
    X = []
    Y = []
    for stock in stocks:
        stock = np.asarray(stock)
        if(stock.shape[0] > (input_length + output_length)):
            x = np.zeros(shape = (stock.shape[0] - input_length - output_length + 1, input_length, stock.shape[1]), dtype = 'float32')
            y = np.zeros(shape = (stock.shape[0] - input_length - output_length + 1, output_length), dtype = 'float32')
            for batch in range(stock.shape[0] - input_length - output_length + 1):       
                x[batch, :, :] = stock[batch : batch + input_length, :]
                y[batch, :] = stock[batch + input_length : batch + input_length + output_length, 0]
            X.append(x)
            Y.append(y)
    return np.concatenate(X, axis = 0), np.concatenate(Y, axis = 0)


# In[11]:


def get_fourier_indicators(stock):
    stock_ft = stock['High']
    high_fft = np.fft.fft(np.asarray(stock_ft.tolist()))
    fft_df = pd.DataFrame({'fft':close_fft})
    fft_df['absolute'] = fft_df['fft'].apply(lambda x: np.abs(x))
    fft_df['angle'] = fft_df['fft'].apply(lambda x: np.angle(x))
    fft_list = np.asarray(fft_df['fft'].tolist())
    for num_ in [3, 6, 9, 100]:
        fft_list_m10= np.copy(fft_list); fft_list_m10[num_:-num_]=0


# In[12]:


def get_technical_indicators(stock):
    stock['ma7'] = stock['High'].rolling(window=7).mean()
    stock['ma21'] = stock['High'].rolling(window=21).mean()
    stock['26ema'] = stock['High'].ewm(span=26)
    stock['12ema'] = stock['High'].ewm(span=12)
    stock['MACD'] = (stock['12ema']-stock['26ema'])
    stock['20sd'] = pd.stats.moments.rolling_std(stock['High'],20)
    stock['upper_band'] = stock['ma21'] + (stock['20sd']*2)
    stock['lower_band'] = stock['ma21'] - (stock['20sd']*2)
    stock['ema'] = stock['High'].ewm(com=0.5).mean()
    stock['momentum'] = stock['High']-1
    return stock


# In[ ]:


def return_top_data(stocks):
    sorted_stocks = []
    stocks_length_usable = []
    for stock in stocks:
        


# In[13]:


stocks = preprocess(load_data(PATH))


# In[12]:


X, Y = create_dataset(stocks, input_length = 90, output_length = 1)


# In[54]:


temp = get_technical_indicators(stocks[0])


# In[41]:


model = Sequential()
model.add(Bidirectional(LSTM(128, activation = 'relu', return_sequences = True, dropout = 0.33, recurrent_dropout = 0.33), input_shape = (X.shape[1], X.shape[2])))
model.add(BatchNormalization())
model.add(Dropout(0.25))
# model.add(Bidirectional(LSTM(128, activation = 'relu', return_sequences = True, dropout = 0.33, recurrent_dropout = 0.33)))
# model.add(BatchNormalization())
# model.add(Dropout(0.25))
# model.add(Bidirectional(LSTM(128, activation = 'relu', return_sequences = True, dropout = 0.33, recurrent_dropout = 0.33)))
# model.add(BatchNormalization())
# model.add(Dropout(0.25))
# model.add(Bidirectional(LSTM(128, activation = 'relu', return_sequences = False, dropout = 0.33, recurrent_dropout = 0.33)))
# model.add(BatchNormalization())
# model.add(Dropout(0.25))
model.add(Dense(1, activation = 'relu'))


# In[42]:


model.compile(optimizer='rmsprop', loss='mae', metrics = ['mean_squared_error'])
model.summary()


# In[43]:


model.fit(x = X, y = Y, epochs = 20, batch_size = 2048)


# In[44]:


stocks[0]


# In[25]:


Y.shape


# In[26]:


X.shape


# In[ ]:




