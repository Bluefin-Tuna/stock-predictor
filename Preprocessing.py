import os
import sys
import pandas as pd
import numpy as np

class Preprocessing():


    @staticmethod
    def loadData(path, manipulate_date = True, drop_date = True):

        stocks = os.listdir(path)
        stock_data = []
        for ticker in stocks:
            stock_path = path + "\{}".format(ticker)
            stock = pd.DataFrame(pd.read_csv(stock_path))
            if(manipulate_date):
                stock['Date'] = pd.to_datetime(stock['Date'])
                stock['Day'] = stock['Date'].apply(lambda x: (x.weekday() + 1))
                stock['Month'] = stock['Date'].apply(lambda x: x.month)
                stock['Year'] = stock['Date'].apply(lambda x: x.year)
            if(drop_date):
                stock.drop(labels = 'Date', axis = 1, inplace = True)
            stock_data.append(stock)

        return stock_data


    @staticmethod
    def removeNulls(stock):

        stock.dropna(inplace = True)
        
        return stock


    @staticmethod
    def changeDate(stock):

        stock['Date'] = pd.to_datetime(stock['Date'])
        stock['Day'] = stock['Date'].apply(lambda x: (x.weekday() + 1))
        stock['Month'] = stock['Date'].apply(lambda x: x.month)
        stock['Year'] = stock['Date'].apply(lambda x: x.year)

        return stock


    @staticmethod
    def createDataset(stocks, input_length, output_length, metric_column_index = 3):
        X = []
        Y = []
        for stock in stocks:
            stock = np.asarray(stock)
            if(stock.shape[0] > (input_length + output_length)):
                x = np.zeros(shape = (stock.shape[0] - input_length - output_length + 1, input_length, stock.shape[1]), dtype = 'float32')
                y = np.zeros(shape = (stock.shape[0] - input_length - output_length + 1, output_length), dtype = 'float32')
                for batch in range(stock.shape[0] - input_length - output_length + 1):       
                    x[batch, :, :] = stock[batch : batch + input_length, :]
                    y[batch, :] = stock[batch + input_length : batch + input_length + output_length, metric_column_index]
                X.append(x)
                Y.append(y)
        return np.concatenate(X, axis = 0), np.concatenate(Y, axis = 0)