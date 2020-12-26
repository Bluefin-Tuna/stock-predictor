import pandas as pd
import numpy as np

class TechnicalAnalysis():


	@staticmethod
	def wilderSmoothing(stock, period, metric = 'Close'):
		
		stock['WA'] = stock[metric].ewm(alpha = (1.0 / period), adjust = False).mean()

		return stock


	@staticmethod
	def movingAverage(stock, metric = 'Close', long_interval = 30, short_interval = 7, std_interval = 29, bollinger = True):
		
		stock['MA_{}'.format(short_interval)] = stock[metric].rolling(window = short_interval).mean()
		stock['MA_{}'.format(long_interval)] = stock[metric].rolling(window = long_interval).mean()
		stock['MA_Diff'] = stock['MA_{}'.format(long_interval)] - stock['MA_{}'.format(short_interval)]
		stock['MA_Ratio'] = stock['MA_{}'.format(long_interval)] / stock['MA_{}'.format(short_interval)]
		stock['MA_Volume_{}'.format(short_interval)] = stock['Volume'].rolling(window = short_interval).mean()
		stock['MA_Volume_{}'.format(long_interval)] = stock['Volume'].rolling(window = long_interval).mean()
		stock['MA_Volume_Diff'] = stock['MA_Volume_{}'.format(long_interval)] - stock['MA_Volume_{}'.format(short_interval)]
		stock['MA_Volume_Ratio'] = stock['MA_Volume_{}'.format(long_interval)] / stock['MA_Volume_{}'.format(short_interval)]
		stock['Momentum'] = stock[metric] - 1

		if(bollinger):
			stock['STD'] = stock[metric].rolling(window = std_interval).std()
			stock['UpperBound'] = stock['MA_{}'.format(long_interval)] + (2 * stock['STD'])
			stock['LowerBound'] = stock['MA_{}'.format(long_interval)] - (2 * stock['STD'])

		return stock


	@staticmethod
	def expMovingAverage(stock, metric = 'Close', long_interval = 26, short_interval = 12):

		stock['EMA_{}'.format(short_interval)] = stock[metric].ewm(span = short_interval).mean()
		stock['EMA_{}'.format(long_interval)] = stock[metric].ewm(span = long_interval).mean()
		stock['EMA'] = stock[metric].ewm(com = 0.5).mean()
		stock['MACD'] = stock['EMA_{}'.format(short_interval)] - stock['EMA_{}'.format(long_interval)]

		return stock


	@staticmethod
	def averageTrueRange(stock):

		previous_close = stock['Close'].shift(1)
		previous_close[0] = stock['Close'][0]
		stock['TR'] = np.maximum((stock['High'] - stock['Low']), np.maximum(abs(stock['High'] - previous_close), abs(stock['Low'] - previous_close)))
		stock['ATR_5'] = stock['TR'].ewm(span = 5).mean()
		stock['ATR_15'] = stock['TR'].ewm(span = 15).mean()
		stock['ATR_Ratio'] = stock['ATR_5'] / stock['ATR_15']

		return stock


	@staticmethod
	def rateOfChange(stock, metric = 'Close', period = 15):

		stock['ROC'] = stock[metric].pct_change(periods = period)

		return stock


	@staticmethod
	def stochasticOscillators(stock):

		stock['Lowest_5D'] = stock['Low'].rolling(window = 5).min()
		stock['Highest_5D'] = stock['High'].rolling(window = 5).max()
		stock['Lowest_15D'] = stock['Low'].rolling(window = 15).min()
		stock['Highest_15D'] = stock['High'].rolling(window = 15).max()
		stock['Stochastic_5'] = ((stock['Close'] - stock['Lowest_5D']) / (stock['Highest_5D'] - stock['Lowest_5D'])) * 100
		stock['Stochastic_15'] = ((stock['Close'] - stock['Lowest_15D']) / (stock['Highest_15D'] - stock['Lowest_15D'])) * 100
		stock['Stochastic_MA_5'] = stock['Stochastic_5'].rolling(window = 5).mean()
		stock['Stochastic_MA_15'] = stock['Stochastic_15'].rolling(window = 15).mean()
		stock['Stochastic_Ratio'] = stock['Stochastic_MA_5'] / stock['Stochastic_MA_15']

		return stock


	@staticmethod
	def fourierTrend(stock):

		stock_fft = np.fft.fft(np.asarray(list(stock['Close'])))
		print(stock_fft)
		fft_data = pd.DataFrame({'fft': stock_fft})
		stock['FourierRough'] = fft_data
		stock['Absolute'] = stock['FourierRough'].apply(lambda x: np.abs(x))
		stock['Angle'] = stock['FourierRough'].apply(lambda x: np.angle(x)) 

		return stock


	@staticmethod
	def relativeStrengthIndex(stock, n = 15):
		
		delta = stock['Close'].diff()
		dUp, dDown = delta.copy(), delta.copy()
		dUp[dUp < 0] = 0
		dDown[dDown > 0] = 0
		RolUp = dUp.rolling(window = n).mean()
		RolDown = dDown.rolling(window = n).mean().abs()
		RS = RolUp / RolDown
		stock['RSI'] = 100.0 - (100.0 / (1.0 + RS))
		
		return stock


	@staticmethod
	def averageDirectionalIndex(stock):

		def getCDM(stock):
			dmpos = stock["High"][len(stock) - 1] - stock["High"][len(stock) - 2]
			dmneg = stock["Low"][len(stock) - 2] - stock["Low"][len(stock) - 1]
			if dmpos > dmneg:
				return dmpos
			else:
				return dmneg

		def getDMnTR(stock):
			DMpos = []
			DMneg = []
			TRarr = []
			n = round(len(stock)/14)
			idx = n
			while n <= (len(stock)):
				dmpos = stock["High"][n-1] - stock["High"][n-2]
				dmneg = stock["Low"][n-2] - stock["Low"][n-1]
				DMpos.append(dmpos)
				DMneg.append(dmneg)
				a1 = stock["High"][n-1] - stock["High"][n-2]
				a2 = stock["High"][n-1] - stock["Close"][n-2]
				a3 = stock["Low"][n-1] - stock["Close"][n-2]
				TRarr.append(max(a1,a2,a3))
				n = idx + n
			return DMpos, DMneg, TRarr

		def getDI(stock):
			DMpos, DMneg, TR = getDMnTR(stock)
			CDM = getCDM(stock)
			POSsmooth = (sum(DMpos) - sum(DMpos)/len(DMpos) + CDM)
			NEGsmooth = (sum(DMneg) - sum(DMneg)/len(DMneg) + CDM)
			DIpos = (POSsmooth / (sum(TR)/len(TR))) *100
			DIneg = (NEGsmooth / (sum(TR)/len(TR))) *100
			return DIpos, DIneg

		def getADX(stock):
			DIpos, DIneg = getDI(stock)
			dx = (abs(DIpos- DIneg) / abs(DIpos + DIneg)) * 100
			ADX = dx/14
			return ADX

		stock['ADX'] = getADX(stock)

		return stock

if __name__ == '__main__':
	PATH = r'C:\Users\Tanush\Deep Learning\Current Datasets\Yahoo Finance Historical Data\AAPL.csv'
	aapl = pd.DataFrame(pd.read_csv(PATH))
	aapl.dropna(inplace = True)
	aapl = TechnicalAnalysis.movingAverage(aapl)
	aapl = TechnicalAnalysis.wilderSmoothing(aapl, period = 30)
	aapl = TechnicalAnalysis.expMovingAverage(aapl)
	aapl = TechnicalAnalysis.averageTrueRange(aapl)
	aapl = TechnicalAnalysis.rateOfChange(aapl)
	aapl = TechnicalAnalysis.stochasticOscillators(aapl)
	aapl = TechnicalAnalysis.relativeStrengthIndex(aapl)
	aapl = TechnicalAnalysis.fourierTrend(aapl)
	aapl = TechnicalAnalysis.averageDirectionalIndex(aapl)
	aapl.dropna(inplace = True)
	print(aapl['Lowest_5D'])