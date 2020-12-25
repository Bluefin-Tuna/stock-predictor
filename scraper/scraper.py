import selenium
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd
import os
import time

tickers = pd.DataFrame(pd.read_csv("tickers.csv"))
tickers = list(tickers['tickers'])
PATH = "C:\Program Files (x86)\chromedriver.exe"
DELAY = 1

for ticker in tickers:	
	browser = webdriver.Chrome(PATH)
	try:
		print(ticker)
		browser.get("https://finance.yahoo.com/quote/{tick}/history?p={tick}".format(tick = ticker))

		# search = browser.find_element_by_id("yfin-usr-qry")
		# search.send_keys(ticker)
		# time.sleep(2*DELAY)dsfsdfsdfs
		# search.send_keys(Keys.RETURN)

		# link = browser.find_element_by_link_text("Historical Data")
		# link.click()
		# time.sleep(DELAY)

		main = browser.find_element_by_id("Main")
		inner = main.find_element_by_id("Col1-1-HistoricalDataTable-Proxy")
		button = inner.find_element_by_css_selector("#Col1-1-HistoricalDataTable-Proxy > section > div.Pt\(15px\) > div.Bgc\(\$lv1BgColor\).Bdrs\(3px\).P\(10px\) > div:nth-child(1) > div > div > div > span")
		button.click()

		dropdown_max = main.find_element_by_css_selector("#dropdown-menu > div > ul:nth-child(2) > li:nth-child(4) > button > span")
		dropdown_max.click()

		apply_change = inner.find_element_by_css_selector("#Col1-1-HistoricalDataTable-Proxy > section > div.Pt\(15px\) > div.Bgc\(\$lv1BgColor\).Bdrs\(3px\).P\(10px\) > button > span")
		apply_change.click()

		download = inner.find_element_by_css_selector("#Col1-1-HistoricalDataTable-Proxy > section > div.Pt\(15px\) > div.C\(\$tertiaryColor\).Mt\(20px\).Mb\(15px\) > span.Fl\(end\).Pos\(r\).T\(-6px\) > a > span")
		download.click()
		time.sleep(3*DELAY)

		browser.quit()

	except:

		browser.quit()
		continue