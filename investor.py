from datetime import datetime, timedelta
import json
import _pickle as cPickle
from tqdm import tqdm

# Data libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


# Webscraping
from bs4 import BeautifulSoup
import requests

# Financial Analysis
import yfinance as yf
import yahooquery

# import itertools

# Portfolio optimisation
from itertools import chain,repeat,islice,count
from collections import Counter

import time
import re

# Set up the base directory
BASE_DIR = os.getcwd()

# Check if folder for components of main indicies exists. If not, create a folder to save the webscraped info (to be used later on).
if 'index_components' in os.listdir():
	INDEX_COMP_DIR = os.path.join(BASE_DIR, 'index_components')
else:
	os.mkdir('index_components')
	INDEX_COMP_DIR = os.path.join(BASE_DIR, 'index_components')


# Decorator:
def timer(original_func):

	def wrapper(*args, **kwargs):
		start = datetime.today()
		result = original_func(*args, **kwargs)
		end = datetime.today() - start
		print('Total run time: {} '.format(str(end)))
		print()
		return result

	return wrapper



# Main Functions
def gbx_to_gbp(num):
    
    if num < 1000:
        return num
    else:
        return num/100



def simple_returns(data):
    
	'''Simple rate of returns.'''

	returns = (data / data.shift(1)) - 1
	return returns    



def logarithmic_returns(data):
    
	'''Logarithmic rate of returns.'''

	log_returns = np.log(data/data.shift(1))
	return log_returns



def stock_variance(data):
    
	'''Calcuates the variance of the returns of each stock.'''

	variances = data.var() 
	return variances



def stock_covariances(data):
    
	'''Calculates the covariances between the returns of each stock.'''

	cov_matrix = data.cov()
	return cov_matrix



def get_etf_prices(index='eqqq'):
        
    '''Returns a dataframe with the historical prices for the FTSE100 Market Index.
    Uses investing.com as the source for the data.'''
    
    
    indices = {'ftse100':'https://www.investing.com/indices/uk-100-historical-data',
               'ftse250':'https://www.investing.com/indices/uk-250-historical-data',
               'ftseAIM':'https://www.investing.com/indices/ftse-aim-all-share-historical-data',
               's&p500':'https://www.investing.com/indices/us-spx-500-historical-data',
              'nasdaq':'https://www.investing.com/indices/nasdaq-composite-historical-data',
              'dax':'https://www.investing.com/indices/germany-30-historical-data',
              'nikkei225':'https://www.investing.com/indices/japan-ni225-historical-data',
              'eqqq':'https://www.investing.com/etfs/powershares-eqqq-historical-data'}
    
    files = {'ftse100':'historical_market_data/FTSE 100 Historical Data.csv',
             'ftse250':'historical_market_data/FTSE 250 Historical Data.csv',
             'ftseAIM':'historical_market_data/FTSE AIM All Share Historical Data.csv',
             's&p500':'historical_market_data/S&P 500 Historical Data.csv',
            'nasdaq':'historical_market_data/NASDAQ Composite Historical Data.csv',
            'dax':'historical_market_data/DAX Historical Data.csv',
            'nikkei225':'historical_market_data/Nikkei 225 Historical Data.csv',
            'eqqq':'C:/Users/Krist/OneDrive/Desktop/Stock Market Application/Financial Analysis/historical_market_data/eqqq data.csv'}
    
    # Read in the available historical data.
    if index == 'ftse100':
        df1 = pd.read_csv(files['ftse100'], usecols=["date","high","low","open","close","vol"])
    elif index == 'ftse250':
        df1 = pd.read_csv(files['ftse250'], usecols=["date","high","low","open","close","vol"])
    elif index == 'ftseAIM':
        df1 = pd.read_csv(files['ftseAIM'], usecols=["date","high","low","open","close","vol"])
    elif index == 's&p500':
        df1 = pd.read_csv(files['s&p500'], usecols=["date","high","low","open","close","vol"])
    elif index == 'nasdaq':
        df1 = pd.read_csv(files['nasdaq'], usecols=["date","high","low","open","close","vol"])
    elif index == 'dax':
        df1 = pd.read_csv(files['dax'], usecols=["date","high","low","open","close","vol"])
    elif index == 'nikkei225':
        df1 = pd.read_csv(files['nikkei225'], usecols=["date","high","low","open","close","vol"])
    elif index == 'eqqq':
        df1 = pd.read_csv(files['eqqq'], usecols=["date","high","low","open","close","vol"])
        
    
    
    df1['date'] = df1['date'].apply(lambda x: datetime.strptime(x, "%Y-%m-%d"))
    
    #Webscrape latest data
    i = 0
    
    # Save the relevant scraped dated to the correct list
    dates = []
    prices = []
    opens = []
    highs = []
    lows = []
    volumes = []
    
    # Connect to the webpage
    source = requests.get(indices[index], headers={'User-Agent': 'Mozilla/5.0'}).text
    soup = BeautifulSoup(source, 'lxml')

    data = soup.find('table', class_='genTbl closedTbl historicalTbl')
    
    try:
        # review each row of data and append the correct info to each list.
        for element in data.find_all('td'):
    
            if i == 0:
                dates.append(element.text)
            elif i == 1:
                prices.append(element.text)
            elif i == 2:
                opens.append(element.text)
            elif i == 3:
                highs.append(element.text)
            elif i == 4:
                lows.append(element.text)
            elif i == 5:
                volumes.append(element.text)
            elif i == 6:
                pass
    
            i += 1
            if i == 7:
                i = 0
    
    except AttributeError as e:
        print('Could not obtain latest market data for ' + index.upper())
    
    # Create a second dataframe from the lists.
    df2 = pd.DataFrame({'date': dates, 'price': prices, 'high': highs, 'low':lows, 'vol':volumes})
    
    # Format columns;
    # Date
    df2['date'] = df2['date'].apply(lambda x: datetime.strptime(' '.join(''.join(x.split(',')).split()).strip(), '%b %d %Y'))
    
    # Volume
    def format_volume(volume):
        
        '''Formats the volume colume from strings into floats.'''
        
        if 'K' in volume:
            chars = [char for char in volume][:-1]
            chars = ''.join(chars)
            chars = float(chars) * 1000
        
        elif 'M' in volume:
            chars = [char for char in volume][:-1]
            chars = ''.join(chars)
            chars = float(chars) * 1000000
        
        elif 'B' in volume:
            chars = [char for char in volume][:-1]
            chars = ''.join(chars)
            chars = float(chars) * 1000000000
        
        elif volume == '-':
            chars = np.nan
            
        
        return chars
        
    # Turn the volume strings into floats (e.g. 100M -> 1000000)
    df2['vol'] = df2['vol'].apply(format_volume)
    
    # Rename 'price' to 'close'
    df2 = df2.rename(columns={"price": 'close'})
       
    # Combine dataframes and remove duplicates:
    df = pd.concat([df1, df2], axis=0, sort=True)
    df = df.drop_duplicates(subset=['date'], keep='last')
        
    # Set index as date
    df = df.set_index('date')
    df = df.sort_index()
    
    
    # Overrite the existing df1 with the new dataframe. This will only the historical data csv file to be constantly updated.
    df.reset_index().to_csv(files[index], index=False)
    
    return df



class Index():
	
	'''Class to represent a financial market index e.g. the FTSE100 or S&P500.'''
	
	def __init__(self, index):

		# Use yahoofinance to create a Ticker object.
		self.index = index.upper()
		self.ticker = yf.Ticker(self.index)

	
	def historical_prices(self):

		'''Obtains historical price data. Returns close price as a dataframe.'''

		data = pd.DataFrame(self.ticker.history(period='max')['Close'])
		return data


	def get_components(self):

		'''Obtains the equity components of a market index such as the FTSE100 or S&P500.'''
		
		def wiki_connection(url):

			'''Connects to a wepbage to enable web scraping of equity data. url argument is the webpage (usually wikipedia) to connect to.'''

			source = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}).text
			soup = BeautifulSoup(source, 'lxml')

			# Each webpage has a different HTML structure. 
			if self.index == '^IXIC' or self.index == '^GDAXI':
				data = soup.find_all('table', {'class':'wikitable sortable'})
			elif self.index == '^FTAI':
				data = soup.find('table', {'class':'stockTable'})
			else:
				data = soup.find('table', {'class':'wikitable sortable'})
			
			# FTSE100 webpage has a different HTML structure.
			if self.index == '^FTSE':
				table = soup.find_all('tbody')
				return table
			else:
				return data

		# @timer
		def index_loader(index):

			'''Function which loads files containing the constituents of each major index. If the files do not exist, 
			the function connects to relevant wikipedia pages and scrapes relevant data.'''

			pickle_files = {
			'^FTSE': 'ftse100_equities.pickle',
			'^FTMC': 'ftse250_equities.pickle',
			'^FTAI': 'ftseAIM_equities.pickle',
			'^GSPC': 'sp500_equities.pickle',
			'^IXIC': 'nasdaq_equities.pickle',
			'^GDAXI': 'dax_equities.pickle'
			}

			webpages = {
			'^FTSE': 'https://en.wikipedia.org/wiki/FTSE_100_Index',
			'^FTMC': 'https://en.wikipedia.org/wiki/FTSE_250_Index',
			'^FTAI': 'https://www.hl.co.uk/shares/stock-market-summary/ftse-aim-100',
			'^GSPC': 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies',
			'^IXIC': 'https://en.wikipedia.org/wiki/NASDAQ-100',
			'^GDAXI': 'https://en.wikipedia.org/wiki/DAX'
			}

			
			file_location = os.path.join(INDEX_COMP_DIR, pickle_files[index])

			try:
				pickle_in = open(file_location, 'rb')
				table = cPickle.load(pickle_in)
				print()
				print('Loading stock/security components of ' +index+ ' from ' + pickle_files[index])
				return table, 'loaded'
			except:
				table = wiki_connection(webpages[index])
				print()
				print('Scraping stock/security components of ' +index+ ' from ' + webpages[index])
				return table, 'extracted'


		# Stores scraped webpage info (as strings) for each company - name, ticker symbol and sector.
		company = []
		ticker = []
		sector = []
		
		# Counter which tracks the information that is being extracted - company name, ticker or sector.
		i = 0

		# Call the scraper function and determine whether the data has been preloaded or whether futher scraping is needed.
		table, method = index_loader(self.index)

		# If pre-loaded, do nothing.
		if method == 'loaded':
			pass
		# Undertake further scraping specific to the wikipedia page that is being scraped.
		else:
			# FTSE100 wikipedia page:
			if self.index == '^FTSE':
				for item in table[3].find_all('td'):

					# Append company name to 'company' list.
					if i == 0:
						company.append(item.text.strip())

					elif i == 1:
						# Special case to obtain ticker for BT group.
						if item.text.strip().lstrip() == 'BT.A':
							ticker.append('BT-A.L')

						# Special case to obtain ticker for Fresnillo Plc.
						elif item.text.strip().lstrip() == 'FNN':
							ticker.append('FRES.L')

						# Alter ticker to format 'ABC.L' for compatibility with yfinance module. Append to 'ticker' list.
						elif '.' in [char for char in item.text.strip()]:
							ticker.append(item.text.strip() + 'L')
						else:
							ticker.append(item.text.strip() + '.L')

					# Append sector info of the company.
					elif i == 2:
						sector.append(item.text.strip())

					# Once one row has been reviewed from the WIKI table, move to the next row and reset.
					i += 1
					if i == 3:
						i = 0
		        
				# Convert the lists into dataframes and combine into one.
				df = pd.DataFrame(company)
				qf = pd.DataFrame(ticker)
				cf = pd.DataFrame(sector)
				table = pd.concat([df, qf, cf], axis=1)
				table.columns = ['company', 'ticker', 'sector']

				# Write to file.
				FILE_DIR = os.path.join(INDEX_COMP_DIR, 'ftse100_equities.pickle')
				pickle_out = open(FILE_DIR, 'wb')
				cPickle.dump(table, pickle_out)
				pickle_out.close()


			# FTSE250 wikipedia page
			elif self.index == '^FTMC':
				for item in table.find_all('td'):

					# Append company name to 'comapny' list.
					if i == 0:
						company.append(item.text.strip().lstrip())

					# Alter ticker to format 'ABC.L' for compatibility with yfinance module. Append to 'ticker' list.
					elif i == 1:
						if item.text.strip().lstrip() == 'GCC':
							ticker.append('CCR.L')

						elif item.text.strip().lstrip() == 'P2P':
							ticker.append('PSSL.L')

						elif item.text.strip().lstrip() == 'VMU':
							ticker.append('VMUK.L')

						elif '.' in [char for char in item.text.strip()]:
							ticker.append(item.text.strip().lstrip() + 'L')
						else:
							ticker.append(item.text.strip().lstrip() + '.L')

					# Once one row has been reviewed from the WIKI table, move to the next row and reset.
					i += 1
					if i == 2:
						i = 0

				# Convert the lists into dataframes and combine into one.
				df = pd.DataFrame(company)
				qf = pd.DataFrame(ticker)
				table = pd.concat([df, qf], axis=1)
				table.columns = ['company', 'ticker']

				# Write to file.
				FILE_DIR = os.path.join(INDEX_COMP_DIR, 'ftse250_equities.pickle')
				pickle_out = open(FILE_DIR, 'wb')
				cPickle.dump(table, pickle_out)
				pickle_out.close()



			# FTSEAIM wikipedia page
			elif self.index == '^FTAI':
				for item in table.find_all('td'):

					# Append ticker info to 'ticker' list.
					if i == 0:
						ticker.append(item.text.strip() + '.L')

					# Append company name to 'comapny' list.
					elif i == 1:
						company.append(item.text.strip())

					else:
						pass

					i+=1
					if i == 6:
						i=0

				df = pd.DataFrame(company)
				qf = pd.DataFrame(ticker)
				table = pd.concat([df, qf], axis=1)
				table.columns = ['company', 'ticker']

				# Write to file.
				FILE_DIR = os.path.join(INDEX_COMP_DIR, 'ftseAIM_equities.pickle')
				pickle_out = open(FILE_DIR, 'wb')
				cPickle.dump(table, pickle_out)
				pickle_out.close()


			# S&P500 wikipedia page
			elif self.index == '^GSPC':
				for item in table.find_all('td'):

					# Append ticker info to 'ticker' list.
					if i == 0:

						# New Berkshire Hathaway symbol
						if item.text.strip() == 'BRK.B':
							ticker.append('BRK-B')

						# New Brown-Forman Corp symbol
						elif item.text.strip() == 'BF.B':
							ticker.append('BF-B')

						else:
							ticker.append(item.text.strip())

					# Append company name to 'comapny' list.
					elif i == 1:
						company.append(item.text.strip())

					# Do not scrape irrelevant info.
					elif i == 2:
						pass

					# Append sector info of the company.
					elif i == 3:
						sector.append(item.text.strip())

					# Do not scrape irrelevant info.
					else:
						pass

					# Once one row has been reviewed from the WIKI table, move to the next row and reset.
					i += 1
					if i == 9:
						i = 0

				# Convert the lists into dataframes and combine into one.
				df = pd.DataFrame(company)
				qf = pd.DataFrame(ticker)
				cf = pd.DataFrame(sector)
				table = pd.concat([df, qf, cf], axis=1)
				table.columns = ['company', 'ticker', 'sector']

				# Write to file.
				FILE_DIR = os.path.join(INDEX_COMP_DIR, 'sp500_equities.pickle')
				pickle_out = open(FILE_DIR, 'wb')
				cPickle.dump(table, pickle_out)
				pickle_out.close()


			# NASDAQ wikipedia page
			elif self.index == '^IXIC':
				for item in table[1].find_all('td'):

					# Append company name to 'comapny' list.
					if i == 0:
						company.append(item.text.strip())
	 
					# Append ticker info to 'ticker' list.
					elif i == 1:
						ticker.append(item.text.strip().lstrip())

					# Once one row has been reviewed from the WIKI table, move to the next row and reset.
					i += 1
					if i == 2:
						i = 0

				# Convert the lists into dataframes and combine into one.
				df = pd.DataFrame(company)
				qf = pd.DataFrame(ticker)
				table = pd.concat([df, qf], axis=1)
				table.columns = ['company', 'ticker']

				# Write to file.
				FILE_DIR = os.path.join(INDEX_COMP_DIR, 'nasdaq_equities.pickle')
				pickle_out = open(FILE_DIR, 'wb')
				cPickle.dump(table, pickle_out)
				pickle_out.close()


			# DAX wikipedia page
			elif self.index == '^GDAXI':
				for item in table[1].find_all('td'):

					# Do not scrape irrelevant info.
					if i == 0:
						pass

					# Append company name to 'comapny' list.
					elif i == 1:
						company.append(item.text.strip())

					# Append sector info of the company.
					elif i == 2:
						sector.append(item.text.strip())

					# Alter ticker to format 'ABC.DE' for compatibility with yfinance module. Append to 'ticker' list.
					elif i == 3:
						ticker.append(item.text.strip() + '.DE')

					# Do not scrape irrelevant info.
					else:
						pass

					# Once one row has been reviewed from the WIKI table, move to the next row and reset.
					i += 1
					if i == 7:
						i = 0

				# Convert the lists into dataframes and combine into one.
				df = pd.DataFrame(company)
				qf = pd.DataFrame(ticker)
				cf = pd.DataFrame(sector)
				table = pd.concat([df, qf, cf], axis=1)
				table.columns = ['company', 'ticker', 'sector']

				# Write to file.
				FILE_DIR = os.path.join(INDEX_COMP_DIR, 'dax_equities.pickle')
				pickle_out = open(FILE_DIR, 'wb')
				cPickle.dump(table, pickle_out)
				pickle_out.close()

		return table



class Portfolio():

	def __init__(self, assets=[]):

		self.assets = assets

		self.weights = np.round(np.random.random(len(self.assets)), 2)
		self.weights /= np.sum(self.weights)

		self.returns = 0
		self.risk = 0

		self.optimise_tracker = None

		self.optimised_return = np.nan
		self.optimised_risk = np.nan
		self.optimised_sharpe_ratio = np.nan
		self.optimised_beta = np.nan
		self.optimised_predicted_returns = np.nan

	
	def set_weights(self, new_weights=[]):

		'''Manually set weights for the portfolio.'''
		
		new_weights = np.array(new_weights)

		if len(new_weights) == 0:
			print('No weights provided. Randomly generating weights.')
			weights = np.round(np.random.random(len(self.assets)), 2)
			self.weights /= np.sum(weights)
		else:
			self.weights = new_weights


	def historical_returns(self, historical_prices, period='10yr'):
		
		'''Obtains the historical prices for each stock/equity in the portfolio.'''
		

		# Format the length of time to calculate returns over.
		self.analysis_period = int(period.split('yr')[0])

		# Select the historical prices for the portfolio from the historical prices dataframe.
		# 252 trading days in a year. Negative number as we will be selecting the most recent values from the dataframe below.
		self.info = historical_prices[list(self.assets)].iloc[-252 * self.analysis_period:]

		# # Calculate the portfolio return.
		an_returns = logarithmic_returns(self.info).mean() * 250
		portfolio_return = np.dot(an_returns, self.weights)
		self.returns = portfolio_return


	def calculate_risk(self):

		'''Calculates the portfolio risk.'''
		
		# Calculating variance:
		variance = np.dot(self.weights.T, np.dot(logarithmic_returns(self.info).cov() * 250, self.weights))
		volatility = variance ** 0.5

		risks = {'variance': variance, 'volatility/std': volatility}
		self.risk = risks



	def add_assets(self, new_assets={}):
		'''Adds new assets to the portfolio.'''

		self.assets = {**self.assets, **new_assets}



def equity_combinations(r, iterable=None, values=None, counts=None):

	'''Function to calculate all unique combinations from an iterable.

	Function ignores duplicate combinations (i.e. same values in different order).
	E.g.
	r (length of combination) = 2
	iterable = ['a','b','c']

	Gives: [('a', 'b'), ('a', 'c'), ('b', 'c')]
	[('b', 'a'), ('c', 'a'), ('c', 'b')] excluded as they have the same values, just a different order.'''

	if iterable:
		values, counts = zip(*Counter(iterable).items())

	f = lambda i,c: chain.from_iterable(map(repeat, i, c))
	n = len(counts)
	indices = list(islice(f(count(),counts), r))
	if len(indices) < r:
		return
	while True:
		yield tuple(values[i] for i in indices)
		for i,j in zip(reversed(range(r)), f(reversed(range(n)), reversed(counts))):
			if indices[i] != j:
				break
		else:
			return
		j = indices[i]+1
		for i,j in zip(range(i,r), f(count(j), islice(counts, j, None))):
			indices[i] = j



def optimise_portfolio(portfolio_object, index_prices, revised_portfolio_dict, n_iterations=1000):

	indices = index_prices
	port_dict = revised_portfolio_dict


	# 20 year UK risk free rate. 
	risk_free_rate = 0.0078

	# Holds the 5-stock combination
	tickers = list(portfolio_object.info.columns)

	# Holds the annual returns of the portfolio for each random weight allocation
	portfolio_returns = []

	# Holds the annual volatilities of the portfolio for each random weight allocation
	portfolio_volatilities = []

	# Holds the sharpe ratio for each random weight allocation
	sharpes = []

	# Holds the random weights for each iteration
	weightings = []

	portfolio_betas = []

	portfolio_capm = []

	# Comparing 1000 different random weights
	for x in range(n_iterations):

		# Generate the weights and save them.
		weights = np.random.random(len(portfolio_object.info.columns))
		weights /= np.sum(weights)
		zipped_weights = list(zip(portfolio_object.info.columns, weights))
		weightings.append(zipped_weights)

		# Calculate the annual returns and save.
		an_returns = logarithmic_returns(portfolio_object.info.iloc[-252 * portfolio_object.analysis_period:]).mean() * 252
		port_return = np.dot(an_returns, weights)
		portfolio_returns.append(port_return)

		# Calculate the annual volatilities and save.
		vols = np.sqrt(np.dot(weights.T, np.dot(logarithmic_returns(portfolio_object.info.iloc[-252 * portfolio_object.analysis_period:]).cov() * 252, weights)))
		portfolio_volatilities.append(vols)

		# Calculate the sharpe ratios and save.
		sharpe = (port_return - risk_free_rate)/vols
		sharpes.append(sharpe)

		# Portfolio Beta & predicted return
		beta = 0
		capm = 0

		# Unpack the zipped weights into two separate lists
		eqs, wghts = zip(*zipped_weights)

		for i in eqs:
			# Find the portfolio weight associated with the company
			wght = float(wghts[eqs.index(i)])

			# Find the beta of the company
			# company_beta = get_beta(ticker=i, historical_prices=portfolio_object.info[i].iloc[-252 * portfolio_object.analysis_period:], index_prices=indices, port_dict=port_dict)
			asset_beta = port_dict[i]['w_beta']

			# Calculate the weighted_beta in relation to the portfolio weighting
			w_beta = wght * asset_beta

			# Portfolio beta is sum of all weighted betas.
			beta += w_beta

		portfolio_betas.append(beta)


		# CAPM
		for i in eqs:
			# Find the portfolio weight associated with the company
			wght = float(wghts[eqs.index(i)])

			asset_capm = port_dict[i]['w_capm']

			w_capm = wght * asset_capm

			capm += w_capm

		portfolio_capm.append(capm)


	portfolio_returns = np.array(portfolio_returns)
	portfolio_volatilities = np.array(portfolio_volatilities)
	sharpes = np.array(sharpes)

	# Create a dataframe holding all info for each of the 1000 iterations and save within the complete portfolio dict.
	df = pd.DataFrame({'weights': weightings, 'return': portfolio_returns, 'volatility': portfolio_volatilities, 'sharpe_ratio':sharpes, 'portfolio_beta':portfolio_betas, 'predicted_returns':portfolio_capm})
	portfolio_object.optimise_tracker = df

	portfolio_object.optimised_return = df[df['return'] == df['return'].max()]
	portfolio_object.optimised_risk = df[df['volatility'] == df['volatility'].min()]
	portfolio_object.optimised_sharpe_ratio = df[df['sharpe_ratio'] == df['sharpe_ratio'].max()]

	portfolio_object.optimised_beta = df[df['portfolio_beta'] == df['portfolio_beta'].min()]
	portfolio_object.optimised_predicted_returns = df[df['predicted_returns'] == df['predicted_returns'].max()]



def find_best_portfolio(portfolios, metric='sharpe_ratio'):

	if metric.lower() not in ['sharpe_ratio', 'return', 'volatility']:
		print("\nProvide a valid peromance metric ('sharpe_ratio', 'return' or 'volatility').")
	else:
		final = {}
		if metric == 'sharpe_ratio':
			for key in portfolios:
				final[portfolios[key].assets] = portfolios[key].optimised_sharpe_ratio['sharpe_ratio'].iloc[0]
			key = max(final, key = lambda k: final[k])

		elif metric == 'return':
			for key in portfolios:
				final[portfolios[key].assets] = portfolios[key].optimised_return['return'].iloc[0]
			key = max(final, key = lambda k: final[k])

		elif metric == 'volatility':
			for key in portfolios:
				final[portfolios[key].assets] = portfolios[key].optimised_volatility['volatility'].iloc[0]
			key = min(final, key = lambda k: final[k])

		final_portfolio = portfolios[key]
		return final_portfolio




def index_prices(indices=[]):

	'''Obtains historical closing prices and components equities for specified market indices.'''

	names = {'^FTSE':'FTSE100 close',
	'^FTMC':'FTSE250 close',
	'^FTAI':'FTSEAIM close',
	'^GSPC':'S&P500 close',
	'^IXIC':'NASDAQ close',
	'^GDAXI':'DAX close'}

	data = []
	index_components = {}

	for i in indices:
		index = Index(i)
		index_components[i] = index.get_components()
		data.append(index.historical_prices())

	df_indices = pd.concat(data, axis=1)
	df_indices.columns = [names[i] for i in indices]

	return df_indices, index_components



# @timer
def equity_prices(index=None, historical_prices=None, list_of_companies=None):
    
	'''Obtains historical closing prices for specified stocks/equities.'''
    
	names = {'^FTSE':'FTSE100',
	'^FTMC':'FTSE250',
	'^FTAI':'FTSEAIM',
	'^GSPC':'S&P500',
	'^IXIC':'NASDAQ',
	'^GDAXI':'DAX'}

	try:
		index = index.upper()

		# Obtain historical data and add to historical prices dictionary.
		print('\nObtaining historical closing prices for ' + names[index] + ' stocks/securities:')
		for item in tqdm(list_of_companies[index]['ticker']):
			historical_prices[index][item] = yf.Ticker(item).history(period='max')['Close']

	except:
		print('Could not obtain data.')



# @timer
def streamline(equities={}, index=None, market_benchmark=None, selection=None):

	'''Initial filtering of specified stock/equities. Returns the stocks with the highest returns over the last 10 years.'''
	
	index = index.upper()
	top_performers = {'^FTSE':{}, '^FTMC':{}, '^FTAI':{}, '^GSPC':{}, '^IXIC':{}, '^GDAXI':{}, '^CMC200':{}}

	if selection == 'equities':

		names = {'^FTSE':'FTSE100 close',
		'^FTMC':'FTSE250 close',
		'^FTAI':'FTSEAIM close',
		'^GSPC':'S&P500 close',
		'^IXIC':'NASDAQ close',
		'^GDAXI':'DAX close'}

		market_returns = market_benchmark[names[index]]

	elif selection == 'tradeables':
		
		# Get CRYPTO Market benchamrk prices
		cmc200 = yf.Ticker('^CMC200').history(period='max')['Close']

		# Pattern matching
		phrases = {'FTSE':'^FTSE', 'Nasdaq':'^IXIC', '500':'^GSPC', 'Gold':'^FTSE', 'coin':'^CMC200'}
		for i in equities:
			if i == 'EQQQ.L':
				name = 'PowerShares EQQQ Nasdaq-100 UCITS ETF'
			else:
				try:
					name = yf.Ticker(i).info['longName']
				except:
					name = yf.Ticker(i).info['shortName']

			for phrase in phrases:
				# Search the name of the tradable ans see if the phrase is present.
				pattern = re.compile(phrase)
				if pattern.search(name):
					try:
						market_returns = market_benchmark[names[phrases[phrase]]]
						index = phrases[phrase]
						break
					except:
						market_returns = logarithmic_returns(cmc200.iloc[-2520:]).mean() * 250
						index = phrases[phrase]
						break
				else:
					market_returns = market_benchmark['FTSE100 close']

	try:
		if selection == 'equities':
			# Obtain annual returns for each stock over the last 10 years.
			print('\nObtaining annual logarithmic returns for each stock/security in ' +index+ ' (last 10 years).')
		elif selection == 'tradeables':
			print('\nObtaining annual logarithmic returns for each tradeable (last 10 years).')
		
		for i in equities:
			# Get the 10 yearly data.
			prices = equities[i].iloc[-2520:]

			# Calculate the returns over that period.
			log_annual_return = logarithmic_returns(prices).mean() * 250

			if selection == 'equities':
				if log_annual_return < 1 and log_annual_return > 0.1 and log_annual_return > market_returns:
					top_performers[index][i] = log_annual_return

				elif (stock_variance(logarithmic_returns(prices)) * 250) ** 0.5 < 0.2:
					top_performers[index][i] = log_annual_return

			elif selection == 'tradeables':

				if log_annual_return > 0.05 or log_annual_return > market_returns:
					top_performers[index][i] = log_annual_return

				elif (stock_variance(logarithmic_returns(prices)) * 250) ** 0.5 < 0.2:
					top_performers[index][i] = log_annual_return

		print("Obtained highest returning stocks/securities over the last 10 years.")
		return top_performers

	except:
		print('Error, could not calculate annual returns.')



@timer
def performance_analysis(historical_equity_prices, streamlined_equities={}, selection=None, historical_index_prices=None, market_index=None):

	'''Returns key financial data for a specified stocks/equities. Calculates KPIs including returns, variances/volatilities, betas and makes predictions on future performance via the Capital Asset Pricing Model.'''

	names = {'^FTSE':'FTSE100 close',
	'^FTMC':'FTSE250 close',
	'^FTAI':'FTSEAIM close',
	'^GSPC':'S&P500 close',
	'^IXIC':'NASDAQ close',
	'^GDAXI':'DAX close'}

	def market_performance(index=None):

		'''Obtains key financial data for a specified market index including annualised returns and variances.'''

		try:
			# Obtain data from pre-defined index dataframe.
			df = df_indices[names[index]]

			one_yr_return = logarithmic_returns(df.iloc[-252:])
			one_yr_annualised = one_yr_return.mean() * 252
			one_yr_var = logarithmic_returns(df.iloc[-252:]).var() * 252

			five_yr_return = logarithmic_returns(df.iloc[-1260:])
			five_yr_annualised = five_yr_return.mean() * 252
			five_yr_var = logarithmic_returns(df.iloc[-1260:]).var() * 252

			ten_yr_return = logarithmic_returns(df.iloc[-2520:])
			ten_yr_annualised = ten_yr_return.mean() * 252
			ten_yr_var = logarithmic_returns(df.iloc[-2520:]).var() * 252

			return one_yr_return, one_yr_annualised, one_yr_var, five_yr_return, five_yr_annualised, five_yr_var, ten_yr_return, ten_yr_annualised, ten_yr_var

		except:
			pricing = yf.Ticker(index).history(period='max')['Close']
			one_yr_return = logarithmic_returns(pricing.iloc[-252:])
			one_yr_annualised = one_yr_return.mean() * 250
			one_yr_var = logarithmic_returns(pricing.iloc[-252:]).var() * 250

			five_yr_return = logarithmic_returns(pricing.iloc[-1260:])
			five_yr_annualised = five_yr_return.mean() * 250
			five_yr_var = logarithmic_returns(pricing.iloc[-1260:]).var() * 250

			ten_yr_return = logarithmic_returns(pricing.iloc[-2520:])
			ten_yr_annualised = ten_yr_return.mean() * 250
			ten_yr_var = logarithmic_returns(pricing.iloc[-2520:]).var() * 250

			return one_yr_return, one_yr_annualised, one_yr_var, five_yr_return, five_yr_annualised, five_yr_var, ten_yr_return, ten_yr_annualised, ten_yr_var

	index_performance_holder = {'^FTSE':[], '^FTMC':[], '^FTAI':[], '^GSPC':[], '^IXIC':[], '^GDAXI':[], '^CMC200':[]}

	for key in index_performance_holder:
		index_performance_holder[key] = list(market_performance(key))

	# Will store lists of financial/performance data for each equity.
	data = []

	for key in streamlined_equities:

		if selection == 'tradeables':
			print('\nPerforming full historical analysis on tradeables:\n')
		
		else:
			# if selection == 'equities':
			print('\nPerforming full historical analysis on ' + names[key].split()[0] + ' stocks/securities:\n')



		# Obtain the appropriate market index data for each index.
		one_yr_return_market = index_performance_holder[key][0]
		one_yr_annualised_market = index_performance_holder[key][1]
		one_yr_var_market = index_performance_holder[key][2] 
		five_yr_return_market = index_performance_holder[key][3] 
		five_yr_annualised_market = index_performance_holder[key][4] 
		five_yr_var_market = index_performance_holder[key][5] 
		ten_yr_return_market = index_performance_holder[key][6] 
		ten_yr_annualised_market = index_performance_holder[key][7] 
		ten_yr_var_market = index_performance_holder[key][8]


		for equity in tqdm(streamlined_equities[key]):
			# 1. Obtaining general financial information about the company.
			# Create a Ticker object using that company's ticker. Using yahooquery as yf cannot provide info for many tickers.
			stock = yahooquery.Ticker(equity)

			# Historical price
			try:
				hist = historical_equity_prices[key][equity]
			except:
				if equity == 'SGLN.L':
					hist = yf.Ticker(equity).history(period='max')['Close'].apply(gbx_to_gbp)
				else:
					hist = yf.Ticker(equity).history(period='max')['Close']

			# Ticker
			ticker = equity

			if selection == 'equities':
				try:
					name = index_components[key][['company', 'ticker']].set_index(['ticker']).to_dict()['company'][equity].upper()
				except:
					name = equity

			# Get names for tradeables.
			elif selection == 'tradeables':
				if equity == 'EQQQ.L':
					name = 'PowerShares EQQQ Nasdaq-100 UCITS ETF'.upper()
				else:
					try:
						name = yf.Ticker(equity).info['longName'].upper()
					except:
						name = yf.Ticker(equity).info['shortName'].upper()

			# Legal type
			if selection == 'tradeables':

				# Obtain the quotype of the instrument - ETF, CRYPTOCURRENCY, EQUITY etc.
				try:
					legal_type = yf.Ticker(equity).info['quoteType']
				except:
					legal_type = 'MISC'
				
			else:
				legal_type = 'Equity'

			# Obtain the company sector.
			try:
				sector = stock.asset_profile[equity]['sector']
			except:
				sector = 'N/A'

			# Obtain the latest closing price.
			try:
				close = stock.summary_detail[equity]['previousClose']
			except:
				close = np.nan

			# Obtain the company's market capitalisation. 
			try:    
				market_cap = stock.summary_detail[equity]['marketCap']
			except:
				market_cap = np.nan

			# Obtain the dividend rate.
			try:    
				dividend = stock.summary_detail[equity]['dividendRate']
			except:
				dividend = np.nan

			# Obtain the earnings quarterly growth.
			try:    
				growth = stock.key_stats[equity]['earningsQuarterlyGrowth']
				if growth == {}:
					growth = np.nan            
			except:
				growth = np.nan

			# Obtain the foward price/earnings ratio (some companies do not have this data).
			try:
				forwardpe = stock.summary_detail[equity]['forwardPE']
				if forwardpe == {}:
					forwardpe = np.nan  
			except:
				forwardpe = np.nan

			# Obtain the trailing price/earnings ratio.
			try: 
				trailingpe = stock.summary_detail[equity]['trailingPE']
			except:
				trailingpe = np.nan


			# 2. BETA

			if legal_type == 'CRYPTOCURRENCY':
				one_yr_return_equity = logarithmic_returns(hist.iloc[-252:])
				five_yr_return_equity = logarithmic_returns(hist.iloc[-1260:])
				ten_yr_return_equity = logarithmic_returns(hist.iloc[-2520:])

				# Approximate the beta as there is no real arket benchmark to compare the currencies to.
				beta_1, beta_5, beta_10 = (1, 1, 1)

			else:
				# Calculate the returns of the stock/tradeable over the last year.
				one_yr_return_equity = logarithmic_returns(hist.iloc[-252:])
				# Calcualte the Beta of the stock/tradeable by calculating the covariance between the stock and the market, and the variance of the market.
				cov_with_market_1 = one_yr_return_equity.cov(one_yr_return_market) * 252
				beta_1 = cov_with_market_1/one_yr_var_market

				# Calculate the returns of the stock/tradeable over the last year.
				five_yr_return_equity = logarithmic_returns(hist.iloc[-1260:])
				# Calcualte the Beta of the stock/tradeable by calculating the covariance between the stock and the market, and the variance of the market.
				cov_with_market_5 = five_yr_return_equity.cov(five_yr_return_market) * 252
				beta_5 = cov_with_market_5/five_yr_var_market

				# Calculate the returns of the stock/tradeable over the last year.
				ten_yr_return_equity = logarithmic_returns(hist.iloc[-2520:])
				# Calcualte the Beta of the stock/tradeable by calculating the covariance between the stock and the market, and the variance of the market.
				cov_with_market_10 = ten_yr_return_equity.cov(ten_yr_return_market) * 252
				beta_10 = cov_with_market_10/ten_yr_var_market


			# 3. Calculating the volatility of the returns.
			# Annualised returns and risk
			one_yr_annualised_equity = one_yr_return_equity.mean() * 252
			one_yr_vol_equity = (stock_variance(logarithmic_returns(hist[-252:])) * 252) ** 0.5

			five_yr_annualised_equity = five_yr_return_equity.mean() * 252
			five_yr_vol_equity = (stock_variance(logarithmic_returns(hist[-1260:])) * 252) ** 0.5

			ten_yr_annualised_equity = ten_yr_return_equity.mean() * 252
			ten_yr_vol_equity = (stock_variance(logarithmic_returns(hist[-2520:])) * 252) ** 0.5


			# 4. Predicting returns using the CAPM model.
			# CAPM expected returns
			one_yr_exp_return = 0.0008 + (beta_1 * 0.065)
			five_yr_exp_return = 0.0016 + (beta_5 * 0.065)
			ten_yr_exp_return = 0.0033 + (beta_10 * 0.065)


			# 5. Sharpe ratio.
			# Calculating Sharpe ratio
			sharpe_ratio_1 = (one_yr_exp_return - 0.001)/(one_yr_return_equity.std() * 252 ** 0.5)
			sharpe_ratio_5 = (five_yr_exp_return - 0.0031)/(five_yr_return_equity.std() * 252 ** 0.5)
			sharpe_ratio_10 = (ten_yr_exp_return - 0.0055)/(ten_yr_return_equity.std() * 252 ** 0.5)


			# 6. Obtaining final weighted scores for Beta, returns, volatility, expected returns and sharpe ratio.
			# Define the weights. Want greater bias towards 5yr and 10yr performances. 
			weights = np.array([0.25, 0.375, 0.375])


			# Weighted Beta
			betas = np.array([beta_1, beta_5, beta_10])
			weighted_beta = np.dot(betas, weights.T)

			# Weighted Returns
			returns = np.array([one_yr_annualised_equity, five_yr_annualised_equity, ten_yr_annualised_equity])
			weighted_return = np.dot(returns, weights.T)

			# Weighted Volatility
			volatilities = np.array([one_yr_vol_equity, five_yr_vol_equity, ten_yr_vol_equity])
			weighted_volatility = np.dot(volatilities, weights.T)

			# Weighted Exp returns
			exp_returns = np.array([one_yr_exp_return, five_yr_exp_return, ten_yr_exp_return])
			weighted_capm = np.dot(exp_returns, weights.T)

			# Weighted Sharpe Ratio
			sharpes = np.array([sharpe_ratio_1, sharpe_ratio_5, sharpe_ratio_10])
			weighted_sharpe_ratio = np.dot(sharpes, weights.T)


			# 7. Collating all info.
			info = [ticker, name, legal_type, sector, key, close, market_cap, dividend, growth, forwardpe, trailingpe]

			info.extend([one_yr_annualised_equity, one_yr_vol_equity, beta_1, one_yr_exp_return, sharpe_ratio_1,
				five_yr_annualised_equity, five_yr_vol_equity, beta_5, five_yr_exp_return, sharpe_ratio_5,
				ten_yr_annualised_equity, ten_yr_vol_equity, beta_10, ten_yr_exp_return, sharpe_ratio_10,
				weighted_beta, weighted_return, weighted_volatility, weighted_capm, weighted_sharpe_ratio])


			data.append(info)


	# 8. Collate all the information as a single dataframe.    
	stocks = pd.DataFrame(data)
	stocks.columns = ['ticker', 'shortName', 'type','sector', 'index', 'close','marketCap','dividendRate',
	'earningsQuarterlyGrowth', 'forwardPE', 'trailingPE', '1yr_return', '1yr_volatility', 'beta_1', '1yr_exp_return', 
	'sharpe_ratio_1', '5yr_return', '5yr_volatility', 'beta_5', '5yr_exp_return', 'sharpe_ratio_5', '10yr_return', 
	'10yr_volatility', 'beta_10', '10yr_exp_return', 'sharpe_ratio_10', 'weighted_beta', 'weighted_annual_return', 
	'weighted_annual_volatility','weighted_capm', 'weighted_sharpe_ratio']

	print("\nSuccessfully analysed the top performing stocks/securities from the indices provided.")
	time.sleep(3)

	return stocks


# @timer
def add_to_draft_portfolio(index=None, equities=None, draft_portfolio=None, keys=None, selection=None, min_weighted_return=0.08, max_weighted_volatility=0.27, min_capm=0.035):

	'''Adds the stocks/equities to a draft portfolio.'''
	

	# Remove duplicates
	equities = equities.drop_duplicates(subset=['shortName'], keep='last')

	if selection == 'equities':
		# Add AAPL, MSFT & MSFT to the portfolio regardless of performance:
		if index == '^IXIC':
			try:
				for company in ['AAPL', 'MSFT', 'AMZN']:
					draft_portfolio[company] = dict(zip(keys, list(equities.set_index('ticker')[['shortName', 'type', 'sector', 'index','weighted_annual_return', 'weighted_beta', 'weighted_annual_volatility', 'weighted_capm', 'weighted_sharpe_ratio']].loc[company])))
			except:
				print('Could not add ' +company+ ' to the draft portfolio.')

		# Conditions for selection:
		# Positive quarterly growth 
		growth = equities['earningsQuarterlyGrowth'] > 0

		# Minimum expected return of 3.5%
		weighted_capm = equities['weighted_capm'] >= min_capm

		# Min returns
		weighted_returns = equities['weighted_annual_return'] >= min_weighted_return

		# Max volatility
		weighted_volatility = equities['weighted_annual_volatility'] <= max_weighted_volatility

	else:
		pass

	if selection == 'equities':
		# For each equity which matches these conditions:
		for i in range(len(equities[(weighted_returns) & (weighted_volatility) & (weighted_capm) & (growth)])):
	
			# Append to the draft portfolio a dictionary with the following metrics: company name (KEY), and (VALUES); (weighted) returns, beta, volatility and sharpe ratio.
			draft_portfolio[equities[(weighted_returns) & (weighted_volatility) & (weighted_capm) & (growth)]['ticker'].iloc[i]] = dict(zip(keys, list(equities[['shortName', 'type', 'sector', 'index','weighted_annual_return', 'weighted_beta', 'weighted_annual_volatility', 'weighted_capm', 'weighted_sharpe_ratio']][(weighted_returns) & (weighted_volatility) & (weighted_capm) & (growth)].iloc[i])))

	else:
		for i in range(len(equities)):
	
			# Append to the draft portfolio a dictionary with the following metrics: company name (KEY), and (VALUES); (weighted) returns, beta, volatility and sharpe ratio.
			draft_portfolio[equities['ticker'].iloc[i]] = dict(zip(keys, list(equities[['shortName', 'type', 'sector', 'index','weighted_annual_return', 'weighted_beta', 'weighted_annual_volatility', 'weighted_capm', 'weighted_sharpe_ratio']].iloc[i])))

	print("Successfully added " +index+ " stocks/securities to the draft portfolio.")




def etf_prices(etfs=[]):

	# Convert ETF tickers to upper case if not already the case.
	etfs = [etf.upper() for etf in etfs]

	etf_prices = {}
	for etf in etfs:

		# Obtain historical ETF prices (avoid Invesco EQQQ as it is a special case)
		if etf.upper() != 'EQQQ.L':
			try:
				etf_prices[etf] = yf.Ticker(etf).history(period='max')['Close']
			except:
				print('\nCould not obtain historical price data for ' + etf.upper() + '.')

		else:
			eqqq = get_etf_prices()


	# Convert ETF pricing data into a single dataframe. Column names are ETF tickers, values are the historical closing price.
	df = pd.DataFrame(etf_prices)

	if 'EQQQ.L' in etfs:
		# Format the dataframe for the EQQQ ETF.
		eqqq['close'] = eqqq['close'].apply(lambda x: float(''.join(x.split(','))))
		eqqq['high'] = eqqq['high'].apply(lambda x: float(''.join(x.split(','))))
		eqqq['low'] = eqqq['low'].apply(lambda x: float(''.join(x.split(','))))
		eqqq.columns = ['EQQQ.L', 'High', 'Low', 'Open', 'Vol.']

		# Add the EQQQ dataframe to the main dataframe.
		df = pd.concat([df, eqqq['EQQQ.L']], axis=1)

	return df


def index_analysis(df_prices):

	if len(df_prices) > 0:
		try:
			log_annual_return = logarithmic_returns(df_indices.iloc[-2520:]).mean() * 250
			annual_volatility = (stock_variance(logarithmic_returns(df_indices.iloc[-2520:])) * 250) ** 0.5

			return log_annual_return, annual_volatility

		except:
			print('Error, could not analyse the index data provided.')

	else:
		print('No data to analyse.')


def save_to_file(best_portfolio):


	# Check if folder to save portfolio metrics exists. If not, create a folder.
	if 'portfolio_metrics' in os.listdir():
		PORTFOLIO_DIR = os.path.join(BASE_DIR, 'portfolio_metrics')
	else:
		os.mkdir('portfolio_metrics')
		PORTFOLIO_DIR = os.path.join(BASE_DIR, 'portfolio_metrics')


	
	# File displaying key metrics: optimised sharpe, weights, beta, returns, volatility etc
	metrics = {
		'Highest sharpe ratio': best_portfolio.optimised_sharpe_ratio['sharpe_ratio'].iloc[0],
		'weights': best_portfolio.optimised_sharpe_ratio['weights'].iloc[0],
		'return': best_portfolio.optimised_sharpe_ratio['return'].iloc[0],
		'volatility': best_portfolio.optimised_sharpe_ratio['volatility'].iloc[0],
	}

	pd.DataFrame(metrics).to_csv(os.path.join(PORTFOLIO_DIR, 'best_portfolio_metrics.csv'), index=False)
	
	# Save PNG showing sharpe_ratio graph
	
	
	# Save iterations
	best_portfolio.optimise_tracker.to_csv(os.path.join(PORTFOLIO_DIR, 'portfolio_optimisation_tracker.csv'), index=False)
