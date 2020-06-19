from investor import *

if __name__ == '__main__':
	
	# ---MAJOR INDEX ANALYSIS---
	print('\n1. OBTAINING MARKET INDICES AND INDEX COMPONENTS:')
	time.sleep(3)
	df_indices, index_components = index_prices(['^FTSE', '^FTMC', '^FTAI', '^GSPC', '^IXIC', '^GDAXI'])
	log_annual_return = logarithmic_returns(df_indices.iloc[-2520:]).mean() * 250
	annual_volatility = (stock_variance(logarithmic_returns(df_indices.iloc[-2520:])) * 250) ** 0.5

	# ---OBTAINING EQUITY DATA - MUST RUN THIS CODE!---
	# Historical equity prices:
	print('\n2. OBTAINING HISTORICAL PRICES:')
	time.sleep(3)
	historical_prices = {'^FTSE':{}, '^FTMC':{}, '^FTAI':{}, '^GSPC':{}, '^IXIC':{}, '^GDAXI':{}}
	historical_prices_tradeables = {'ETF':{}, 'CRYPTOCURRENCY':{}, 'MISC':{}}

	for i in ['^FTSE', '^FTMC', '^FTAI', '^GSPC', '^IXIC', '^GDAXI']:
		equity_prices(i, historical_prices, index_components)

	# ---ETFs---
	# Obtain and save ETF prices.
	etfs = ['VUKE.L', 'VMID.L', 'VUSA.L', 'VWRL.L', 'EQQQ.L']
	df_etfs = etf_prices(etfs)
	for i in etfs:
		historical_prices_tradeables['ETF'][i] = df_etfs[i]

	# ---CRYPTO---
	cryps = ['BTC-GBP', 'XRP-GBP', 'ETH-GBP', 'LTC-GBP']
	for i in cryps:
		historical_prices_tradeables['CRYPTOCURRENCY'][i] = yf.Ticker(i).history(period='max')['Close']

	# ---MISC---
	misc = ['SGLN.L']
	for i in misc:
		historical_prices_tradeables['MISC'][i] = yf.Ticker(i).history(period='max')['Close'].apply(gbx_to_gbp)


	# ---STREAMLINING---
	# This stage is to reduce the number of equities that will undergo fill analysis - filtering stage.
	
	# Initial filtering of the equities:
	print('\n3. INITIAL FILTERING:')
	time.sleep(3)
	stream_eq = (streamline(equities=historical_prices[key], index=key, market_benchmark=dict(log_annual_return), selection='equities') for key in historical_prices)
	stream_td = (streamline(equities=historical_prices_tradeables[key], index=key, market_benchmark=dict(log_annual_return), selection='tradeables') for key in historical_prices_tradeables)

	streamlined_equities = {}
	streamlined_tradeables = {}

	# Obtaining dictionary of dictionaries for the streamlined equities. In form of {'^FTSE': {'XXX':0.2, 'BBB':0.3}, '^FTMC':{'XXX':0.2, 'BBB':0.3}...}
	holder = ({key:i[key]} for i in stream_eq for key in i if len(i[key]) > 0)            
	for entry in holder:
		streamlined_equities.update(entry)

	holder = ({key:i[key]} for i in stream_td for key in i if len(i[key]) > 0)            
	for entry in holder:
		streamlined_tradeables.update(entry)


	# ---EQUITY PERFORMANCE---
	# Detailed equity performance:
	print('\n4. DETAILED STOCK/SECURITY ANALYSIS:')
	time.sleep(3)
	stocks = performance_analysis(historical_prices, streamlined_equities=streamlined_equities, selection='equities')
	tradeables = performance_analysis(historical_prices_tradeables, streamlined_equities=streamlined_tradeables, selection='tradeables')

	# ---DRAFTING THE PORTFOLIO---
	# Obtaining the draft equity portfolio:
	# Dictionary to hold relevant stocks/commodities/ETFs etc. The keys will be the equities' ticker.
	print('\n5. DRAFTING THE PORTFOLIO:')
	time.sleep(3)
	draft_portfolio = {}

	# Keys for the inner dictionaries of the draft_portfolio.
	portfolio_keys = ['name', 'type', 'sector', 'index', 'w_return', 'w_beta', 'w_std', 'w_capm', 'w_sharpe']
    
	# Filter stocks to add to the draft portfolio. Stocks must exceed minimum annual return, and not have a volatility > the specified volatility.
	for i in stocks['index'].unique():
		if i == '^IXIC':
			add_to_draft_portfolio(i, stocks[stocks['index'] == i], draft_portfolio, portfolio_keys, 'equities', 0.085, 0.32)
		elif i == '^GDAXI':
			add_to_draft_portfolio(i, stocks[stocks['index'] == i], draft_portfolio, portfolio_keys, 'equities', 0.065, 0.32)
		else:
			add_to_draft_portfolio(i, stocks[stocks['index'] == i], draft_portfolio, portfolio_keys, 'equities', 0.08, 0.28)

	# Filter stocks to add to the draft portfolio. Stocks must exceed minimum annual return, and not have a volatility > the specified volatility.
	for i in tradeables['index'].unique():
		add_to_draft_portfolio(i, tradeables[tradeables['index'] == i], draft_portfolio, portfolio_keys, 0.01, 10)


	# ---TRIMMING THE DRAFT PORTFOLIO---
	# Revised portfolio - Select the final stocks. Ideally keep below 10no.
	print('\n6. TRIMMING THE PORTFOLIO:')
	time.sleep(3)
	stks = ['AAPL', 'AMZN', 'EXPN.L', 'MSFT', 'HLMA.L', 'GOOGL', 'VNA.DE', 'COST', 'VUSA.L', 'VWRL.L', 'SGLN.L', 'EQQQ.L', 'BTC-GBP']
	print('Creating a revised portfolio comprising of the following:')
	print(stks)
	revised_portfolio = {}
	for stk in stks:
		try:
			revised_portfolio[stk] = draft_portfolio[stk]
		except:
			continue


	# Obtain the historical prices for the revised portfolio. Using the existing historical prices dictionary somehow brings an error?
	print('\n8. PORTFOLIO OPTIMISATION:')
	time.sleep(3)
	fprices = {}
	etfs = []
	for key in list(revised_portfolio.keys()):
	    if revised_portfolio[key]['type'] == 'ETF':
	        etfs.append(key)
	    else:
	        fprices[key] = yf.Ticker(key).history(period='max')['Close']

	df1 = pd.DataFrame(fprices)
	df2 = etf_prices(etfs)
	df = pd.concat([df1, df2], axis=1)


	# Dictionary to store each Portfolio object; keys are the tuples of the stocks/securities/instruments that form that portfolio.
	list_of_portfolios = {}


	# Generate the unique combinations of the stocks/securities/instruments. Number of stocks/securities/instruments in each combination can be altered.
	num_stock_per_combination = 7
	combinations = list(equity_combinations(num_stock_per_combination, list(revised_portfolio.keys())))


	# For each asset combination, initialise a portfolio, perform a basic portfolio returns calculation and save to the portfolio dictionary.
	print("\nCreating final unique portfolios (" +str(num_stock_per_combination)+ " elements per portfolio).")
	for i in combinations:
	    x = Portfolio(i)
	    x.historical_returns(df)
	    list_of_portfolios[i] = (x)
	print(str(len(list_of_portfolios)) + " unique portfolios created.")

	# Optimise each portfolio to find the best sharpe ratio.
	print("\nOptimising each unique portfolio.")
	for key in tqdm(list_of_portfolios):
	    optimise_portfolio(portfolio_object=list_of_portfolios[key], index_prices=df_indices, revised_portfolio_dict=revised_portfolio, n_iterations=1000)
	print("Complete.")

	# Select the best portfolio - can check for highest sharpe ratio or returns, or lowest volatility.
	final_portfolio = find_best_portfolio(list_of_portfolios, 'sharpe_ratio')
	print('\nBest performing portfolio:')
	for i in final_portfolio.optimised_sharpe_ratio['weights']:
	    print(i)
	print('Optimised Sharpe ratio: ' + str(round(final_portfolio.optimised_sharpe_ratio['sharpe_ratio'].iloc[0], 4)))


	save_to_file(final_portfolio)
    