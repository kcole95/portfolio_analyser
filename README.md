# portfolio_analyser
A script which analyses the financial performance of various stocks/assets, and calculates the optimal portfolio based on several financial metrics. These scripts are biased towards UK investments. This is for research only and should not be used to invest real money in the stock market.

## Getting Started
It is recommended that a virtual environment is created. Download the modules listed in the requirements.txt file.

## Usage
1. Run the main_program.py file. Change the assets/stocks added to the 'revised portfolio' as required. The script will create a Portfolio object containing these assets/stocks.
2. The script will create a csv document with the best performing portfolio (in this case, best performing is taken as the highest sharpe ratio). Other permutations of the Portfolio object can be discovered by using the class attributes.
