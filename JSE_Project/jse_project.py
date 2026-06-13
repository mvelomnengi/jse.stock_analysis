# Import the necessary libraries
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt

# Define global constants
RISK_FREE_RATE = 0.08  
TRADING_DAYS = 252     

# Store stock tickers in a list
TICKERS = ["FSR.JO", "SBK.JO", "NPN.JO", "MTN.JO", "AGL.JO"]

# Cache dictionary to store downloaded stock data and avoid redundant API calls
_data_cache = {}

# Define a function to obtain stock data and determine returns
def get_data(stock):

	# Return cached data if already downloaded for this stock
	if stock in _data_cache:
		return _data_cache[stock]

	# Implement error handling
	try:

		# Download historical data for the specified stock
		data = yf.download(stock, start="2022-01-01", end="2025-01-01", auto_adjust=True)

		# Check if data is empty
		if data.empty:
			print(f"Error: No data found for {stock}")
			_data_cache[stock] = (None, None)
			return None, None

		# Extract the closing prices and calculate daily returns
		close_prices = np.array(data["Close"]).flatten()

		# Validate sufficiency of data points
		if len(close_prices) < 2:
			print(f"Error: Insufficient data for {stock}")
			_data_cache[stock] = (None, None)
			return None, None

		# Calculate daily returns as the percentage change in closing prices
		returns = np.diff(close_prices) / close_prices[:-1]  
		last_price = float(close_prices[-1])                 

		# Store the result in the cache before returning
		_data_cache[stock] = (returns, last_price)
		return returns, last_price

	# Display error message
	except Exception as e:
		print(f"Error fetching data for {stock}: {str(e)}")
		_data_cache[stock] = (None, None)
		return None, None

# Define a function to clear the data cache
def clear_cache():
	_data_cache.clear()
	print("Data cache cleared successfully")

# Define a function to obtain stock data and determine returns
def analyse_stock(stock):

	# Implement error handling
	try:

		# Get the returns and last price for the specified stock
		returns, last_price = get_data(stock)

		# Check if returns and last price were successfully retrieved
		if returns is None or last_price is None:
			return

		# Calculate the expected return and volatility of the stock
		mean_ret = np.mean(returns) * TRADING_DAYS
		vol = np.std(returns) * np.sqrt(TRADING_DAYS)

		# Display the analysis results for the stock
		print("\n" + "="*50)
		print(f"Stock Analysis: {stock}")
		print("="*50)
		print(f"Expected Return: {round(mean_ret, 4)} ({round(mean_ret*100, 2)}%)")
		print(f"Volatility: {round(vol, 4)} ({round(vol*100, 2)}%)")
		print(f"Last Price: R{round(last_price, 2)}")
		print("="*50)

	# Display error message
	except Exception as e:
		print(f"Error analyzing stock {stock}: {str(e)}")

# Define a function to compare the expected returns of two stocks
def compare_stocks(stock1, stock2):

	# Implement error handling
	try:

		# Get the returns and last price for both stocks
		r1, p1 = get_data(stock1)
		r2, p2 = get_data(stock2)

		# Check if data was successfully retrieved for both stocks
		if r1 is None or r2 is None:
			print("Error: Could not retrieve data for comparison")
			return

		# Calculate the expected return for both stocks
		ret1 = np.mean(r1) * TRADING_DAYS
		ret2 = np.mean(r2) * TRADING_DAYS

		# Calculate the volatility for both stocks
		v1 = np.std(r1) * np.sqrt(TRADING_DAYS)
		v2 = np.std(r2) * np.sqrt(TRADING_DAYS)

		# Display the comparison results
		print("\n" + "="*50)
		print("Stock Comparison")
		print("="*50)
		print(f"{stock1}:")
		print(f"  Expected Return: {round(ret1, 4)} ({round(ret1*100, 2)}%)")
		print(f"  Volatility: {round(v1, 4)} ({round(v1*100, 2)}%)")
		print(f"  Last Price: R{round(p1, 2)}")
		print(f"\n{stock2}:")
		print(f"  Expected Return: {round(ret2, 4)} ({round(ret2*100, 2)}%)")
		print(f"  Volatility: {round(v2, 4)} ({round(v2*100, 2)}%)")
		print(f"  Last Price: R{round(p2, 2)}")

		# Determine which stock has better return
		if ret1 > ret2:
			print(f"\n{stock1} has higher expected return by {round((ret1-ret2)*100, 2)}%")

		elif ret2 > ret1:
			print(f"\n{stock2} has higher expected return by {round((ret2-ret1)*100, 2)}%")

		else:
			print("\nBoth stocks have the same expected return")
		print("="*50)

	# Display error message
	except Exception as e:
		print(f"Error comparing stocks: {str(e)}")

# Define a function to rank stocks based on their Sharpe ratio
def rank_stocks():

	# Implement error handling
	try:

		# Initialize an empty list to store the results for each stock
		results = []

		# Loop through each stock and calculate its return, volatility, and Sharpe ratio
		for stock in TICKERS:

			# Get the returns and last price for the current stock
			returns, _ = get_data(stock)

			if returns is None:
				print(f"Skipping {stock} due to data error")
				continue

			# Calculate the expected return and volatility for the current stock
			r = np.mean(returns) * TRADING_DAYS
			v = np.std(returns) * np.sqrt(TRADING_DAYS)

			# Calculate the Sharpe ratio
			sharpe = (r - RISK_FREE_RATE) / v if v != 0 else 0

			# Append the stock, return, volatility, and Sharpe ratio to the results list
			results.append((stock, r, v, sharpe))

		if not results:
			print("No data available for ranking")
			return

		# Sort the results based on the Sharpe ratio in descending order
		results.sort(key=lambda x: x[3], reverse=True)

		# Print the ranking of stocks based on their Sharpe ratio
		print("\n" + "="*50)
		print("Stock Ranking (Best to Worst Sharpe Ratio)")
		print("="*50)
		print(f"{'Rank':<5} {'Stock':<10} {'Return':<12} {'Volatility':<12} {'Sharpe Ratio':<12}")
		print("-"*50)

		# Loop through the sorted results and print the results
		for i, r in enumerate(results, 1):
			print(f"{i:<5} {r[0]:<10} {round(r[1], 4):<12} {round(r[2], 4):<12} {round(r[3], 4):<12}")
		print("="*50)

	except Exception as e:
		print(f"Error ranking stocks: {str(e)}")

# Define a function to perform a Monte Carlo simulation for a stock's future price
def monte_carlo(stock):

	# Implement error handling
	try:

		# Get the returns and last price for the specified stock
		returns, last_price = get_data(stock)

		# Check if returns and last price were successfully retrieved
		if returns is None or last_price is None:
			return

		# Calculate mean and std from historical returns
		mean_ret = np.mean(returns)
		std_ret = np.std(returns)

		# Generate paths and collect final prices
		final_prices = []

		# Set up the plot for the Monte Carlo simulation
		plt.figure(figsize=(10, 5))

		# Simulate 500 paths of future stock prices over the next trading year
		for i in range(500):
			prices = [last_price]
			for day in range(TRADING_DAYS):
				daily_return = np.random.normal(mean_ret, std_ret)
				new_price = prices[-1] * (1 + daily_return)
				prices.append(new_price)
			final_prices.append(prices[-1])
			plt.plot(prices, alpha=0.1, linewidth=0.8)

		# Compute statistics
		final_prices = np.array(final_prices)
		expected_price = np.mean(final_prices)
		prob_loss = (final_prices < last_price).mean()
		prob_gain = (final_prices > last_price).mean()
		ci_95 = (np.percentile(final_prices, 2.5), np.percentile(final_prices, 97.5))

		# Print results
		print("\n" + "="*50)
		print(f"Monte Carlo Simulation Results for {stock}")
		print("="*50)
		print(f"Starting Price:      R{round(last_price, 2)}")
		print(f"Expected Price:      R{round(expected_price, 2)}")
		print(f"95% CI:              [R{round(ci_95[0], 2)}, R{round(ci_95[1], 2)}]")
		print(f"Probability of gain: {round(prob_gain*100, 2)}%")
		print(f"Probability of loss: {round(prob_loss*100, 2)}%")
		print("="*50)

		# Plot
		plt.axhline(last_price, color='black', linewidth=1.5, linestyle='--', label='Starting price')
		plt.title(f"Monte Carlo Simulation — {stock} — 500 paths over {TRADING_DAYS} days")
		plt.xlabel("Days")
		plt.ylabel("Price (R)")
		plt.legend()
		plt.grid(True, alpha=0.3)
		plt.tight_layout()
		plt.show()

	# Display error message
	except Exception as e:
		print(f"Error in Monte Carlo simulation: {str(e)}")

# Define a function to estimate the future value of an investment in a stock based on its expected return
def estimate_investment(stock, amount):

	# Implement error handling
	try:

		# Validate the investment amount
		if amount <= 0:
			print("Error: Investment amount must be positive")
			return

		# Get the returns and last price for the specified stock
		returns, last_price = get_data(stock)

		# Check if returns and last price were successfully retrieved
		if returns is None or last_price is None:
			return

		# Calculate the expected return for the stock
		expected = np.mean(returns) * TRADING_DAYS

		# Calculate the future value of the investment based on the expected return
		future_value = amount * (1 + expected)
		profit = future_value - amount

		# Print the investment estimate results
		print("\n" + "="*50)
		print("Investment Estimate")
		print("="*50)
		print(f"Stock: {stock}")
		print(f"Current Price: R{round(last_price, 2)}")
		print(f"Invested Amount: R{round(amount, 2)}")
		print(f"Expected Return (1Y): {round(expected*100, 2)}%")
		print(f"Expected Value (1Y): R{round(future_value, 2)}")
		print(f"Expected Profit: R{round(profit, 2)}")
		print("="*50)

	# Handle any exceptions and display error message
	except Exception as e:
		print(f"Error in investment estimate: {str(e)}")

# Function to display stock menu and get user selection
def select_stock(prompt_message):

	# Loop to display the stock selection menu
	while True:
		print(f"\n{prompt_message}")
		print("\nAvailable Stocks:")
		for i, stock in enumerate(TICKERS, 1):
			print(f"{i}. {stock}")
		print(f"{len(TICKERS)+1}. Cancel")

		# Get the user's choice and validate it
		try:
			choice = input(f"\nEnter your choice (1-{len(TICKERS)+1}): ")
			choice_num = int(choice)

			# Return the selected stock if the choice is valid, or return None if the user chooses to cancel
			if 1 <= choice_num <= len(TICKERS):
				return TICKERS[choice_num - 1]
			elif choice_num == len(TICKERS) + 1:
				return None
			else:
				print(f"Invalid choice. Please enter a number between 1 and {len(TICKERS)+1}")

		# Handle invalid input that cannot be converted to an integer
		except ValueError:
			print("Invalid input. Please enter a valid number")

		# Handle any other unexpected exceptions
		except Exception as e:
			print(f"Unexpected error: {str(e)}")

# Function to select two stocks for comparison
def select_two_stocks():

	# Select the first stock for comparison
	stock1 = select_stock("Select first stock for comparison:")
	if stock1 is None:
		return None, None

	# Select the second stock for comparison
	stock2 = select_stock("Select second stock for comparison:")
	if stock2 is None:
		return None, None

	# Validate user input to ensure different stocks are selected
	if stock1 == stock2:
		print("Error: You cannot compare the same stock. Please select two different stocks.")
		return None, None

	return stock1, stock2

# Main program with while loop controlled by a variable
def main():

	# Use a boolean variable to control the program
	running = True

	# Loop to display the menu and execute user-selected options until the user chooses to exit
	while running:
		print("\n" + "="*50)
		print("STOCK ANALYSIS SYSTEM")
		print("1. Analyse Stock")
		print("2. Compare Two Stocks")
		print("3. Rank Stocks")
		print("4. Investment Estimate")
		print("5. Monte Carlo Simulation")
		print("6. Clear Data Cache")
		print("7. Exit")
		print("="*50)

		# Get the user's choice from the menu
		choice = input("\nSelect option (1-7): ")

		# Analyse a single stock
		if choice == "1":
			stock = select_stock("Select stock to analyze:")
			if stock:
				analyse_stock(stock)

		# Compare two stocks based on their expected returns and volatilities
		elif choice == "2":
			stock1, stock2 = select_two_stocks()
			if stock1 and stock2:
				compare_stocks(stock1, stock2)

		# Rank stocks based on their Sharpe ratio and display the ranking results
		elif choice == "3":
			rank_stocks()

		# Estimate the future value of an investment in a stock based on its expected return
		elif choice == "4":
			stock = select_stock("Select stock for investment:")
			if stock:
				try:
					amount_input = input("Enter investment amount (R): ")
					amount = float(amount_input)
					if amount <= 0:
						print("Error: Investment amount must be positive")
					else:
						estimate_investment(stock, amount)
				except ValueError:
					print("Error: Please enter a valid numeric amount")

		# Perform Monte Carlo simulation for the selected stock
		elif choice == "5":
			stock = select_stock("Select stock for Monte Carlo simulation:")
			if stock:
				monte_carlo(stock)

		# Clear the data cache to force fresh downloads on next use
		elif choice == "6":
			clear_cache()

		# Exit the program and display a goodbye message
		elif choice == "7":
			print("\n" + "="*50)
			print("Thank you for using the Stock Analysis System!")
			print("Goodbye!")
			print("="*50)
			running = False

		# Handle invalid menu options
		else:
			print("\nInvalid option. Please select 1-7 from the menu.")

		# Pause before showing the menu again if the program is still running
		if running:
			input("\nPress Enter to continue...")

# Run the main program
if __name__ == "__main__":
	try:
		main()
	except KeyboardInterrupt:
		print("\n\nProgram interrupted by user. Goodbye!")
	except Exception as e:
		print(f"\nUnexpected error: {str(e)}")
		print("Please restart the program.")
