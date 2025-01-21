import yfinance as yf
import matplotlib.pyplot as plt


def collect_data(ticker="^GSPC", start_date="1990-01-01"):
    """
    Fetch historical data for a given stock index ticker and preprocess it.
    """
    print("[+] Collecting data...")
    sp500 = yf.Ticker(ticker).history(period="max")

    # Visualize historical prices
    plt.plot(sp500.index, sp500["Close"], label=f"{ticker} Closing Prices")
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.title(f"{ticker} Historical Prices")
    plt.legend()
    plt.show()

    # Drop irrelevant columns
    sp500.drop(["Dividends", "Stock Splits"], axis=1, inplace=True)

    # Create target variables
    sp500["Tomorrow"] = sp500["Close"].shift(-1)
    sp500["Target"] = (sp500["Tomorrow"] > sp500["Close"]).astype(int)

    # Filter data from the specified start date
    return sp500.loc[start_date:].copy()
