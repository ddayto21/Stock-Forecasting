# Repository Overview

This repository demonstrates the application of machine learning techniques to predict the likelihood of stock price movements in the S&P 500 index using historical time-series data. By leveraging the Yahoo Finance API, we acquire and preprocess financial data to train a RandomForestClassifier. This classifier models patterns in stock price behavior, providing insights into market trends and potential price movements.

The Yahoo Finance API is an exceptional tool for accessing a wide range of market data, including stocks, bonds, currencies, and cryptocurrencies. It also provides global news reports, offering comprehensive insights into various financial markets.

![S&P-500](images/s&p-logo.jpg)

## Installation and Setup

### Install Yahoo Finance API

The Yahoo Finance API is required to fetch historical data for the S&P 500 index (^GSPC). Install the library using the following command:

```bash
pip install yfinance
```

### Install Matplotlib

For data visualization, we use Matplotlib to plot trends and patterns in the S&P 500 data:

```python
pip install matplotlib
```

### Install Scikit-learn

For machine learning modeling, we leverage Scikit-learn, a versatile library for predictive analytics:

```bash
pip install scikit-learn
```

## Data Collection and Preprocessing

### Load S&P 500 Data

Using the Yahoo Finance API, we retrieve the complete historical price data for the S&P 500 index:

```python
import yfinance as yf

# Fetch historical data
sp500 = yf.Ticker("^GSPC")
sp500 = sp500.history(period="max")
```

### Visualize the S&P 500 Index

Visualize the historical closing prices of the S&P 500 index to observe long-term trends:

```python
import matplotlib.pyplot as plt

# Plot S&P 500 Closing Prices
plt.plot(sp500.index, sp500["Close"], label="S&P 500 Closing Prices")
plt.xlabel("Date")
plt.ylabel("Closing Price")
plt.title("Historical Trends in S&P 500 Index")
plt.legend()
plt.show()
```

### Set Up Target Variables

To predict whether the stock price will increase or decrease, we engineer target variables:

- Tomorrow: Represents the closing price for the following day.
- Target: A binary variable indicating whether the price increased (1) or decreased (0).

```python
# Create target variables
sp500["Tomorrow"] = sp500["Close"].shift(-1)
sp500["Target"] = (sp500["Tomorrow"] > sp500["Close"]).astype(int)

# Restrict data to post-1990 for a more modern analysis
sp500 = sp500.loc["1990-01-01":].copy()
```

## Train a Machine Learning Model

## Random Forest Classifier

The `RandomForestClassifier` is a robust ensemble learning algorithm that combines decision trees to enhance predictive accuracy and reduce overfitting. We’ll train this model using historical S&P 500 data.

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score


model = RandomForestClassifier(
    n_estimators=200,  # Number of decision trees
    min_samples_split=50,  # Minimum samples required to split an internal node
    random_state=1  # Ensure reproducibility
)
```

## Prepare Features and Train the Model

We use historical prices as features to train the model. The features are lagged values of stock prices, allowing the model to learn patterns over time.

```python
# Define feature columns
predictors = ["Close", "Volume", "Open", "High", "Low"]

# Split the data into training and testing sets
train = sp500.iloc[:-100]
test = sp500.iloc[-100:]

# Train the model
model.fit(train[predictors], train["Target"])
```

## Evaluate the Model

Evaluate the model’s predictive performance on unseen data using metrics such as precision:

```python
# Make predictions
predictions = model.predict(test[predictors])

# Evaluate precision
precision = precision_score(test["Target"], predictions)
print(f"Precision: {precision:.2%}")
```

## Key Takeaways

This project showcases the integration of data engineering and machine learning for financial market predictions. By employing the RandomForestClassifier with historical stock data, the model demonstrates the potential to uncover patterns in market movements, aiding in decision-making processes.

### Future Enhancements

1. Feature Expansion: Incorporate technical indicators (e.g., moving averages, RSI) to improve prediction accuracy.
2. Hyperparameter Tuning: Use tools like GridSearchCV or Optuna for optimizing model performance.
3. Model Variants: Experiment with Gradient Boosting algorithms such as XGBoost or LightGBM for comparison.
4. Real-Time Predictions: Extend the solution to provide real-time market predictions using live data feeds.

### Suggestions for Contributions

- Add new features such as technical indicators (e.g., moving averages, RSI).
- Optimize hyperparameters using tools like GridSearchCV or Optuna.
- Extend the project to predict other indices or stocks.
- Enhance visualization capabilities with advanced charting libraries.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments

- Data sourced from Yahoo Finance.
- Inspired by the intersection of financial markets and data-driven decision-making.
