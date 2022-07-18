# Yahoo Finance API - Stock Index Prices
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
import pandas as pd

def Collect_Data():
    sp500 = yf.Ticker("^GSPC")
    sp500 = sp500.history(period="max")
    plt.plot(sp500.index, sp500["Close"])
    # plt.show()
    del sp500["Dividends"]
    del sp500["Stock Splits"]
    print(sp500)
    # SET UP TARGET VARIABLE
    sp500["Tomorrow"] = sp500["Close"].shift(-1)
    sp500["Target"] = (sp500["Tomorrow"] > sp500["Close"]).astype(int) # Convert True/False --> 1/0
    # 0 --> Price Went Down Tomorrow
    # 1 --> Price Went Up Tomorrow
    sp500 = sp500.loc["1990-01-01":].copy()
    return sp500

def Train_Model(data):
    print("[+] Training Model... \n Data: ", data)
    
    model = RandomForestClassifier(n_estimators=200, min_samples_split=50, random_state=1) 
                                 # n_estimators: Number of Decision Trees | Higher Number --> More Accuracy (to an extent)
    train = data.iloc[:-100] # Use every row in time-series except the last 100 days for training
    print("Train Dataset: ", train)
    test = data.iloc[-100:] # Test the Model on the last 100 rows in time-series for testing
    print("Test Dataset: ", test)
    predictors = ["Close", "Volume", "Open", "High", "Low"]
    model.fit(train[predictors], train["Target"])
    preds = model.predict(test[predictors])
    # print("[+] Preds: \n\n", preds)
    preds = pd.Series(preds, index=test.index)
    print("Initial Predictions: \n\n", preds)
    precision_score(test["Target"], preds)
    # How is the precision score calculated?
    print("Precision Score: \n ", precision_score)
    combined = pd.concat([test["Target"], preds], axis=1)
    predictions = BackTest(data, model, predictors)
    print(predictions["Predictions"].value_counts())
    p_score = precision_score(predictions["Target"], predictions["Predictions"])
    print("Precision Score: ", p_score)
    predictions["Target"].value_counts() / predictions.shape[0]

    horizons = [2,5,60, 250, 1000]
    new_predictors = []
    for x in horizons:
        print("Time-Period: ", x)
        rolling_averages = data.rolling(x).mean()
        print("Rolling Averages: ", rolling_averages)
        ratio_column = f"Close_Ratio_{x}"
        data[ratio_column] = data["Close"] / rolling_averages["Close"]
        trend_column = f"Trend_{x}"
        data[trend_column] = data.shift(1).rolling(x).sum()["Target"]
        new_predictors += [ratio_column, trend_column]
    print("Time-Series Dataframe: ", data)
    print("New Predictors: ", new_predictors)
    # Remove Leakages
    data = data.dropna()
    return data, model, new_predictors
    

def Predict(train, test, predictors, model):
    print("[+] Predicting stock direction...")
    model.fit(train[predictors], train["Target"])
    preds = model.predict_proba(test[predictors])[:, 1]
    preds[preds >= .6] = 1
    preds[preds < .6] = 0
    # Returns the probability the stock price will increase or decrease tomorrow
    preds = pd.Series(preds, index=test.index, name="Predictions")
    combined = pd.concat([test["Target"], preds], axis=1)
    return combined

def BackTest(data, model, predictors, start=2500, step=250):
    print("BackTesting time-series data...")
    all_predictions = []
    for x in range(start, data.shape[0], step):
        train = data.iloc[0:x].copy()
        test = data.iloc[x:(x+step)].copy()
        predictions = Predict(train, test, predictors, model)
        all_predictions.append(predictions)
    print("All Predictions: ", all_predictions)
    return pd.concat(all_predictions)



if __name__ == '__main__':
    sp500 = Collect_Data()
    data, model, new_predictors = Train_Model(sp500)
    predictions = BackTest(data, model,new_predictors)

