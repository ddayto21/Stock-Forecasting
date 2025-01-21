import pandas as pd

def backtest(data, model, predictors, start=2500, step=250):
    """
    Backtest the model using a rolling window approach.
    """
    print("[+] Backtesting...")
    all_predictions = []

    for x in range(start, data.shape[0], step):
        train = data.iloc[:x].copy()
        test = data.iloc[x:x+step].copy()
        predictions = predict(train, test, predictors, model)
        combined = pd.concat([test["Target"], predictions], axis=1)
        all_predictions.append(combined)

    return pd.concat(all_predictions)


def predict(train, test, predictors, model, threshold=0.6):
    """
    Predict stock price movement probabilities and classify based on a threshold.
    """
    probabilities = model.predict_proba(test[predictors])[:, 1]
    predictions = (probabilities >= threshold).astype(int)
    return pd.Series(predictions, index=test.index, name="Predictions")