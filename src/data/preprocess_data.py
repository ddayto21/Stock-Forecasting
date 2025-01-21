def preprocess_data(data, horizons):
    """
    Add rolling averages and trend features to the data for prediction.
    """
    print("[+] Adding time-series features...")

    if data.empty:
        return data, []  # Return empty predictors for empty input

    predictors = []

    for horizon in horizons:
        rolling_averages = data["Close"].rolling(horizon).mean()
        ratio_column = f"Close_Ratio_{horizon}"
        trend_column = f"Trend_{horizon}"
        data[ratio_column] = data["Close"] / rolling_averages
        data[trend_column] = data["Target"].shift(1).rolling(horizon).sum()
        predictors.extend([ratio_column, trend_column])

    # Drop rows with NaN values introduced by rolling operations
    data.dropna(inplace=True)
    return data, predictors
