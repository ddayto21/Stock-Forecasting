from sklearn.ensemble import RandomForestClassifier

def train_model(data, predictors):
    """
    Train a RandomForestClassifier on the given data and predictors.
    """
    print("[+] Training the model...")
    model = RandomForestClassifier(n_estimators=200, min_samples_split=50, random_state=1)

    # Split into training and testing sets
    train = data.iloc[:-100]
    test = data.iloc[-100:]

    model.fit(train[predictors], train["Target"])
    return model, train, test