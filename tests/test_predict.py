import pytest
import pandas as pd
from predict import collect_data, add_time_series_features, train_model, predict, backtest


@pytest.fixture
def sample_data():
    """Fixture to create a small sample DataFrame for testing."""
    data = {
        "Close": [100, 101, 102, 103, 104, 105],
        "Volume": [1000, 1100, 1200, 1300, 1400, 1500],
        "Open": [99, 100, 101, 102, 103, 104],
        "High": [101, 102, 103, 104, 105, 106],
        "Low": [98, 99, 100, 101, 102, 103],
    }
    df = pd.DataFrame(data)
    df["Tomorrow"] = df["Close"].shift(-1)
    df["Target"] = (df["Tomorrow"] > df["Close"]).astype(int)
    return df.dropna()


def test_collect_data():
    """Test that collect_data returns a DataFrame with the correct columns."""
    sp500 = collect_data()
    assert "Close" in sp500.columns
    assert "Target" in sp500.columns


def test_add_time_series_features(sample_data):
    """Test that time-series features are added correctly."""
    horizons = [2, 3]
    data, predictors = add_time_series_features(sample_data, horizons)
    assert all(f"Close_Ratio_{h}" in data.columns for h in horizons)
    assert all(f"Trend_{h}" in data.columns for h in horizons)
    assert len(predictors) == 4  # Two features per horizon


def test_train_model(sample_data):
    """Test that the model trains without errors."""
    horizons = [2]
    data, predictors = add_time_series_features(sample_data, horizons)
    model, train, test = train_model(data, predictors)
    assert model is not None
    assert not train.empty
    assert not test.empty


def test_predict(sample_data):
    """Test predictions from the trained model."""
    horizons = [2]
    data, predictors = add_time_series_features(sample_data, horizons)
    model, train, test = train_model(data, predictors)
    predictions = predict(train, test, predictors, model)
    assert len(predictions) == len(test)
    assert set(predictions.unique()) <= {0, 1}


def test_backtest(sample_data):
    """Test backtesting functionality."""
    horizons = [2]
    data, predictors = add_time_series_features(sample_data, horizons)
    model, _, _ = train_model(data, predictors)
    results = backtest(data, model, predictors, start=2, step=2)
    assert "Target" in results.columns
    assert "Predictions" in results.columns
    assert len(results) > 0