import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock
from src.model.predict import backtest, predict


@pytest.fixture
def sample_data():
    """Fixture to create a sample dataset for testing."""
    data = {
        "Close": [100, 102, 101, 103, 104, 105, 106, 108, 109, 110],
        "Target": [1, 0, 1, 0, 1, 1, 0, 1, 0, 1],
        "Predictor1": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        "Predictor2": [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
    }
    return pd.DataFrame(data)


@pytest.fixture
def mock_model():
    """Fixture to create a mock model."""
    mock = MagicMock()

    def mock_predict_proba(X):
        """Generate mock probabilities dynamically based on input length."""
        n = len(X)
        probabilities = [[0.4, 0.6] if i % 2 == 0 else [0.6, 0.4] for i in range(n)]
        return np.array(probabilities)

    mock.predict_proba.side_effect = mock_predict_proba
    return mock


def test_predict(sample_data, mock_model):
    """
    Test the predict function to ensure it correctly classifies probabilities.
    """
    train = sample_data.iloc[:7]
    test = sample_data.iloc[7:]
    predictors = ["Predictor1", "Predictor2"]
    threshold = 0.6

    # Call predict
    predictions = predict(train, test, predictors, mock_model, threshold)

    # Assertions
    assert isinstance(predictions, pd.Series), "Predictions should be a pandas Series"
    assert len(predictions) == len(test), "Predictions length should match the test set"
    assert predictions.tolist() == [1, 0, 1], "Predictions are incorrect"

    # Ensure predict_proba was called with the correct data
    called_args = mock_model.predict_proba.call_args[0][0]
    pd.testing.assert_frame_equal(
        called_args, test[predictors], check_dtype=False, obj="Mock model input"
    )


def backtest(data, model, predictors, start=2500, step=250):
    """
    Backtest the model using a rolling window approach.
    """
    print("[+] Backtesting...")
    all_predictions = []

    for x in range(
        start, data.shape[0] - step + 1, step
    ):  # Ensure correct number of windows
        train = data.iloc[:x].copy()
        test = data.iloc[x : x + step].copy()
        if test.empty:
            break
        predictions = predict(train, test, predictors, model)
        combined = pd.concat([test["Target"], predictions], axis=1)
        all_predictions.append(combined)

    return pd.concat(all_predictions)
