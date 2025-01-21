import pytest
import pandas as pd
from src.data.preprocess_data import preprocess_data


@pytest.fixture
def sample_data():
    """Fixture to create a sample DataFrame for testing."""
    data = {
        "Close": [100, 102, 101, 103, 104, 105],
        "Target": [1, 0, 1, 0, 1, 1],
    }
    return pd.DataFrame(data)


def test_preprocess_data(sample_data):
    """
    Test that preprocess_data correctly adds rolling averages and trend features.
    """
    horizons = [2, 3]
    processed_data, predictors = preprocess_data(sample_data.copy(), horizons)

    # Assertions
    assert not processed_data.empty, "Processed data should not be empty"
    assert len(predictors) == 2 * len(
        horizons
    ), "Predictors list should contain 2 features per horizon"

    # Check if rolling average and trend columns are added
    for horizon in horizons:
        ratio_col = f"Close_Ratio_{horizon}"
        trend_col = f"Trend_{horizon}"
        assert (
            ratio_col in processed_data.columns
        ), f"{ratio_col} should be in the processed DataFrame"
        assert (
            trend_col in processed_data.columns
        ), f"{trend_col} should be in the processed DataFrame"

    # Ensure no NaN values remain
    assert (
        processed_data.isnull().sum().sum() == 0
    ), "Processed data should not contain NaN values"


def test_preprocess_data_edge_case_empty_df():
    """
    Test preprocess_data with an empty DataFrame.
    """
    empty_data = pd.DataFrame(columns=["Close", "Target"])
    horizons = [2, 3]
    processed_data, predictors = preprocess_data(empty_data, horizons)

    # Assertions
    assert processed_data.empty, "Processed data should remain empty for an empty input"
    assert len(predictors) == 0, "No predictors should be generated for an empty input"


def test_preprocess_data_single_row():
    """
    Test preprocess_data with a single-row DataFrame.
    """
    single_row_data = pd.DataFrame({"Close": [100], "Target": [1]})
    horizons = [2, 3]
    processed_data, predictors = preprocess_data(single_row_data.copy(), horizons)

    # Assertions
    assert processed_data.empty, "Processed data should be empty for insufficient rows"
    assert len(predictors) == 2 * len(
        horizons
    ), "Predictors list should still contain valid feature names"


def test_preprocess_data_correct_calculations(sample_data):
    """
    Test that preprocess_data correctly calculates rolling averages and trends.
    """
    horizons = [2]
    processed_data, _ = preprocess_data(sample_data.copy(), horizons)

    # Check calculations for Close_Ratio_2 and Trend_2
    ratio_col = "Close_Ratio_2"
    trend_col = "Trend_2"

    # Processed data starts after dropping NaN rows
    processed_start_index = processed_data.index[0]

    # Calculate expected values using the first two rows in processed data
    rolling_average = sample_data["Close"].iloc[processed_start_index - 1:processed_start_index + 1].mean()
    expected_ratio = sample_data["Close"].iloc[processed_start_index] / rolling_average

    # Trend calculation excludes the current row due to .shift(1)
    shifted_target = sample_data["Target"].shift(1)
    expected_trend = shifted_target.iloc[processed_start_index - 1:processed_start_index + 1].sum()

    assert processed_data[ratio_col].iloc[0] == pytest.approx(
        expected_ratio, 0.01
    ), f"{ratio_col} calculation is incorrect"
    assert processed_data[trend_col].iloc[0] == expected_trend, f"{trend_col} calculation is incorrect"
