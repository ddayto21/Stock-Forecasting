import pytest
import pandas as pd
from src.data.collect_data import collect_data


@pytest.fixture
def mock_yfinance(mocker):
    """
    Mock the yfinance Ticker.history method to simulate data retrieval.
    """
    mock_ticker = mocker.patch("yfinance.Ticker")
    mock_ticker.return_value.history.return_value = pd.DataFrame(
        {
            "Close": [100, 102, 101, 103, 104],
            "Volume": [1000, 1100, 1200, 1300, 1400],
            "Open": [99, 101, 100, 102, 103],
            "High": [101, 103, 102, 104, 105],
            "Low": [98, 99, 100, 101, 102],
            "Dividends": [0, 0, 0, 0, 0],
            "Stock Splits": [0, 0, 0, 0, 0],
        },
        index=pd.date_range(start="2023-01-01", periods=5),
    )
    return mock_ticker


def test_collect_data(mock_yfinance):
    """
    Test the collect_data function with mocked yfinance data.
    """
    # Call the function
    result = collect_data(ticker="^GSPC", start_date="2023-01-01")

    # Assertions
    assert isinstance(result, pd.DataFrame), "Result should be a DataFrame"
    assert "Close" in result.columns, "DataFrame should contain 'Close' column"
    assert "Target" in result.columns, "DataFrame should contain 'Target' column"
    assert "Tomorrow" in result.columns, "DataFrame should contain 'Tomorrow' column"
    assert "Dividends" not in result.columns, "Irrelevant columns should be removed"
    assert "Stock Splits" not in result.columns, "Irrelevant columns should be removed"

    # Check the target variable
    assert result["Target"].iloc[0] == 1, "Target should correctly indicate price increase"
    assert result["Target"].iloc[2] == 0, "Target should correctly indicate price decrease"

    # Check filtered data range
    assert result.index.min() >= pd.Timestamp("2023-01-01"), "Data should start from the specified date"


def test_empty_data_handling(mocker):
    """
    Test the behavior of collect_data when an empty DataFrame is returned.
    """
    # Mock yfinance to return an empty DataFrame
    mocker.patch("yfinance.Ticker").return_value.history.return_value = pd.DataFrame()

    # Call the function
    result = collect_data(ticker="^GSPC", start_date="2023-01-01")

    # Assertions
    assert result.empty, "Result should be an empty DataFrame when no data is available"