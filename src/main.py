from data.collect_data import collect_data
from data.preprocess_data import preprocess_data
from model.train import train_model
from model.predict import backtest
from utils.logging import setup_logging

# Evaluate model performance
from sklearn.metrics import precision_score


def main():
    # Set up logging
    logger = setup_logging()
    logger.info("Starting the stock prediction pipeline...")

    # Step 1: Collect data
    sp500 = collect_data(ticker="^GSPC", start_date="1990-01-01")
    logger.info("Data collection complete.")

    # Step 2: Preprocess data
    horizons = [2, 5, 60, 250, 1000]
    sp500, predictors = preprocess_data(sp500, horizons)
    logger.info("Data preprocessing complete.")

    # Step 3: Train model
    model, train, test = train_model(sp500, predictors)
    logger.info("Model training complete.")

    # Step 4: Backtest predictions
    predictions = backtest(sp500, model, predictors)
    logger.info("Backtesting complete.")

    precision = precision_score(predictions["Target"], predictions["Predictions"])
    logger.info(f"Model Precision: {precision:.2%}")


if __name__ == "__main__":
    main()
