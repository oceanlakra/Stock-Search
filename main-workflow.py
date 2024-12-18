import os
from reddit_stock_scraper import RedditStockScraper
from text_preprocessor import TextPreprocessor
from stock_prediction_model import StockPredictionModel
from datetime import datetime, timedelta
import pandas as pd

def main():
    # Reddit API credentials (replace with your own)
    CLIENT_ID = "1BhQqDXxVbYM0oXt-q3zvA"
    CLIENT_SECRET = "4bPdbo4vtLRDcmyrO4k8-5GzKnNddA"
    USER_AGENT = "python:com.example.stock:v1.0 (by /u/Dangerous_Horse1773)"

    # Initialize components
    scraper = RedditStockScraper(CLIENT_ID, CLIENT_SECRET, USER_AGENT)
    preprocessor = TextPreprocessor()
    predictor = StockPredictionModel()

    # Define parameters
    symbol = 'SPY'  # S&P 500 ETF
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)  # One year of data
    
    # Get market data
    market_data = predictor.get_stock_data(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date
    )
    
    # Scrape Reddit data
    subreddits = ['stocks', 'investing', 'wallstreetbets']
    keywords = ['SPY', 'S&P 500', 'market', symbol]
    raw_data = scraper.scrape_stock_discussions(subreddits, keywords)
    
    # Process sentiment
    raw_data['cleaned_text'] = raw_data['body'].apply(preprocessor.clean_text)
    raw_data['sentiment'] = raw_data['cleaned_text'].apply(preprocessor.get_sentiment)
    
    # Convert timestamp to datetime index
    raw_data['date'] = pd.to_datetime(raw_data['created_utc'], unit='s')
    raw_data.set_index('date', inplace=True)
    
    # Prepare features and train model
    X, y = predictor.prepare_features(raw_data, market_data)
    model_results = predictor.train_model(X, y)
    
    print(f"Model Performance for {symbol}:")
    print(model_results['classification_report'])

if __name__ == '__main__':
    main()
