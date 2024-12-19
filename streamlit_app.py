import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
from reddit_stock_scraper import RedditStockScraper
from text_preprocessor import TextPreprocessor
from stock_prediction_model import StockPredictionModel
import os
from dotenv import load_dotenv

# Load environment variables from .env file (local development)
load_dotenv()

# Function to get credentials from environment or secrets
def get_credentials():
    # Check if running on Streamlit Cloud
    try:
        return {
            "client_id": st.secrets["REDDIT_CLIENT_ID"],
            "client_secret": st.secrets["REDDIT_CLIENT_SECRET"],
            "user_agent": st.secrets["REDDIT_USER_AGENT"]
        }
    except (FileNotFoundError, KeyError):
        # Fallback to environment variables for local development
        return {
            "client_id": os.getenv("REDDIT_CLIENT_ID"),
            "client_secret": os.getenv("REDDIT_CLIENT_SECRET"),
            "user_agent": os.getenv("REDDIT_USER_AGENT")
        }

# Get credentials
credentials = get_credentials()
CLIENT_ID = credentials["client_id"]
CLIENT_SECRET = credentials["client_secret"]
USER_AGENT = credentials["user_agent"]

# Validate credentials
if not all([CLIENT_ID, CLIENT_SECRET, USER_AGENT]):
    raise ValueError("Missing Reddit API credentials. Please check your environment variables or Streamlit secrets.")

# Update the STOCK_INDICES dictionary with a comprehensive list
STOCK_INDICES = {
    "US Indices": {
        "S&P 500 (SPY)": {
            "symbol": "SPY",
            "keywords": ["SPY", "S&P 500", "SP500"],
        },
        "Dow Jones (DIA)": {
            "symbol": "DIA",
            "keywords": ["DIA", "Dow Jones", "DJIA"],
        },
        "NASDAQ 100 (QQQ)": {
            "symbol": "QQQ",
            "keywords": ["QQQ", "NASDAQ", "Tech Index"],
        },
        "Russell 2000 (IWM)": {
            "symbol": "IWM",
            "keywords": ["IWM", "Russell 2000", "Small Caps"],
        },
        "S&P 100 (OEF)": {
            "symbol": "OEF",
            "keywords": ["OEF", "S&P 100"],
        },
    },
    "Indian Indices": {
        "NIFTY 50 (^NSEI)": {
            "symbol": "^NSEI",
            "keywords": ["NIFTY", "NIFTY50", "NSE"],
        },
        "BANK NIFTY (^NSEBANK)": {
            "symbol": "^NSEBANK",
            "keywords": ["BANKNIFTY", "Bank NIFTY"],
        },
        "BSE SENSEX (^BSESN)": {
            "symbol": "^BSESN",
            "keywords": ["SENSEX", "BSE"],
        },
        "NIFTY IT (NIFTYIT.NS)": {
            "symbol": "NIFTYIT.NS",
            "keywords": ["NIFTY IT", "IT Index"],
        },
        "NIFTY AUTO (NIFTYAUTO.NS)": {
            "symbol": "NIFTYAUTO.NS",
            "keywords": ["NIFTY Auto", "Auto Index"],
        },
        "NIFTY PHARMA (NIFTYPHARMA.NS)": {
            "symbol": "NIFTYPHARMA.NS",
            "keywords": ["NIFTY Pharma", "Pharma Index"],
        },
        "NIFTY METAL (NIFTYMETAL.NS)": {
            "symbol": "NIFTYMETAL.NS",
            "keywords": ["NIFTY Metal", "Metal Index"],
        },
        "NIFTY FMCG (NIFTYFMCG.NS)": {
            "symbol": "NIFTYFMCG.NS",
            "keywords": ["NIFTY FMCG", "FMCG Index"],
        },
        "NIFTY ENERGY (NIFTYENERGY.NS)": {
            "symbol": "NIFTYENERGY.NS",
            "keywords": ["NIFTY Energy", "Energy Index"],
        },
        "NIFTY REALTY (NIFTYREALTY.NS)": {
            "symbol": "NIFTYREALTY.NS",
            "keywords": ["NIFTY Realty", "Real Estate Index"],
        }
    },
    "European Indices": {
        "FTSE 100 (^FTSE)": {
            "symbol": "^FTSE",
            "keywords": ["FTSE", "UK Index"],
        },
        "DAX (^GDAXI)": {
            "symbol": "^GDAXI",
            "keywords": ["DAX", "German Index"],
        },
        "CAC 40 (^FCHI)": {
            "symbol": "^FCHI",
            "keywords": ["CAC", "French Index"],
        },
    },
    "Asian Indices": {
        "Nikkei 225 (^N225)": {
            "symbol": "^N225",
            "keywords": ["Nikkei", "Japanese Index"],
        },
        "Hang Seng (^HSI)": {
            "symbol": "^HSI",
            "keywords": ["HSI", "Hong Kong Index"],
        },
        "Shanghai Composite (000001.SS)": {
            "symbol": "000001.SS",
            "keywords": ["SSEC", "Shanghai Index"],
        },
    }
}

def create_candlestick_chart(df, sentiment_data=None):
    """Create a candlestick chart"""
    fig = go.Figure()
    
    # Candlestick chart only
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='Price'
    ))
    
    fig.update_layout(
        title='Price History',
        yaxis_title='Price',
        xaxis_title='Date',
        height=600
    )
    return fig

def get_sentiment_summary(sentiment_df, market_df):
    """Generate a summary of sentiment analysis"""
    recent_sentiment = sentiment_df['sentiment'].mean()
    sentiment_ma5 = sentiment_df['sentiment'].rolling(window=5).mean().iloc[-1]
    sentiment_ma20 = sentiment_df['sentiment'].rolling(window=20).mean().iloc[-1]
    mention_count = len(sentiment_df)
    price_change = market_df['Close'].pct_change().iloc[-1] * 100
    
    return {
        'Recent Sentiment': round(recent_sentiment, 3),
        'Sentiment MA5': round(sentiment_ma5, 3),
        'Sentiment MA20': round(sentiment_ma20, 3),
        'Mention Count': mention_count,
        'Recent Price Change %': round(price_change, 2)
    }

def get_market_summary(summary, sentiment_trend):
    """Generate an interpretative market summary"""
    sentiment_status = "positive" if summary['Recent Sentiment'] > 0 else "negative"
    trend_status = "improving" if sentiment_trend > 0 else "declining"
    
    interpretation = f"""
    ### Market Analysis Summary for {datetime.now().strftime('%Y-%m-%d')}

    **Current Market Sentiment:**
    - Overall sentiment is {sentiment_status} ({summary['Recent Sentiment']:.3f})
    - Short-term sentiment (5-day MA: {summary['Sentiment MA5']:.3f}) is {trend_status} compared to
    - Long-term sentiment (20-day MA: {summary['Sentiment MA20']:.3f})
    
    **Activity Metrics:**
    - Found {summary['Mention Count']} relevant discussions
    - Recent price change: {summary['Recent Price Change %']}%
    
    **Interpretation:**
    1. Market sentiment is {sentiment_status} overall
    2. Short-term trend is {trend_status}
    3. Discussion volume is {'high' if summary['Mention Count'] > 100 else 'moderate' if summary['Mention Count'] > 50 else 'low'}
    4. Price action {'supports' if (summary['Recent Price Change %'] > 0) == (summary['Recent Sentiment'] > 0) else 'contradicts'} sentiment
    """
    return interpretation

def predict_next_day_price(market_data, sentiment_score):
    """
    Predict next day's closing price using technical indicators and sentiment
    """
    # Calculate technical indicators
    data = market_data.copy()
    
    # Technical features
    data['SMA_5'] = data['Close'].rolling(window=5).mean()
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['RSI'] = data['Close'].pct_change().rolling(window=14).apply(
        lambda x: 100 - (100 / (1 + (x[x > 0].mean() / -x[x < 0].mean())))
    )
    data['Volatility'] = data['Close'].pct_change().rolling(window=20).std()
    
    # Get latest values
    last_close = data['Close'].iloc[-1]
    sma_5 = data['SMA_5'].iloc[-1]
    sma_20 = data['SMA_20'].iloc[-1]
    rsi = data['RSI'].iloc[-1]
    volatility = data['Volatility'].iloc[-1]
    
    # Calculate momentum
    momentum = (sma_5 - sma_20) / sma_20
    
    # Combine technical signals
    technical_signal = 0
    technical_signal += 1 if sma_5 > sma_20 else -1  # Trend
    technical_signal += 1 if rsi > 50 else -1        # Momentum
    technical_signal += 1 if momentum > 0 else -1    # Moving average trend
    
    # Normalize technical signal to [-1, 1]
    technical_signal = technical_signal / 3
    
    # Combine technical and sentiment signals (70-30 weight)
    combined_signal = (0.7 * technical_signal) + (0.3 * sentiment_score)
    
    # Predict price change
    predicted_change = combined_signal * volatility
    predicted_price = last_close * (1 + predicted_change)
    
    return {
        'current_price': last_close,
        'predicted_price': predicted_price,
        'predicted_change': predicted_change * 100,
        'technical_confidence': abs(technical_signal) * 100,
        'sentiment_impact': sentiment_score * 100
    }

def main():
    st.set_page_config(page_title="Global Market Sentiment Analyzer", layout="wide")
    st.title("Global Market Sentiment Analysis Dashboard")

    # Sidebar configurations
    st.sidebar.header("Select Market Index")
    
    # Two-level selection for better organization
    category = st.sidebar.selectbox(
        "Select Category",
        list(STOCK_INDICES.keys())
    )
    
    selected_index = st.sidebar.selectbox(
        "Choose Index",
        list(STOCK_INDICES[category].keys())
    )
    
    days_back = st.sidebar.slider("Days of Historical Data", 30, 180, 60)

    if st.sidebar.button("Analyze"):
        try:
            # Initialize components
            scraper = RedditStockScraper(CLIENT_ID, CLIENT_SECRET, USER_AGENT)
            preprocessor = TextPreprocessor()
            predictor = StockPredictionModel()

            # Get selected index details
            index_details = STOCK_INDICES[category][selected_index]
            symbol = index_details["symbol"]
            
            # Set dates
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)

            # Create progress tracking elements
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Step 1: Market Data (20%)
            status_text.text("📊 Step 1/5: Fetching market data from Yahoo Finance...")
            market_data = predictor.get_stock_data(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date
            )
            progress_bar.progress(20)
            
            # Step 2: Reddit Data (40%)
            status_text.text("🤖 Step 2/5: Scraping Reddit discussions...")
            subreddits = ['stocks', 'investing', 'wallstreetbets']
            raw_data = scraper.scrape_stock_discussions(
                subreddits, 
                index_details["keywords"]
            )
            progress_bar.progress(40)

            # Step 3: Text Processing (60%)
            status_text.text("🔍 Step 3/5: Processing and cleaning text data...")
            raw_data['cleaned_text'] = raw_data['body'].apply(preprocessor.clean_text)
            progress_bar.progress(60)

            # Step 4: Sentiment Analysis (80%)
            status_text.text("💭 Step 4/5: Analyzing market sentiment...")
            raw_data['sentiment'] = raw_data['cleaned_text'].apply(preprocessor.get_sentiment)
            raw_data['date'] = pd.to_datetime(raw_data['created_utc'], unit='s')
            raw_data.set_index('date', inplace=True)
            progress_bar.progress(80)

            # Step 5: Visualization (100%)
            status_text.text("📈 Step 5/5: Preparing visualizations...")
            fig = create_candlestick_chart(market_data, raw_data)
            progress_bar.progress(100)
            
            # Clear progress indicators
            status_text.empty()
            progress_bar.empty()

            # Display results
            st.plotly_chart(fig)

            # Display sentiment summary
            st.header("Sentiment Analysis Summary")
            summary = get_sentiment_summary(raw_data, market_data)
            
            # Create three columns for metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Current Sentiment", f"{summary['Recent Sentiment']:.3f}")
                st.metric("5-Day Sentiment MA", f"{summary['Sentiment MA5']:.3f}")
            
            with col2:
                st.metric("20-Day Sentiment MA", f"{summary['Sentiment MA20']:.3f}")
                st.metric("Mention Count", summary['Mention Count'])
            
            with col3:
                st.metric("Recent Price Change", f"{summary['Recent Price Change %']}%")
            
            # Prediction
            sentiment_trend = summary['Sentiment MA5'] - summary['Sentiment MA20']
            prediction = "Bullish 📈" if sentiment_trend > 0 else "Bearish 📉"
            confidence = abs(sentiment_trend) * 100
            
            st.header("Market Prediction")
            st.subheader(f"Prediction: {prediction}")
            
            # Fix for progress bar
            normalized_confidence = min(abs(sentiment_trend), 1.0)  # Ensure value is between 0 and 1
            st.progress(normalized_confidence)
            st.caption(f"Confidence: {min(confidence, 100):.1f}%")
            
            # Add market summary
            st.header("Market Summary")
            st.markdown(get_market_summary(summary, sentiment_trend))

            # Display recent relevant posts
            st.header("Recent Relevant Discussions")
            recent_posts = raw_data.sort_index(ascending=False).head(3)
            if len(recent_posts) > 0:
                for idx, row in recent_posts.iterrows():
                    with st.expander(f"{row['title']} (Sentiment: {row['sentiment']:.2f})"):
                        st.write(f"Subreddit: r/{row['subreddit']}")
                        st.write(f"Score: {row['score']} | Comments: {row['num_comments']}")
                        st.write(row['body'][:500] + "..." if len(row['body']) > 500 else row['body'])
            else:
                st.write("No recent discussions found.")

            # Price Prediction
            st.header("Next Day Price Prediction")
            
            price_prediction = predict_next_day_price(
                market_data, 
                summary['Recent Sentiment']
            )
            
            # Create columns for prediction display
            pred_col1, pred_col2 = st.columns(2)
            
            with pred_col1:
                st.metric(
                    "Current Price", 
                    f"${price_prediction['current_price']:.2f}",
                    f"{price_prediction['predicted_change']:.2f}%"
                )
                st.caption("Predicted % change for tomorrow")
            
            with pred_col2:
                st.metric(
                    "Predicted Price", 
                    f"${price_prediction['predicted_price']:.2f}"
                )
                st.caption("Expected price for tomorrow")
            
            # Prediction details
            st.subheader("Prediction Analysis")
            st.write(f"""
            - Technical Analysis Confidence: {price_prediction['technical_confidence']:.1f}%
            - Sentiment Impact: {price_prediction['sentiment_impact']:.1f}%
            - Prediction combines:
                * 70% weight on technical indicators (SMA, RSI, Momentum)
                * 30% weight on market sentiment
            """)
            
            # Add prediction disclaimer
            st.warning("""
            **Disclaimer**: This prediction is based on historical data and sentiment analysis. 
            Market movements are subject to many external factors and may not follow predicted patterns. 
            This should not be considered as financial advice.
            """)

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.error("Full error details:", exc_info=True)  # Added for better error tracking

if __name__ == "__main__":
    main()