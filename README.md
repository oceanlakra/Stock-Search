# Stocker - A Stock Market Sentiment Analyzer

A sophisticated Streamlit dashboard that combines real-time market data analysis with Reddit sentiment to provide market insights and predictions for global stock indices.

## Overview

This project leverages Natural Language Processing (NLP) and Machine Learning to analyze market sentiment from Reddit discussions, combining it with technical analysis to generate market insights. It uses FinBERT for sentiment analysis and LSTM networks for price predictions.

### Key Features

- 📊 Real-time market data visualization with interactive candlestick charts
- 🤖 Sentiment analysis of Reddit posts using FinBERT
- 📈 Technical indicators (SMA, RSI, MACD)
- 🔮 Next-day price predictions using hybrid LSTM model
- 🌐 Support for multiple market indices:
  - US Markets (S&P 500, Dow Jones, NASDAQ)
  - European Markets (FTSE, DAX, CAC 40)
  - Asian Markets (Nikkei, Hang Seng)
  - Indian Markets (NIFTY 50, BANK NIFTY)

> **Note**: The analysis for Indian indices currently has limited functionality due to insufficient India-focused discussions on Reddit. We are working on incorporating additional data sources to improve coverage of Indian markets.

## Installation

### Prerequisites

- Python 3.8+
- Reddit API credentials
- Git

### Setup Steps

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/stock-market-sentiment-analyzer.git
   cd stock-market-sentiment-analyzer
   ```

2. **Create Virtual Environment** (Recommended)
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # Linux/Mac
   source venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Reddit API**
   
   Create a `.env` file in the project root:
   ```env
   REDDIT_CLIENT_ID=your_client_id
   REDDIT_CLIENT_SECRET=your_client_secret
   REDDIT_USER_AGENT=your_user_agent
   ```

   To get Reddit API credentials:
   1. Go to https://www.reddit.com/prefs/apps
   2. Click "Create App" or "Create Another App"
   3. Select "script"
   4. Fill in required information
   5. Copy the generated credentials

5. **Run the Application**
   ```bash
   streamlit run streamlit_app.py
   ```
