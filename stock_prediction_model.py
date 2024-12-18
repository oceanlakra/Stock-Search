import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import yfinance as yf

class SentimentLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super(SentimentLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out

class StockDataset(Dataset):
    def __init__(self, features, targets):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

class StockPredictionModel:
    def __init__(self, sequence_length=10, hidden_size=128, num_layers=3):
        """
        Initialize stock prediction model with LSTM
        
        :param sequence_length: Number of time steps to look back
        :param hidden_size: Number of features in hidden state
        :param num_layers: Number of LSTM layers
        """
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.scaler = RobustScaler()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def get_stock_data(self, symbol='SPY', start_date=None, end_date=None):
        """
        Fetch stock market data
        """
        stock = yf.Ticker(symbol)
        df = stock.history(start=start_date, end=end_date)
        
        # Calculate technical indicators
        df['Returns'] = df['Close'].pct_change()
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['RSI'] = self.calculate_rsi(df['Close'])
        df['MACD'] = self.calculate_macd(df['Close'])
        df['Volatility'] = df['Returns'].rolling(window=20).std()
        
        return df
    
    def prepare_features(self, sentiment_df, market_df):
        """
        Prepare features combining sentiment and market data
        """
        # Resample sentiment data to daily frequency
        daily_sentiment = sentiment_df.resample('D').agg({
            'sentiment_score': 'mean',
            'score': 'sum',
            'num_comments': 'sum'
        }).fillna(0)
        
        # Merge market data with sentiment data
        combined_df = pd.merge(
            market_df, 
            daily_sentiment, 
            left_index=True, 
            right_index=True, 
            how='left'
        ).fillna(0)
        
        # Create features
        combined_df['sentiment_score'] = combined_df['sentiment_score'].fillna(0)
        combined_df['log_volume'] = np.log1p(combined_df['Volume'])
        combined_df['sentiment_strength'] = combined_df['sentiment_score'].abs()
        combined_df['price_momentum'] = combined_df['Returns'].rolling(5).mean()
        combined_df['sentiment_ma'] = combined_df['sentiment_score'].rolling(5).mean()
        combined_df['sentiment_std'] = combined_df['sentiment_score'].rolling(5).std()
        
        # Create target variable (1 if next day's return is positive)
        combined_df['target'] = np.where(combined_df['Returns'].shift(-1) > 0, 1, 0)
        
        # Select final features
        features = [
            'Returns', 'log_volume', 'sentiment_score', 'sentiment_strength',
            'sentiment_ma', 'RSI', 'MACD', 'Volatility', 'price_momentum',
            'sentiment_std'
        ]
        
        return combined_df[features], combined_df['target']
    
    @staticmethod
    def calculate_rsi(prices, period=14):
        """Calculate RSI technical indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def calculate_macd(prices, fast=12, slow=26):
        """Calculate MACD technical indicator"""
        exp1 = prices.ewm(span=fast).mean()
        exp2 = prices.ewm(span=slow).mean()
        return exp1 - exp2
    
    def create_sequences(self, data, targets):
        """
        Create sequences for LSTM input
        
        :param data: DataFrame with scaled features
        :param targets: Series with target values
        :return: numpy arrays of sequences and targets
        """
        sequences = []
        target_values = []
        
        for i in range(len(data) - self.sequence_length):
            # Get sequence of features
            seq = data[i:i + self.sequence_length].values
            # Get target value
            target = targets.iloc[i + self.sequence_length]
            sequences.append(seq)
            target_values.append(target)
            
        return np.array(sequences), np.array(target_values)
    
    def train_model(self, df, epochs=50, batch_size=32, learning_rate=0.001):
        """
        Train the LSTM model
        """
        # Select features
        features = ['sentiment_score', 'log_score', 'comment_engagement', 
                   'sentiment_strength', 'sentiment_ma', 'sentiment_std']
        
        # Prepare features and target
        X = df[features].fillna(0)
        y = df['price_movement']  # Target variable
        
        # Scale features
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X),
            columns=features
        )
        
        # Create sequences with separate feature and target data
        X_seq, y_seq = self.create_sequences(X_scaled, y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)
        
        # Create datasets and dataloaders
        train_dataset = StockDataset(X_train, y_train)
        test_dataset = StockDataset(X_test, y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        
        # Initialize model
        self.model = SentimentLSTM(
            input_size=len(features),
            hidden_size=self.hidden_size,
            num_layers=self.num_layers
        ).to(self.device)
        
        # Loss and optimizer
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Training loop
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                # Forward pass
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y.unsqueeze(1))
                
                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.4f}')
        
        # Evaluation
        self.model.eval()
        y_pred = []
        y_true = []
        
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X = batch_X.to(self.device)
                outputs = self.model(batch_X)
                predicted = (outputs.cpu().numpy() > 0.5).astype(int)
                y_pred.extend(predicted)
                y_true.extend(batch_y.numpy())
        
        return {
            'classification_report': classification_report(y_true, y_pred),
            'confusion_matrix': confusion_matrix(y_true, y_pred)
        }
