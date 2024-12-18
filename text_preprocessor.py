import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np

class TextPreprocessor:
    def __init__(self):
        """
        Initialize text preprocessing utilities with FinBERT
        """
        try:
            nltk.download('punkt')
            nltk.download('stopwords')
            self.stop_words = set(stopwords.words('english'))
            
            # Initialize FinBERT
            self.tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
            self.model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
            self.model.eval()  # Set to evaluation mode
            
        except Exception as e:
            print(f"Error initializing models: {str(e)}")
    
    def clean_text(self, text):
        """
        Clean and preprocess text data
        """
        if not isinstance(text, str):
            text = str(text)
            
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters but keep important financial symbols
        text = re.sub(r'[^a-zA-Z0-9$%.\s]', '', text)
        
        # Tokenize
        try:
            tokens = word_tokenize(text)
        except Exception as e:
            print(f"Tokenization error: {str(e)}")
            return text
        
        # Remove stopwords
        tokens = [token for token in tokens if token not in self.stop_words]
        
        return ' '.join(tokens)
    
    def get_sentiment(self, text):
        """
        Perform sentiment analysis using FinBERT
        Returns: sentiment score between -1 and 1
        """
        if not isinstance(text, str):
            text = str(text)
        
        try:
            # Tokenize text for FinBERT
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            
            # Get prediction
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            # FinBERT outputs: [negative, neutral, positive]
            scores = predictions.numpy()[0]
            
            # Convert to a single score between -1 and 1
            # Negative: -1, Neutral: 0, Positive: 1, weighted by probabilities
            sentiment_score = (-1 * scores[0] + 1 * scores[2])
            
            return float(sentiment_score)
            
        except Exception as e:
            print(f"Error in sentiment analysis: {str(e)}")
            return 0.0
