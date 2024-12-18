import praw
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class RedditStockScraper:
    def __init__(self, client_id, client_secret, user_agent):
        """
        Initialize Reddit API connection
        
        :param client_id: Reddit API client ID
        :param client_secret: Reddit API client secret
        :param user_agent: Unique user agent string
        """
        self.reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent
        )
    
    def scrape_stock_discussions(self, subreddits, keywords, days_back=30):
        """
        Scrape stock-related discussions from specified subreddits
        
        :param subreddits: List of subreddit names to scrape
        :param keywords: List of stock-related keywords
        :param days_back: Number of days to scrape back
        :return: DataFrame with scraped data
        """
        all_posts = []
        
        for subreddit_name in subreddits:
            subreddit = self.reddit.subreddit(subreddit_name)
            
            # Calculate timestamp for days_back
            time_filter = int((datetime.now() - timedelta(days=days_back)).timestamp())
            
            for post in subreddit.search(f"({' OR '.join(keywords)})", 
                                         sort='new', 
                                         limit=None, 
                                         time_filter='month'):
                if post.created_utc > time_filter:
                    all_posts.append({
                        'title': post.title,
                        'body': post.selftext,
                        'score': post.score,
                        'num_comments': post.num_comments,
                        'created_utc': post.created_utc,
                        'subreddit': subreddit_name
                    })
        
        return pd.DataFrame(all_posts)
