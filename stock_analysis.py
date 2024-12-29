import os
import alpaca_trade_api as tradeapi
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import requests

# Retrieve API keys from environment variables
api_key = os.getenv("ALPACA_API_KEY")
api_secret = os.getenv("ALPACA_SECRET_KEY")
news_api_key = os.getenv("NEWS_API_KEY")  # News API Key

# Initialize the Alpaca API client
api = tradeapi.REST(api_key, api_secret, base_url='https://paper-api.alpaca.markets')

# Function to fetch the latest stock price
def get_stock_price(symbol):
    try:
        bar = api.get_latest_bar(symbol)
        return bar.close
    except Exception as e:
        print(f"Error fetching stock price for {symbol}: {e}")
        return None

# Function to fetch financial news from News API
def get_financial_news(symbol):
    url = f"https://newsapi.org/v2/everything?q={symbol}&apiKey={news_api_key}"
    try:
        response = requests.get(url)
        response.raise_for_status()  # Will raise an exception for 4xx or 5xx errors
        articles = response.json().get('articles', [])
        news_list = []
        for article in articles:
            news_list.append({
                'title': article['title'],
                'summary': article['description'],
                'url': article['url']
            })
        return news_list
    except requests.exceptions.RequestException as e:
        print(f"Error fetching news for {symbol}: {e}")
        return []

# Function to analyze sentiment using FinBERT
def analyze_sentiment(news_articles):
    # Load pre-trained FinBERT model and tokenizer
    tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert')
    model = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert')

    sentiments = []
    for article in news_articles:
        inputs = tokenizer(article['summary'], return_tensors='pt', truncation=True, padding=True, max_length=512)
        outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=1)
        sentiment = 'positive' if prediction == 2 else 'negative' if prediction == 0 else 'neutral'
        sentiments.append({'title': article['title'], 'sentiment': sentiment, 'url': article['url']})

    return sentiments

# Example usage
if __name__ == "__main__":
    # Replace 'AAPL' with the stock symbol you want to test
    stock_symbol = "AAPL"

    # Fetch and print stock price
    stock_price = get_stock_price(stock_symbol)
    if stock_price:
        print(f"Latest stock price for {stock_symbol}: ${stock_price}")

    # Fetch and print financial news
    news_articles = get_financial_news(stock_symbol)
    if news_articles:
        print(f"\nLatest news for {stock_symbol}:")
        for article in news_articles:
            print(f"- {article['title']}: {article['url']}")

        # Analyze sentiment of the news articles
        sentiment_results = analyze_sentiment(news_articles)
        print(f"\nSentiment analysis results for {stock_symbol}:")
        for result in sentiment_results:
            print(f"- {result['title']} -> Sentiment: {result['sentiment']}")