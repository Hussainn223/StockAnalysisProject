# Stock Sentiment Analysis and Trading Decision System

This project is a **Stock Sentiment Analysis and Trading Decision System** that uses sentiment analysis to evaluate financial news and predict whether a stock's value will rise or fall. The system fetches stock data using the **Alpaca-py SDK**, analyzes the sentiment of financial news with **FinBERT**, and makes buy or sell recommendations based on sentiment.

## Features
- Fetches stock prices using the Alpaca API.
- Uses FinBERT for sentiment analysis on financial news articles.
- Makes trading decisions (Buy/Sell) based on sentiment scores.

## Requirements
- Python 3.7 or higher
- alpaca-trade-api
- transformers (for FinBERT)
- pandas
- numpy

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Hussainn223/stock-sentiment-analysis.git

