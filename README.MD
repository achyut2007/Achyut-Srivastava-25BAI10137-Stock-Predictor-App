# StockVision: AI-Powered Stock Trend Predictor 📈

## Overview
**StockVision** is a real-time web application that analyzes stock market trends using Machine Learning. It fetches live historical data for any stock ticker and applies a **Linear Regression** model to visualize the long-term price trend.

This project was built to demonstrate the practical application of Python libraries (`Streamlit`, `Scikit-Learn`, `Pandas`) in solving financial data problems.

## 🚀 Features
* **Live Data Fetching:** Uses the Yahoo Finance API (`yfinance`) to get up-to-the-minute stock data.
* **Interactive Dashboard:** Built with Streamlit for a clean, web-based user interface.
* **ML Trend Analysis:** implements Linear Regression to calculate and plot the "Best Fit Line" for stock prices.
* **Dynamic Visualization:** Interactive charts using Matplotlib/Streamlit to compare actual prices vs. predicted trends.
* **Indian Market Support:** Fully supports NSE/BSE tickers (e.g., `RELIANCE.NS`, `TATASTEEL.NS`).

## 🛠️ Tech Stack
* **Language:** Python 3.x
* **Frontend:** Streamlit
* **Machine Learning:** Scikit-Learn (Linear Regression)
* **Data Handling:** Pandas, NumPy
* **Data Source:** yfinance API

## ⚙️ Installation & Setup

### 1. Prerequisites
Ensure you have Python installed on your system.

### 2. Install Dependencies
Open your terminal or command prompt and run the following command to install the required libraries:

```bash
pip install streamlit yfinance scikit-learn pandas matplotlib