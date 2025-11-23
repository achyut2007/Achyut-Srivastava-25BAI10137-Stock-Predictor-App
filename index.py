import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Simple title
st.title("Stock Price Predictor")

# Sidebar for inputs
st.sidebar.header("User Options")
stock_id = st.sidebar.text_input("Enter Stock Symbol", "RELIANCE.NS")

# Basic date range
start = "2015-01-01"
# Getting today's date
end = pd.to_datetime("today").strftime("%Y-%m-%d")

if st.button("Predict"):
    # getting data
    st.write("Downloading data...")
    df = yf.download(stock_id, start=start, end=end)
    
    if len(df) == 0:
        st.write("Error: Could not find stock data.")
    else:
        # resetting index to get Date column
        df = df.reset_index()
        
        # processing dates for linear regression
        df['date_id'] = df['Date'].map(pd.Timestamp.toordinal)
        
        # x and y for model
        x = df[['date_id']]
        y = df['Close']
        
        # training
        lr = LinearRegression()
        lr.fit(x, y)
        
        # predicting for the graph
        df['y_pred'] = lr.predict(x)
        
        # results
        st.write("Data loaded successfully.")
        
        # Simple prints, no fancy columns
        current_val = df['Close'].iloc[-1]
        pred_val = df['y_pred'].iloc[-1]
        
        st.write("Current Price: ", round(current_val, 2))
        st.write("Predicted Trend Price: ", round(pred_val, 2))
        
        # Basic matplotlib graph
        st.subheader("Graph")
        plt.figure(figsize=(10,5))
        plt.plot(df['Date'], df['Close'], label='Actual')
        plt.plot(df['Date'], df['y_pred'], label='Trend Line', linestyle='--')
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        
        # Passing the global plt object (typical student style)
        st.pyplot(plt)
        
        if st.checkbox("Show Data"):
            st.write(df.tail())