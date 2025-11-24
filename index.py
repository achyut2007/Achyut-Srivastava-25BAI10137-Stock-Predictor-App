import streamlit as st
import yfinance as yf
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import datetime
st.title("Stock Price Predictor")

# Input
st.sidebar.header("User Options")
stock_id = st.sidebar.text_input("Enter Stock Symbol", "RELIANCE.NS")
start = "2015-01-01"
# Date for today
end = datetime.date.today()
if st.button("Predict Data"):
    # Button to get data and predict the data
    # Getting Data
    st.write("Downloading data")
    df = yf.download(stock_id, start=start, end=end)
#   Error Handling
    if len(df) == 0:
        st.write("Error: Data not found.")
    else:
        # getting gate column
        df = df.reset_index()
        
        # processing dates for linear regression
        df['date_id'] = df['Date'].map(pd.Timestamp.toordinal)
        # Here this function toordinal converts the time and date to simple ordinal number.
        # Remember: this is to get rid of the date wise data and instead have serialized data on the model.
        # x and y of the data for the model
        x = df[['date_id']] # x is the date
        y = df['Close']  # Y is the price of the stock
        
        # creating the model
        lr = LinearRegression()
        lr.fit(x, y)
        
        # prediction of the data
        df['y2'] = lr.predict(x)
        
        # final result
        st.write("Data loaded successfully.")
        
        # shows the values
        currentVal = df['Close'].iloc[-1]
        predVal = df['y2'].iloc[-1]




        
        st.write("Current Price: ", round(currentVal, 2))
        st.write("Predicted Trend Price: ", round(predVal, 2))
        
        # graph to show the data variation
        st.subheader("Graph")
        plt.figure(figsize=(10,5))
        plt.plot(df['Date'], df['Close'], label='Actual')
        plt.plot(df['Date'], df['y2'], label='Trend Line', linestyle='--')
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        
        st.pyplot(plt)
        
        if st.checkbox("Show Data"):
            st.write(df.tail())

        # end of program