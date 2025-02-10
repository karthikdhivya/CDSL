# create stremlit app
import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np


st.title("Stock analysis app")

stock_name=st.text_input("Enter stock name","CDSL.NS")

if st.button("Predict"):
    data=yf.download(stock_name,start="2020-01-01",end="2025-02-10")
    st.write("stock data:",data.tail())


    st.subheader("stock closing price")
    fig, ax=plt.subplots()
    ax.plot(data["Close"],label="Close Price")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    st.pyplot(fig)


    data["Prediction"]=data["Close"].shift(-5)
    x=np.array(data.drop(["Prediction"],axis=1))[:-5]
    y=np.array(data["Prediction"])[:-5]


    model=LinearRegression()
    model.fit(x,y)


    future_prediction=model.predict([x[-1]])[0]
    st.write(f"predicted stock price: {future_prediction:.2f}")