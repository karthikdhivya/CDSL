# Load the stock data from yfinance
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# Download cdsl stock history

stock_name="CDSL.NS"
data=yf.download(stock_name,start="2020-01-01",end="2025-02-10")


print(data.head())


#plot closing price

plt.figure(figsize=(15,5))
plt.plot(data['Close'],label="Close Prize")
plt.title(f"{stock_name}closing price")
plt.xlabel("Date")
plt.ylabel("Prize")
plt.legend()
plt.show


#Train a model 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np

data["Prediction"]=data["Close"].shift(-5)


#Remove last 5 rows
x=np.array(data.drop(["Prediction"],axis=1))[:-5]
y=np.array(data["Prediction"])[:-5]


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)


model=LinearRegression()
model.fit(x_train,y_train)


accuracy=model.score(x_test,y_test)

print(f"Model accuracy:{accuracy * 100:.2f}%")



