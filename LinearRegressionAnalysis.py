import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, TheilSenRegressor, RANSACRegressor
from pandas._libs.tslibs.offsets import BDay
import time
from scipy import stats
import matplotlib.pyplot as plt

import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

#First thing is to design the regression set up


def reset_my_index(df):
    res = df[::-1].reset_index(drop=True)
    return (res)

def DowJones():
    path = 'Dow_Jones_1896.csv'
    df = pd.read_csv(path)
    df.Date = pd.to_datetime(df.Date)
    df = df[["Date", "Close"]]
    df.rename(columns={'Close': 'close'}, inplace=True)
    isBusinessDay = BDay().is_on_offset
    match_series = pd.to_datetime(df['Date']).map(isBusinessDay)
    df = df[match_series]
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def getData():
    CryptoData = []
    Variables = ["BNBUSDT", "ADAUSDT", "BTCUSDT", "ETHUSDT",
                 "NEOUSDT", "QTUMUSDT", "XRPUSDT", "LTCUSDT"]
    for x in Variables:
        path = 'CryptoMinuteData/Binance_' + x + "_1h.csv"
        data = pd.read_csv(path)
        #data = reset_my_index(data)
        CryptoData.append(data)

    return CryptoData, Variables

def SlopeVec(length):
    X = np.arange(1, length, 1)
    Denominator = 0
    Numerator = []

    for i in X:
        Denominator += abs((i - np.mean(X)))**2
        #Denominator += abs(1)
        Numerator.append(i - np.mean(X))

    Numerator = Numerator/Denominator
    print(Numerator)
    return Numerator

def SubPlots(Data):
    fig = px.line(x=Data.index, y=Data['close'])
    fig.update_xaxes(rangeslider_visible=True)
    fig.show()
    fig1 = px.line(x=Data.index, y=Data['Slope'])
    fig1.update_xaxes(rangeslider_visible=True)
    fig1.show()
    fig2 = px.line(x=Data.index, y=Data['Residual'])
    fig2.update_xaxes(rangeslider_visible=True)
    fig2.show()

#Function to check that the variances can differ a lot despite having the same mean?
#Demonstrates why log values are extremely important
def Check():
    #Prices1 = np.array([2.5, 5, 1.5, 5])
    #Prices2 = np.array([2.5, 1000, 996.5, 5])
    Prices1 = np.log(np.array([2.5, 5, 1.5, 5]))
    Prices2 = np.log(np.array([2.5, 1000, 996.5, 5]))
    Time = np.array([1,2,3,4]).reshape(-1,1)
    Reg1 = LinearRegression()
    Reg1.fit(Time, Prices1)
    Reg2 = LinearRegression()
    Reg2.fit(Time, Prices2)
    #Slope does in fact measure returns and can measure it properly
    print(Reg1.coef_)
    print(Reg2.coef_)
    print(Reg1.intercept_)
    print(Reg2.intercept_)

#Given A time period We generate a random slope, through random numbers.
#We then have an algorithm which tweaks the values to shows the change in
#variance of the values. What matters the most is

def SlopeAnalysis():
    Crypto = DowJones()
    i = 0
    Window = 20
    j = Window
    Time = np.arange(start=1, stop = Window+1, step = 1)
    start = time.process_time()

    pct_change = Crypto['close'].pct_change(periods=Window)
    Slope = np.empty(len(Crypto['close']))
    Var = np.empty((len(Crypto['close'])))
    Kurtosis = np.empty((len(Crypto['close'])))
    Skewness = np.empty(len(Crypto['close']))
    Slope[:] = np.NAN
    Kurtosis[:] = np.NAN
    Skewness[:] = np.NAN
    X = SlopeVec(Window+1)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=Time, y=X,
                             mode='lines+markers',
                             name='Weighted Time'))

    fig.show()
    #regr = LinearRegression()
    Crypto['close'] = np.log(Crypto['close'])
    log_diff = Crypto['close'].diff(periods=Window)
    start = time.process_time()
    while j != len(Crypto):
        Data = np.array(Crypto['close'][i:j])
        Slope[j] = np.dot(X, Data)
        Var[j] = np.var(Data)
        Kurtosis[j] = stats.kurtosis(Data, bias=True)
        Skewness[j] = stats.skew(Data, bias=True)
        i += 1
        j += 1

    print(time.process_time() - start)

    Crypto['Slope'] = Slope
    Crypto['Var'] = Var
    Crypto['Kurtosis'] = Kurtosis
    Crypto['Skewness'] = Skewness

    fig1 = px.line(Crypto, x=Crypto.index, y='Slope', title='Slope Mapped Against Time')
    fig1.update_xaxes(rangeslider_visible=True)
    fig1.show()

    fig1 = px.scatter(Crypto, x='Var', y='Slope', title='Scatter plot of Slope and Var')
    fig1.show()

    fig = px.density_contour(Crypto, x="Var", y="Slope", marginal_x="histogram", marginal_y="histogram")
    fig.show()

    fig1 = px.scatter(Crypto, x='Kurtosis', y='Slope', title='Scatter plot of Slope and Kurtosis', marginal_x="histogram", marginal_y="histogram")
    fig1.show()

    fig = px.density_contour(Crypto, x="Kurtosis", y="Slope", marginal_x="histogram", marginal_y="histogram")
    fig.show()

    fig1 = px.scatter(Crypto, x='Skewness', y='Slope', title='Scatter plot of Slope and Skewness', marginal_x="histogram", marginal_y="histogram")
    fig1.show()

    fig = px.density_contour(Crypto, x="Skewness", y="Slope", marginal_x="histogram", marginal_y="histogram")
    fig.show()

    fig = go.Figure(data=[go.Histogram(x=Crypto['Slope'], histnorm='probability')])
    fig.update_layout(title="Probability Distribution of the Slope", xaxis_title="Slope Values", yaxis_title="Probability Values")
    fig.show()

    fig = go.Figure(data=[go.Histogram(x=Crypto['Kurtosis'], histnorm='probability')])
    fig.update_layout(title="Probability Distribution of the Kurtosis", xaxis_title="Kurtosis Values", yaxis_title="Probability Values")
    fig.show()

    fig = go.Figure(data=[go.Histogram(x=Crypto['Skewness'], histnorm='probability')])
    fig.update_layout(title="Probability Distribution of the Skewness", xaxis_title="Skewness Values", yaxis_title="Probability Values")
    fig.show()

    fig = go.Figure(data=[go.Histogram(x=pct_change, histnorm='probability')])
    fig.update_layout(title="Probability Distribution of pct_changes", xaxis_title="pct_change", yaxis_title="Probability Values")
    fig.show()

    fig = go.Figure(data=[go.Histogram(x=log_diff, histnorm='probability')])
    fig.update_layout(title="Probability Distribution of log_diff", xaxis_title="log_diff", yaxis_title="Probability Values")
    fig.show()

    fig = go.Figure(data=[go.Histogram(x=Var, histnorm='probability')])
    fig.update_layout(title="Probability Distribution of Variance", xaxis_title="Variance", yaxis_title="Probability Values")
    fig.show()

    print(f"Window Length = {Window} - Model : Linear Regression")

    print(f"Mean of Slope Values = {np.mean(Crypto['Slope'])}")
    SD = np.sqrt(np.var(Crypto['Slope']))
    print(f"Mean - SD = {np.mean(Crypto['Slope']) - SD} and Mean + SD = {np.mean(Crypto['Slope']) + SD}")
    print(f"Variance of Slope Values = {np.var(Crypto['Slope'])}")

    print(f"pct_change mean = {np.mean(pct_change)}")
    SD = np.sqrt(np.var(pct_change))
    print(f"Mean - SD = {np.mean(pct_change) - SD} and Mean + SD = {np.mean(pct_change) + SD}")
    print(f"pct_change var = {np.var(pct_change)}")

    print(f"log_diff mean = {np.mean(log_diff)}")
    SD = np.sqrt(np.var(log_diff))
    print(f"Mean - SD = {np.mean(log_diff) - SD} and Mean + SD = {np.mean(log_diff) + SD}")
    print(f"log_diff var = {np.var(log_diff)}")

    print(f"Variance mean = {np.mean(Crypto['Var'])}")
    SD = np.sqrt(np.var(Crypto['Var']))
    print(f"Mean + SD = {np.mean(Crypto['Var']) + SD}")
    print(f"Variance var = {np.var(Crypto['Var'])}")

    #Now we look at the rate of change of Slope Value
    #Rate of change of Variance Values

    Crypto['Slope_Change'] = Crypto['Slope'].diff(periods=1)
    Crypto['Var_Change'] = Crypto['Var'].diff(periods=1)

    print(f"Slope_Change mean = {np.mean(Crypto['Slope_Change'])}")
    SD = np.sqrt(np.var(Crypto['Slope_Change']))
    print(f"Mean - SD = {np.mean(Crypto['Slope_Change']) - SD} and Mean + SD = {np.mean(Crypto['Slope_Change']) + SD}")
    print(f"Slope_Change var = {np.var(Crypto['Slope_Change'])}")

    Crypto = Crypto.dropna()

    print(f"Var_Change mean = {np.mean(Crypto['Var_Change'])}")
    SD = np.sqrt(np.var(Crypto['Var_Change']))
    print(f"Mean - SD = {np.mean(Crypto['Var_Change']) - SD} and Mean + SD = {np.mean(Crypto['Var_Change']) + SD}")
    print(f"Var_Change var = {np.var(Crypto['Var_Change'])}")

    fig1 = px.scatter(Crypto, x='Var_Change', y='Slope_Change', title='Scatter plot of Slope Rates and Var Rates', marginal_x="histogram", marginal_y="histogram")
    fig1.show()

    fig = go.Figure(data=[go.Histogram(x=Crypto['Slope_Change'], histnorm='probability')])
    fig.update_layout(title="Probability Distribution of Change of Slope", xaxis_title="Change of Slope Values", yaxis_title="Probability Values")
    fig.show()

    fig = go.Figure(data=[go.Histogram(x=Crypto['Var_Change'], histnorm='probability')])
    fig.update_layout(title="Probability Distribution of Change of Variance", xaxis_title="Change of Variance Values", yaxis_title="Probability Values")
    fig.show()

def TimePlotsWithPriceGraph(df, Name):
    fig = make_subplots(
        rows = 2, cols=1,
        subplot_titles=('Close Prices', Name),
        shared_xaxes = True,
        vertical_spacing=0.1
    )

    fig.add_trace(go.Scatter(x=df.index, y=df['close'], name='Asset Price'), row=1, col=1)
    fig.update_layout(
        autosize=False,
        width=1000,
        height=500,
    )
    fig.add_trace(go.Scatter(x=df.index, y=df[Name], name=Name + 'Analysis'), row=2, col=1)
    fig.update_layout(legend_orientation="h",
                      xaxis2_rangeslider_visible=True)

    fig.show()

def ResidualAnalysis():
    Crypto = DowJones()
    i = 0
    Window = 20
    j = Window
    Time = np.arange(start=1, stop = Window+1, step = 1).reshape(-1,1)
    RSS = np.empty(len(Crypto['close']))
    RS = np.empty(len(Crypto['close']))
    Crypto['close'] = np.log(Crypto['close'])

    regr = LinearRegression()
    RSSMean = np.empty(len(Crypto['close']))
    RSMean = np.empty(len(Crypto['close']))
    RSSVar = np.empty(len(Crypto['close']))
    RSVar = np.empty(len(Crypto['close']))
    RSS[:] = np.NAN
    RS[:] = np.NAN
    RSSMean[:] = np.NAN
    RSMean[:] = np.NAN
    RSVar[:] = np.NAN
    RSSVar[:] = np.NAN


    while j != len(Crypto):
        Data = np.array(Crypto['close'][i:j])
        regr.fit(Time, Data)
        Pred = regr.predict(Time)
        X = (Data-Pred)**2
        X1 = Data - Pred
        RSS[j] = sum(X)
        RS[j] = sum(X1)
        RSSMean[j] = np.mean(X)
        RSSVar[j] = np.var(X)
        RSMean[j] = np.mean(X1)
        RSVar[j] = np.var(X1)
        #Residual[j] = np.dot(X, Data)
        i += 1
        j += 1

    Crypto['RSS'] = RSS
    Crypto['RS'] = RS
    Crypto['RSSMean'] = RSSMean
    Crypto['RSSVar'] = RSSVar
    Crypto['RSMean'] = RSMean
    Crypto['RSVar'] = RSVar

    Crypto['close'] = np.exp(Crypto['close'])

    Lists = ['RSS', 'RS', 'RSSMean', 'RSSVar', 'RSMean', 'RSVar']
    for i in Lists:
        TimePlotsWithPriceGraph(Crypto, i)


if __name__ == "__main__":
    #SlopeAnalysis()
    ResidualAnalysis()