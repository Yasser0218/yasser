# Import required libraries
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import streamlit as st
from pypfopt import EfficientFrontier, risk_models, expected_returns
from pypfopt import plotting
from pypfopt import objective_functions

# Streamlit App Title
st.title("Portfolio Optimization and Risk Analysis Tool")

# Sidebar for user input
st.sidebar.header("User Input")
tickers = st.sidebar.text_input("Enter stock tickers (comma-separated)", "AAPL,MSFT,GOOG,AMZN,TSLA")
start_date = st.sidebar.text_input("Start Date (YYYY-MM-DD)", "2020-01-01")
end_date = st.sidebar.text_input("End Date (YYYY-MM-DD)", "2023-01-01")

# Fetch data from Yahoo Finance
@st.cache_data
def get_data(tickers, start_date, end_date):
    data = yf.download(tickers, start=start_date, end=end_date)["Close"]
    return data

try:
    tickers = tickers.split(",")
    data = get_data(tickers, start_date, end_date)
    st.write("### Historical Stock Prices")
    st.line_chart(data)

    # Calculate expected returns and covariance matrix
    mu = expected_returns.mean_historical_return(data)
    S = risk_models.sample_cov(data)

    # Portfolio Optimization
    st.write("### Portfolio Optimization")
    ef = EfficientFrontier(mu, S)
    ef.add_objective(objective_functions.L2_reg, gamma=0.1)  # Regularization
    weights = ef.max_sharpe()  # Maximize Sharpe Ratio
    cleaned_weights = ef.clean_weights()
    st.write("#### Optimal Portfolio Weights")
    st.write(pd.Series(cleaned_weights).round(4))

    # Display Performance Metrics
    st.write("#### Portfolio Performance")
    performance = ef.portfolio_performance(verbose=True)
    st.write(f"- Expected Annual Return: {performance[0]*100:.2f}%")
    st.write(f"- Annual Volatility: {performance[1]*100:.2f}%")
    st.write(f"- Sharpe Ratio: {performance[2]:.2f}")

    # Plot Efficient Frontier
    st.write("#### Efficient Frontier")
    fig, ax = plt.subplots()
    ef_plot = EfficientFrontier(mu, S)
    plotting.plot_efficient_frontier(ef_plot, ax=ax, show_assets=True)
    st.pyplot(fig)

    # Risk Analysis: Value at Risk (VaR) and Conditional VaR (CVaR)
    st.write("### Risk Analysis")
    returns = data.pct_change().dropna()
    portfolio_returns = (returns * pd.Series(cleaned_weights)).sum(axis=1)

    # Calculate VaR and CVaR
    confidence_level = 0.95
    var = portfolio_returns.quantile(1 - confidence_level)
    cvar = portfolio_returns[portfolio_returns <= var].mean()

    st.write(f"- Value at Risk (VaR) at {confidence_level*100:.0f}% confidence: {var*100:.2f}%")
    st.write(f"- Conditional Value at Risk (CVaR): {cvar*100:.2f}%")

    # Plot Portfolio Returns Distribution
    st.write("#### Portfolio Returns Distribution")
    fig, ax = plt.subplots()
    portfolio_returns.hist(bins=50, ax=ax, alpha=0.75)
    ax.axvline(var, color="red", linestyle="--", label=f"VaR at {confidence_level*100:.0f}%")
    ax.axvline(cvar, color="green", linestyle="--", label="CVaR")
    ax.legend()
    st.pyplot(fig)

except Exception as e:
    st.error(f"An error occurred: {e}")
