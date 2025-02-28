import streamlit as st
import pandas as pd
import requests
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
import time
import os

# Load FMP API Key from Environment Variables
FMP_API_KEY = st.secrets["api"]["FMP_API_KEY"]

if not FMP_API_KEY:
    st.error("FMP API key not found. Set the 'FMP_API_KEY' environment variable.")
    st.stop()

# Initialize Session State
if 'tickers' not in st.session_state:
    st.session_state.tickers = []
    
# Sidebar User Guide with Key Concepts
st.sidebar.title("User Guide")
st.sidebar.markdown("""
**How to Use the Tool**:
1. **Add Tickers**: Click "Add Ticker" to input securities (stocks, stock funds, options funds, or fixed income assets).
2. **Fetch Data**: Select asset type, enter the ticker, and click "Fetch Data" to retrieve details.
3. **Set Shares**: Input your owned shares for each ticker in the "Shares Owned" field.
4. **Set Cash Balance**: Adjust the "Cash Balance" to reflect funds available for reinvestment.
5. **Calculate Allocations**: Click "Calculate Allocations" for reinvestment recommendations based on attractiveness scores.
6. **Run Forecast**: Click "Run Forecast," set the horizon (years), and view portfolio performance scenarios.

**Field Definitions**:
- **Shares Owned**: Current number of shares owned, used for portfolio value and income calculations.
- **Manual Yield (%)**: Expected annual yield for options funds (default: 65%) or fixed income (default: 3.72%).
- **Manual Expense Ratio (%)**: Annual expense ratio for options funds (default: 0.99%) or fixed income (default: 0.03%).
- **Current Price**: Latest market price.
- **Attractiveness Score**: Normalized score (0-1) for reinvestment potential.
- **Trailing 12-Month Dividend Yield**: Annual historical dividend yield for stocks/stock funds.
- **Trailing 12-Month Distribution Yield**: Annual historical distribution yield for options funds.
- **Trailing 12-Month Yield**: Annual historical yield for fixed income assets.
- **1-Year Historical NAV Erosion**: Percentage decline in options fund NAV from its 1-year peak.
- **Trailing 12-Month P/E Ratio**: Price-to-earnings ratio for stocks/stock funds.
- **5-Year Historical Price Growth (CAGR)**: Compounded annual growth rate over 5 years, adjusted 80% conservatively.
- **30-Day Forecasted Price Change (ARIMA)**: Short-term price trend prediction.

**Key Concepts**:
- **Attractiveness Score**: Combines yield, valuation, payout ratios, and ARIMA forecasts (15% weight).
- **Allocation Logic**: Cash distributed by attractiveness scores, totaling available cash (e.g., $1,000).
- **NAV Erosion**: Reduces options funds’ attractiveness if significant decay.
- **Price Growth**: 5-year CAGR adjusted 80% conservatively.
- **Forecast Scenarios**:
  - **Dynamic Reinvestment**: Monthly dividends reinvested by attractiveness scores.
  - **Proportional Reinvestment**: Cash allocated by current holdings, no rebalancing.
  - **No Reinvestment**: Dividends accrue as cash, assets grow by price appreciation.
""")

# Enhanced CSS for Sticky Header
st.markdown("""
    <style>
    .main-content {
        margin-top: 0; /* Reduced from 25px to move title up, adjusted by inline -25px */
        width: 80%;
        margin-left: auto;
        margin-right: auto;
    }
    @media (max-width: 768px) {
        .stDataFrame, .stPlotlyChart {
            width: 100%;
            overflow-x: auto;
        }
    }
    .sticky-controls {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        background-color: white;
        z-index: 10000;
        padding: 5px 20px;
        border-bottom: 1px solid #ccc;
        display: flex;
        flex-direction: row;
        justify-content: center;
        align-items: center;
        gap: 10px;
    }
    .sticky-controls [data-testid="stNumberInput"] {
        max-width: 150px;
    }
    </style>
""", unsafe_allow_html=True)

# Fetch Data Functions Using FMP
def fetch_stock_data(ticker, asset_type):
    """Fetch stock data using FMP with sanity checks."""
    retries = 3
    delay = 2
    for attempt in range(retries):
        try:
            # Fetch real-time price
            url = f"https://financialmodelingprep.com/api/v3/quote/{ticker}?apikey={FMP_API_KEY}"
            response = requests.get(url)
            if response.status_code != 200:
                raise ValueError(f"FMP API error: {response.status_code}")
            data = response.json()
            if not data:
                raise ValueError("No data returned.")
            price = float(data[0].get('price', 0)) or 0

            # Fetch dividend yield and P/E ratio from key metrics
            url = f"https://financialmodelingprep.com/api/v3/key-metrics-ttm/{ticker}?apikey={FMP_API_KEY}"
            response = requests.get(url)
            if response.status_code != 200:
                raise ValueError(f"FMP API error: {response.status_code}")
            metrics = response.json()
            dividend_yield = float(metrics[0].get('dividendYieldTTM', 0) or 0) * 100 if metrics else 0
            pe_ratio = float(metrics[0].get('peRatioTTM', 20) or 20) if metrics else 20

            # Fetch historical data for price growth
            url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{ticker}?apikey={FMP_API_KEY}"
            response = requests.get(url)
            if response.status_code != 200:
                raise ValueError(f"FMP API error: {response.status_code}")
            hist_data = response.json().get('historical', [])
            if not hist_data:
                raise ValueError("No historical data.")
            hist_df = pd.DataFrame(hist_data).sort_values('date').tail(1260)  # ~5 years
            if len(hist_df) < 2:
                price_growth = 4
            else:
                start_price = float(hist_df['close'].iloc[0])
                end_price = float(hist_df['close'].iloc[-1])
                price_growth = ((end_price / start_price) ** (1/5) - 1) * 100 * 0.8

            # ARIMA forecast (30 days)
            forecast_data, error = forecast_trend(hist_df, steps=30)
            forecast_change = forecast_data[0] if forecast_data else 0

            # Sanity checks
            if dividend_yield < 0:
                dividend_yield = 0
            if price <= 0:
                price = 1

            return {
                'type': asset_type,
                'ticker': ticker,
                'dividend_yield': dividend_yield,
                'pe_ratio': pe_ratio,
                'price': price,
                'price_growth': price_growth,
                'div_growth': 0,  # Placeholder
                'forecast_change': forecast_change,
                'hist': hist_df,
                'forecast': forecast_data[1] if forecast_data else [],
                'forecast_dates': forecast_data[2] if forecast_data else []
            }
        except Exception as e:
            if attempt < retries - 1:
                st.warning(f"Attempt {attempt + 1} failed for {ticker}: {str(e)}. Retrying...")
                time.sleep(delay)
            else:
                st.error(f"Error fetching {ticker}: {str(e)}")
                return None

def fetch_options_fund_data(ticker, asset_type, manual_yield=65, manual_expense_ratio=0.99):
    """Fetch options fund data with editable yield."""
    retries = 3
    delay = 2
    for attempt in range(retries):
        try:
            url = f"https://financialmodelingprep.com/api/v3/quote/{ticker}?apikey={FMP_API_KEY}"
            response = requests.get(url)
            if response.status_code != 200:
                raise ValueError(f"FMP API error: {response.status_code}")
            data = response.json()
            if not data:
                raise ValueError("No data returned.")
            price = float(data[0].get('price', 0)) or 1

            url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{ticker}?apikey={FMP_API_KEY}"
            response = requests.get(url)
            if response.status_code != 200:
                raise ValueError(f"FMP API error: {response.status_code}")
            hist_data = response.json().get('historical', [])
            if not hist_data:
                raise ValueError("No historical data.")
            hist_df = pd.DataFrame(hist_data).sort_values('date').tail(252)  # ~1 year
            if len(hist_df) < 2:
                nav_erosion = 0
                price_growth = 4
            else:
                max_price = float(hist_df['close'].max())
                nav_erosion = ((max_price - price) / max_price) * 100 if max_price > 0 else 0
                start_price = float(hist_df['close'].iloc[0])
                end_price = float(hist_df['close'].iloc[-1])
                price_growth = ((end_price / start_price) ** (1/5) - 1) * 100 * 0.8

            forecast_data, error = forecast_trend(hist_df, steps=30)
            forecast_change = forecast_data[0] if forecast_data else 0

            if price <= 0:
                price = 1

            return {
                'type': asset_type,
                'ticker': ticker,
                'dist_yield': manual_yield,
                'expense_ratio': manual_expense_ratio,
                'price': price,
                'price_growth': price_growth,
                'nav_erosion': nav_erosion,
                'forecast_change': forecast_change,
                'hist': hist_df,
                'forecast': forecast_data[1] if forecast_data else [],
                'forecast_dates': forecast_data[2] if forecast_data else []
            }
        except Exception as e:
            if attempt < retries - 1:
                st.warning(f"Attempt {attempt + 1} failed for {ticker}: {str(e)}. Retrying...")
                time.sleep(delay)
            else:
                st.error(f"Error fetching {ticker}: {str(e)}")
                return None

def fetch_fixed_income_data(ticker, asset_type, manual_yield=3.72, manual_expense_ratio=0.03):
    """Fetch fixed income data with updated yield logic."""
    retries = 3
    delay = 2
    for attempt in range(retries):
        try:
            url = f"https://financialmodelingprep.com/api/v3/quote/{ticker}?apikey={FMP_API_KEY}"
            response = requests.get(url)
            if response.status_code != 200:
                raise ValueError(f"FMP API error: {response.status_code}")
            data = response.json()
            if not data:
                raise ValueError("No data returned.")
            price = float(data[0].get('price', 0)) or 100

            # Attempt to fetch yield from FMP's ETF profile or key metrics
            url = f"https://financialmodelingprep.com/api/v3/etf-holder/{ticker}?apikey={FMP_API_KEY}"
            response = requests.get(url)
            if response.status_code == 200:
                etf_data = response.json()
                yield_value = float(etf_data.get('yield', manual_yield) or manual_yield) * 100 if etf_data else manual_yield
            else:
                yield_value = manual_yield

            url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{ticker}?apikey={FMP_API_KEY}"
            response = requests.get(url)
            if response.status_code != 200:
                raise ValueError(f"FMP API error: {response.status_code}")
            hist_data = response.json().get('historical', [])
            if not hist_data:
                raise ValueError("No historical data.")
            hist_df = pd.DataFrame(hist_data).sort_values('date').tail(1260)  # ~5 years
            if len(hist_df) < 2:
                price_growth = 4
            else:
                start_price = float(hist_df['close'].iloc[0])
                end_price = float(hist_df['close'].iloc[-1])
                price_growth = ((end_price / start_price) ** (1/5) - 1) * 100 * 0.8

            forecast_data, error = forecast_trend(hist_df, steps=30)
            forecast_change = forecast_data[0] if forecast_data else 0

            if price <= 0:
                price = 100

            return {
                'type': asset_type,
                'ticker': ticker,
                'yield': yield_value,
                'expense_ratio': manual_expense_ratio,
                'price': price,
                'price_growth': price_growth,
                'forecast_change': forecast_change,
                'hist': hist_df,
                'forecast': forecast_data[1] if forecast_data else [],
                'forecast_dates': forecast_data[2] if forecast_data else []
            }
        except Exception as e:
            if attempt < retries - 1:
                st.warning(f"Attempt {attempt + 1} failed for {ticker}: {str(e)}. Retrying...")
                time.sleep(delay)
            else:
                st.error(f"Error fetching {ticker}: {str(e)}")
                return None

def forecast_trend(hist_df, steps=30):
    """Generate a 30-day ARIMA forecast."""
    try:
        close_prices = hist_df['close']
        if len(close_prices) < 100:
            return None, "Not enough data for forecast."
        close_prices.index = pd.to_datetime(hist_df['date'])
        model = ARIMA(close_prices, order=(5, 1, 0))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=steps)
        predicted_change = (forecast.iloc[-1] - close_prices.iloc[-1]) / close_prices.iloc[-1] * 100
        future_dates = pd.date_range(start=hist_df['date'].max(), periods=steps + 1, freq='D')[1:]
        return (predicted_change, forecast, future_dates), None
    except Exception as e:
        return None, f"Forecasting failed: {str(e)}"

# Main App Logic
st.markdown(
    '<div style="width: 80%; margin-left: auto; margin-right: auto; text-align: center; padding: 10px 0; position: relative; top: -25px;">'
    '<h1 style="color: #ffffff; font-size: 30px; margin: 0; font-weight: bold;">Dividend Reinvestment Tool</h1></div>',
    unsafe_allow_html=True
)
st.markdown('<div class="sticky-controls">', unsafe_allow_html=True)

st.number_input(
    "Cash Balance ($)",
    min_value=0.0,
    value=1000.0,
    step=100.0,
    key="cash_balance_input"
)

if st.button("Add Ticker", key="add_ticker_btn"):  # Changed key to avoid conflict
    st.session_state.tickers.append({'type': None, 'ticker': None, 'data': None, 'shares': 0.0})

col1, col2 = st.columns(2)
with col1:
    if st.button("Calculate Allocations", key="calc_allocations_btn"):  # Changed key
        st.session_state.calculate_allocations = True
with col2:
    if st.button("Run Forecast", key="run_forecast_btn"):  # Changed key
        st.session_state.run_forecast = True

st.markdown('</div>', unsafe_allow_html=True)  # Close sticky-controls
st.markdown('<div class="main-content" style="margin-top: 25px;">', unsafe_allow_html=True)  # Ensure 25px gap


for i, ticker_info in enumerate(st.session_state.tickers):
    with st.expander(f"Ticker {i + 1}", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            asset_type = st.selectbox("Asset Type", ["Stock", "Stock Fund", "Options Fund", "Fixed Income"], key=f"asset_type_{i}")
        with col2:
            ticker_symbol = st.text_input("Ticker Symbol", key=f"ticker_symbol_{i}")
        if st.button("Fetch Data", key=f"fetch_{i}"):
            if asset_type in ["Stock", "Stock Fund"]:
                data = fetch_stock_data(ticker_symbol, asset_type)
            elif asset_type == "Options Fund":
                manual_yield = st.number_input("Manual Yield (%)", min_value=0.0, value=65.0, key=f"yield_{i}")
                manual_expense_ratio = st.number_input("Manual Expense Ratio (%)", min_value=0.0, value=0.99, key=f"expense_{i}")
                data = fetch_options_fund_data(ticker_symbol, asset_type, manual_yield, manual_expense_ratio)
            elif asset_type == "Fixed Income":
                manual_yield = st.number_input("Manual Yield (%)", min_value=0.0, value=3.72, key=f"yield_{i}")
                manual_expense_ratio = st.number_input("Manual Expense Ratio (%)", min_value=0.0, value=0.03, key=f"expense_{i}")
                data = fetch_fixed_income_data(ticker_symbol, asset_type, manual_yield, manual_expense_ratio)
            if data:
                st.session_state.tickers[i] = {'type': asset_type, 'ticker': ticker_symbol, 'data': data, 'shares': float(ticker_info['shares'])}
                st.rerun()
        if ticker_info['data']:
            data = ticker_info['data']
            st.session_state.tickers[i]['shares'] = float(st.number_input("Shares Owned", min_value=0.0, value=float(ticker_info['shares']), key=f"shares_{i}"))
            # Editable yield for options funds
            if data['type'] == "Options Fund":
                data['dist_yield'] = st.number_input("Distribution Yield (%)", min_value=0.0, value=float(data['dist_yield']), key=f"yield_edit_{i}")
            # Display all relevant data
            if data['type'] in ["Stock", "Stock Fund"]:
                details_df = pd.DataFrame({
                    "Metric": ["Current Price ($)", "Dividend Yield (%)", "P/E Ratio", "5-Year Price Growth (%)", "30-Day Forecast Change (%)"],
                    "Value": [data['price'], data['dividend_yield'], data['pe_ratio'], data['price_growth'], data['forecast_change']]
                })
            elif data['type'] == "Options Fund":
                details_df = pd.DataFrame({
                    "Metric": ["Current Price ($)", "Distribution Yield (%)", "Expense Ratio (%)", "NAV Erosion (%)", "5-Year Price Growth (%)", "30-Day Forecast Change (%)"],
                    "Value": [data['price'], data['dist_yield'], data['expense_ratio'], data['nav_erosion'], data['price_growth'], data['forecast_change']]
                })
            elif data['type'] == "Fixed Income":
                details_df = pd.DataFrame({
                    "Metric": ["Current Price ($)", "Yield (%)", "Expense Ratio (%)", "5-Year Price Growth (%)", "30-Day Forecast Change (%)"],
                    "Value": [data['price'], data['yield'], data['expense_ratio'], data['price_growth'], data['forecast_change']]
                })
            st.write("**Asset Details**")
            st.dataframe(details_df.style.format({"Value": "{:.2f}"}))
            # Matplotlib chart with 30-day forecast
            if not data['hist'].empty and len(data['forecast']) > 0:
                fig, ax = plt.subplots(figsize=(6, 3))
                hist_dates = pd.to_datetime(data['hist']['date']).tail(252)
                hist_prices = data['hist']['close'].tail(252)
                ax.plot(hist_dates, hist_prices, label="History", color="blue", linestyle="-")
                ax.plot(data['forecast_dates'], data['forecast'], label="Forecast (30 days)", color="red", linestyle="--")
                ax.set_xlabel("Date")
                ax.set_ylabel("Price ($)")
                ax.set_title(f"{ticker_symbol} Price History & 30-Day Forecast")
                ax.legend()
                ax.grid(True)
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)

if 'calculate_allocations' in st.session_state and st.session_state.calculate_allocations:
    attractiveness_scores = []
    for ticker_info in st.session_state.tickers:
        if ticker_info['data']:
            data = ticker_info['data']
            if data['type'] in ["Stock", "Stock Fund"]:
                yield_score = min(data['dividend_yield'] / 5, 1)
                valuation_score = min(20 / max(data['pe_ratio'], 0.01), 1)  # Avoid division by zero
                payout_score = min(data['dividend_yield'] / (data['price_growth'] + data['div_growth'] + 0.01), 1)
                forecast_score = min(max(data['forecast_change'] / 30, -1), 1) * 0.15  # Adjusted for 30 days
                score = yield_score * 0.35 + valuation_score * 0.35 + payout_score * 0.15 + forecast_score
            elif data['type'] == "Options Fund":
                yield_score = min(data['dist_yield'] / 65, 1)
                erosion_score = max(1 - data['nav_erosion'] / 20, 0)
                forecast_score = min(max(data['forecast_change'] / 30, -1), 1) * 0.15
                score = yield_score * 0.5 + erosion_score * 0.35 + forecast_score
            elif data['type'] == "Fixed Income":
                yield_score = min(data['yield'] / 5, 1)
                forecast_score = min(max(data['forecast_change'] / 30, -1), 1) * 0.15
                score = yield_score * 0.85 + forecast_score
            attractiveness_scores.append(score)
        else:
            attractiveness_scores.append(0)
    total_score = sum(attractiveness_scores)
    normalized_scores = [score / total_score if total_score > 0 else 0 for score in attractiveness_scores]
    allocations = [st.session_state.cash_balance_input * score for score in normalized_scores] if total_score > 0 else [0] * len(st.session_state.tickers)
    allocation_df = pd.DataFrame({
        "Ticker": [t['ticker'] for t in st.session_state.tickers],
        "Asset Type": [t['type'] for t in st.session_state.tickers],
        "Attractiveness Score": normalized_scores,
        "Allocation ($)": allocations,
        "Shares to Buy": [alloc / t['data']['price'] if t['data'] and t['data']['price'] > 0 else 0 for alloc, t in zip(allocations, st.session_state.tickers)]
    })
    st.write("### Recommended Allocations")
    st.dataframe(allocation_df.style.format({"Allocation ($)": "{:.2f}", "Shares to Buy": "{:.2f}", "Attractiveness Score": "{:.4f}"}))

    # Written Rationale
    st.write("#### Allocation Rationale")
    for i, score in enumerate(normalized_scores):
        ticker = st.session_state.tickers[i]['ticker']
        data = st.session_state.tickers[i]['data']
        if data:
            if data['type'] in ["Stock", "Stock Fund"]:
                yield_desc = "high" if data['dividend_yield'] > 2.5 else "low"
                val_desc = "attractive" if data['pe_ratio'] < 20 else "high"
                growth_desc = "strong" if data['price_growth'] > 5 else "modest"
                rationale = f"{ticker}: This asset has a {yield_desc} dividend yield, a {val_desc} valuation, and {growth_desc} historical growth, contributing to its score."
            elif data['type'] == "Options Fund":
                yield_desc = "high" if data['dist_yield'] > 32.5 else "low"
                erosion_desc = "low" if data['nav_erosion'] < 10 else "significant"
                rationale = f"{ticker}: This options fund offers a {yield_desc} distribution yield with {erosion_desc} NAV erosion, driving its attractiveness."
            elif data['type'] == "Fixed Income":
                yield_desc = "high" if data['yield'] > 2.5 else "low"
                rationale = f"{ticker}: This fixed income asset provides a {yield_desc} yield, making it appealing for income."
            st.write(rationale)

if 'run_forecast' in st.session_state and st.session_state.run_forecast:
    years = st.slider("Forecast Horizon (Years)", 1, 10, 5)
    months = years * 12
    dates = [datetime.now() + timedelta(days=30 * i) for i in range(months + 1)]
    # Initialize portfolios for each strategy
    dynamic_portfolio = [{'ticker': t['ticker'], 'shares': t['shares'], 'data': dict(t['data'])} for t in st.session_state.tickers if t['data']]
    proportional_portfolio = [{'ticker': t['ticker'], 'shares': t['shares'], 'data': dict(t['data'])} for t in st.session_state.tickers if t['data']]
    no_reinvestment_portfolio = [{'ticker': t['ticker'], 'shares': t['shares'], 'data': dict(t['data'])} for t in st.session_state.tickers if t['data']]
    dynamic_values = [st.session_state.cash_balance_input + sum(a['data']['price'] * a['shares'] for a in dynamic_portfolio)]
    proportional_values = [st.session_state.cash_balance_input + sum(a['data']['price'] * a['shares'] for a in proportional_portfolio)]
    no_reinvestment_values = [st.session_state.cash_balance_input + sum(a['data']['price'] * a['shares'] for a in no_reinvestment_portfolio)]
    dynamic_income, proportional_income, no_reinvestment_income = [], [], []
    
    for month in range(months):
        # Calculate monthly income and update prices
        dynamic_income_month, proportional_income_month, no_reinvestment_income_month = 0, 0, 0
        for d_asset, p_asset, n_asset in zip(dynamic_portfolio, proportional_portfolio, no_reinvestment_portfolio):
            d_data, p_data, n_data = d_asset['data'], p_asset['data'], n_asset['data']
            if d_data['type'] in ["Stock", "Stock Fund"]:
                monthly_dividend = d_data['price'] * (d_data['dividend_yield'] / 100) / 12
                growth_factor = (1 + (d_data['price_growth'] / 100)) ** (1/12)
            elif d_data['type'] == "Options Fund":
                monthly_dividend = d_data['price'] * (d_data['dist_yield'] / 100) / 12
                growth_factor = (1 + (d_data['price_growth'] / 100)) ** (1/12)
            elif d_data['type'] == "Fixed Income":
                monthly_dividend = d_data['price'] * (d_data['yield'] / 100) / 12
                growth_factor = (1 + (d_data['price_growth'] / 100)) ** (1/12)
            d_data['price'] *= growth_factor
            p_data['price'] *= growth_factor
            n_data['price'] *= growth_factor
            dynamic_income_month += monthly_dividend * d_asset['shares']
            proportional_income_month += monthly_dividend * p_asset['shares']
            no_reinvestment_income_month += monthly_dividend * n_asset['shares']
        dynamic_income.append(dynamic_income_month)
        proportional_income.append(proportional_income_month)
        no_reinvestment_income.append(no_reinvestment_income_month)
        
        # Dynamic Reinvestment
        attractiveness_scores = []
        for asset in dynamic_portfolio:
            data = asset['data']
            if data['type'] in ["Stock", "Stock Fund"]:
                yield_score = min(data['dividend_yield'] / 5, 1)
                valuation_score = min(20 / max(data['pe_ratio'], 0.01), 1)
                payout_score = min(data['dividend_yield'] / (data['price_growth'] + data['div_growth'] + 0.01), 1)
                forecast_score = min(max(data['forecast_change'] / 30, -1), 1) * 0.15
                score = yield_score * 0.35 + valuation_score * 0.35 + payout_score * 0.15 + forecast_score
            elif data['type'] == "Options Fund":
                yield_score = min(data['dist_yield'] / 65, 1)
                erosion_score = max(1 - data['nav_erosion'] / 20, 0)
                forecast_score = min(max(data['forecast_change'] / 30, -1), 1) * 0.15
                score = yield_score * 0.5 + erosion_score * 0.35 + forecast_score
            elif data['type'] == "Fixed Income":
                yield_score = min(data['yield'] / 5, 1)
                forecast_score = min(max(data['forecast_change'] / 30, -1), 1) * 0.15
                score = yield_score * 0.85 + forecast_score
            attractiveness_scores.append(score)
        total_score = sum(attractiveness_scores)
        if total_score > 0:
            for i, asset in enumerate(dynamic_portfolio):
                allocation = dynamic_income_month * (attractiveness_scores[i] / total_score)
                shares_to_buy = allocation / asset['data']['price'] if asset['data']['price'] > 0 else 0
                asset['shares'] += shares_to_buy
        
        # Proportional Reinvestment
        total_portfolio_value = sum(a['data']['price'] * a['shares'] for a in proportional_portfolio if a['data']['price'] > 0)
        if total_portfolio_value > 0:
            for asset in proportional_portfolio:
                allocation = proportional_income_month * (asset['shares'] * asset['data']['price'] / total_portfolio_value)
                shares_to_buy = allocation / asset['data']['price'] if asset['data']['price'] > 0 else 0
                asset['shares'] += shares_to_buy
        
        # Update portfolio values
        dynamic_value = sum(a['data']['price'] * a['shares'] for a in dynamic_portfolio)
        proportional_value = sum(a['data']['price'] * a['shares'] for a in proportional_portfolio)
        no_reinvestment_value = sum(a['data']['price'] * a['shares'] for a in no_reinvestment_portfolio) + sum(no_reinvestment_income)
        dynamic_values.append(dynamic_value)
        proportional_values.append(proportional_value)
        no_reinvestment_values.append(no_reinvestment_value)
    
    # Total Return Chart
    fig, ax = plt.subplots()
    ax.plot(dates, dynamic_values, label="Dynamic", color="green")
    ax.plot(dates, proportional_values, label="Proportional", color="blue")
    ax.plot(dates, no_reinvestment_values, label="No Reinvestment", color="red")
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value ($)")
    ax.set_title("Total Return Forecast")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)
    
    # Income Forecast
    income_df = pd.DataFrame({
        "Month": [d.strftime("%Y-%m") for d in dates[1:]],
        "Dynamic": dynamic_income,
        "Proportional": proportional_income,
        "No Reinvestment": no_reinvestment_income
    })
    st.write("### Income Forecast")
    st.dataframe(income_df.style.format({"Dynamic": "{:.2f}", "Proportional": "{:.2f}", "No Reinvestment": "{:.2f}"}))

st.markdown('</div>', unsafe_allow_html=True)