import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import streamlit as st
import datetime
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# Set dark theme for Matplotlib
plt.style.use("dark_background")

# Custom CSS for full dark theme styling
st.markdown(
    """
    <style>
    .reportview-container {
        background-color: #121212;
        color: white;
    }
    .sidebar .sidebar-content {
        background-color: #1e1e1e;
        color: white;
    }
    h1, h2, h3, h4, h5, h6 {
        color: white;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
    }
    .stTextInput>div>div>input {
        background-color: #333;
        color: white;
        border: 1px solid #555;
    }
    .stDataFrame, .stTable {
        background-color: #1e1e1e;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Authentication function
def creds_entered():
    if st.session_state['user'].strip() == "admin" and st.session_state['passwd'].strip() == "admin":
        st.session_state['authenticated'] = True
    else:
        st.session_state['authenticated'] = False
        st.error("❌ Invalid username/password. Please try again!")

# User authentication form
def authenticate_user():
    if "authenticated" not in st.session_state:
        with st.container():
            st.text_input("Username", key="user", on_change=creds_entered, placeholder="Enter username")
            st.text_input("Password", key="passwd", type="password", on_change=creds_entered, placeholder="Enter password")
        return False
    else:
        if st.session_state["authenticated"]:
            return True
        else:
            with st.container():
                st.text_input("Username", key="user", on_change=creds_entered, placeholder="Enter username")
                st.text_input("Password", key="passwd", type="password", on_change=creds_entered, placeholder="Enter password")
            return False

if authenticate_user():
    # App Title
    st.title("Yahoo Stock Dashboard 📊")

    # Sidebar Inputs
    st.sidebar.header("Stock Selection")
    ticker_options = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "NFLX", "META", "NVDA", "INTC", "AMD"]
    user_input = st.sidebar.selectbox("Enter Ticker Symbol", ticker_options)
    start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2020-01-01"))
    end_date = st.sidebar.date_input("End Date", datetime.date.today())
    # Interactivity options
    show_ma100 = st.sidebar.checkbox("Show 100-Day Moving Average", value=True)
    show_ma200 = st.sidebar.checkbox("Show 200-Day Moving Average", value=True)
    interactive_range = st.sidebar.slider("Select Data Range (Days)", 30, 2000, 500, 10)

    # Fetch and process data
    if user_input:
        df = yf.download(user_input, start=start_date, end=end_date)
        df = df.reset_index()
        df['Date'] = pd.to_datetime(df['Date']).dt.date  # Format Date for display
        df = df[-interactive_range:]  # Apply range filter

        st.markdown(f"### {user_input} Stock Overview")

        # Tabs for different views
        tab1, tab2, tab3 = st.tabs(["Price Summary", "Historical Data", "Charts"])

        with tab1:
            st.markdown("""
    ### **Stock Market Data Overview**  
    The displayed table provides a **detailed historical summary of AAPL (Apple Inc.) stock prices**, covering various key metrics for different trading days. It includes:  

    - **Date:** The specific trading day for each record.  
    - **Close Price:** The final price at which the stock traded on that day.  
    - **High Price:** The highest price reached during the trading session.  
    - **Low Price:** The lowest price recorded during the session.  
    - **Open Price:** The stock's price at the start of the trading session.  
    - **Volume:** The total number of shares traded on that day.  

    This data is crucial for analyzing **market trends, price fluctuations, and trading volume**, enabling traders and investors to make informed decisions based on historical performance.  
    The information can also be used for calculating **technical indicators** such as moving averages, volatility, and trend analysis.
    """)
            st.dataframe(df)  # Display dataframe

        with tab2:
            st.write(df.describe())

        with tab3:
            st.markdown("### Charts")

            # Calculate moving averages
            ma100 = df['Close'].rolling(window=100).mean()
            ma200 = df['Close'].rolling(window=200).mean()

            # Plot: Closing Price
            fig1 = plt.figure(figsize=(10, 5))
            plt.plot(df['Date'], df['Close'], label='Close Price', color='cyan')
            plt.title(f'{user_input} Closing Price')
            plt.xlabel('Date')
            plt.ylabel('Price')
            plt.legend()
            st.pyplot(fig1)

            # Plot: 100-Day MA
            if show_ma100:
                st.markdown("""
            ### 100-Day Moving Average
            The **100-day moving average (MA100)** is a widely used technical indicator that smooths out short-term fluctuations 
            to highlight long-term trends. Traders use it to determine whether a stock is in an **uptrend** or **downtrend**.
            If the stock price remains above this level, it indicates strength, whereas falling below suggests weakness.
            """)
                fig2 = plt.figure(figsize=(10, 5))
                plt.plot(df['Date'], ma100, label='100-Day MA', color='red')
                plt.title(f'{user_input} 100-Day Moving Average')
                plt.xlabel('Date')
                plt.ylabel('Price')
                plt.legend()
                st.pyplot(fig2)

            # Plot: 200-Day MA
            if show_ma200:
                st.markdown("""
            ### 200-Day Moving Average
            The **200-day moving average (MA100)** is a widely used long-term trend indicator in stock market analysis. It smooths out price fluctuations by averaging the closing prices over the past 200 days. Traders and investors often use it to identify bullish or bearish trends.
            """)
                
                fig3 = plt.figure(figsize=(10, 5))
                plt.plot(df['Date'], ma200, label='200-Day MA', color='green')
                plt.title(f'{user_input} 200-Day Moving Average')
                plt.xlabel('Date')
                plt.ylabel('Price')
                plt.legend()
                st.pyplot(fig3)

            # Plot: Combined MAs
            st.subheader("Combined Moving Average")
            fig4 = plt.figure(figsize=(10, 5))
            plt.plot(df['Date'], df['Close'], label='Close Price', color='cyan', alpha=0.5)
            if show_ma100:
                plt.plot(df['Date'], ma100, label='100-Day MA', color='red')
            if show_ma200:
                plt.plot(df['Date'], ma200, label='200-Day MA', color='green')
            plt.title(f'{user_input} Combined Moving Averages')
            plt.xlabel('Date')
            plt.ylabel('Price')
            plt.legend()
            st.pyplot(fig4)

            # LSTM Prediction
            data_training = df['Close'][0:int(len(df) * 0.70)]
            data_testing = df['Close'][int(len(df) * 0.70):]

            scaler = MinMaxScaler(feature_range=(0, 1))
            data_training_array = scaler.fit_transform(np.array(data_training).reshape(-1, 1))

            model = load_model("keras_model.h5")

            past_100_days = data_training.tail(100)
            final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
            input_data = scaler.transform(final_df.values.reshape(-1, 1))

            x_test, y_test = [], []
            for i in range(100, input_data.shape[0]):
                x_test.append(input_data[i-100:i])
                y_test.append(input_data[i, 0])

            x_test, y_test = np.array(x_test), np.array(y_test)
            y_predicted = model.predict(x_test)

            # Reverse scaling
            scale_factor = 1 / scaler.scale_[0]
            y_predicted = y_predicted * scale_factor
            y_test = y_test * scale_factor

            # Plot: Predictions vs Original
            st.subheader("Predictions vs Original")
            fig5 = plt.figure(figsize=(12, 5))
            plt.plot(y_test, label="Original Price", color="cyan", linewidth=2)
            plt.plot(y_predicted, label="Predicted Price", color="magenta", linestyle="dashed", linewidth=2)
            plt.title("Predictions vs Original")
            plt.xlabel("Time")
            plt.ylabel("Price")
            plt.legend()
            st.pyplot(fig5)
                    # --- Trend Analysis ---
            st.subheader("📈 Prediction Analysis and Recommendation")

            # Determine trend direction from predicted data
            trend = "Neutral"
            recommendation = "Hold"

            if y_predicted[-1] > y_predicted[0] * 1.02:  # >2% increase
                trend = "Uptrend"
                recommendation = "Buy ✅"
            elif y_predicted[-1] < y_predicted[0] * 0.98:  # >2% decrease
                trend = "Downtrend"
                recommendation = "Don't Buy ❌"
            else:
                trend = "Sideways"
                recommendation = "Hold 🤔"

            # Show trend and recommendation with styled markdown
            st.markdown(f"""
            <div style="padding:20px; background-color:#1e1e1e; border-radius:10px; border-left:5px solid #4CAF50;">
                <h3 style="color:white;">📊 Trend: <span style="color:cyan;">{trend}</span></h3>
                <h3 style="color:white;">💡 Recommendation: <span style="color:lightgreen;">{recommendation}</span></h3>
            </div>
            """, unsafe_allow_html=True)



    else:
        st.warning("Please select a stock ticker to display data.")
