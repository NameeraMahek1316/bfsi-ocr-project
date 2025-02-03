import streamlit as st
import pandas as pd
import time
from pymongo import MongoClient
import plotly.express as px

# MongoDB Connection
MONGO_URI = "mongodb://localhost:27017/"
DB_SUPERVISED = "supervised"
DB_UNSUPERVISED = "unsupervised"
DB_SEMI_SUPERVISED = "semi_supervised"

client = MongoClient(MONGO_URI)
db_supervised = client[DB_SUPERVISED]
db_unsupervised = client[DB_UNSUPERVISED]
db_semi_supervised = client[DB_SEMI_SUPERVISED]

# Streamlit UI
st.set_page_config(page_title="Finsight", layout="wide")
st.title("📊 Finsight - Financial Document Classifier")
st.sidebar.header("Upload Files")

# Document Type Selection
doc_type = st.sidebar.radio("Select Document Type:", [
    "Invoice", "Payslip", "Bank Statement", "Bank Transactions", "Stock Market"
])

# Stock Symbol Selection (Only for Stock Market option)
compare_stocks = st.sidebar.checkbox("Compare Two Stocks")
stock_choice1 = None
stock_choice2 = None
if doc_type == "Stock Market":
    if compare_stocks:
        stock_choice1 = st.sidebar.selectbox("Select First Stock Symbol", ["AAPL", "GOOGL", "MSFT"])
        stock_choice2 = st.sidebar.selectbox("Select Second Stock Symbol", ["AAPL", "GOOGL", "MSFT"])
    else:
        stock_choice1 = st.sidebar.selectbox("Select Stock Symbol", ["AAPL", "GOOGL", "MSFT"])

uploaded_files = st.sidebar.file_uploader("Upload Files", type=["png", "jpg", "jpeg", "pdf"], accept_multiple_files=True)

def display_insights(categories):
    # More specific insights based on the document type
    if categories:
        return f"💡 You've spent a large amount on {', '.join(categories)}. Consider reviewing your spending habits."
    return "💡 Your spending seems balanced across categories."

def stock_insights(stock_symbol):
    if stock_symbol == "AAPL":
        return "📉 Apple’s stock has been on a rollercoaster lately. Keep an eye out for any major news events!"
    elif stock_symbol == "GOOGL":
        return "💼 Google's stock is still holding strong, but you might want to think about diversifying your investments."
    elif stock_symbol == "MSFT":
        return "📈 Microsoft has been outperforming! But remember, the market can change quickly. Stay alert!"
    return "📊 Keep monitoring the market closely!"

if st.sidebar.button("Process Files"):
    with st.spinner("Processing... Please wait."):
        time.sleep(3)  # Simulating backend processing
        
        if doc_type == "Invoice":
            collection = db_supervised["invoice_classified"]
            data = pd.DataFrame(list(collection.find({}, {"_id": 0})))
            st.subheader("Extracted Transactions")
            st.dataframe(data)
            
            st.subheader("Visualizations")
            # Interactive Pie chart using Plotly with dark background
            fig = px.pie(data, names='category', values='price', title="Invoice Category Distribution",
                         color_discrete_sequence=px.colors.sequential.Plasma)
            st.plotly_chart(fig)
            
            # Insights for invoice categories (more specific insight)
            high_expense_categories = [category for category in data['category'].unique() 
                                        if data[data['category'] == category]['price'].sum() > 500]
            st.write(display_insights(high_expense_categories))
            
            # Interactive Bar chart using Plotly
            fig = px.bar(data, x='description', y='price', title="Invoice Description vs Price",
                         color='category', color_discrete_sequence=px.colors.sequential.Plasma)
            st.plotly_chart(fig)

        elif doc_type == "Payslip":
            collection = db_supervised["payslip_classified"]
            data = pd.DataFrame(list(collection.find({}, {"_id": 0})))
            st.subheader("Extracted Transactions")
            st.dataframe(data)
            
            st.subheader("Visualizations")
            # Interactive Pie chart using Plotly with dark background
            fig = px.pie(data, names='Description', values='Price', title="Payslip Description Distribution",
                         color_discrete_sequence=px.colors.sequential.Plasma)
            st.plotly_chart(fig)
            
            # Insights for payslip descriptions (more specific insight)
            high_salary_categories = [description for description in data['Description'].unique() 
                                      if data[data['Description'] == description]['Price'].sum() > 1000]
            st.write(display_insights(high_salary_categories))

            # Interactive Bar chart using Plotly
            fig = px.bar(data, x='Description', y='Price', title="Payslip Description vs Price",
                         color='Description', color_discrete_sequence=px.colors.sequential.Plasma)
            st.plotly_chart(fig)

        elif doc_type == "Bank Statement":
            collection = db_supervised["classified_bankstatements"]
            data = pd.DataFrame(list(collection.find({}, {"_id": 0})))
            st.subheader("Extracted Transactions")
            st.dataframe(data)
            
            st.subheader("Visualizations")
            # Interactive Pie chart using Plotly with dark background
            fig = px.pie(data, names='Category', values='Amount', title="Bank Statement Category Distribution",
                         color_discrete_sequence=px.colors.sequential.Plasma)
            st.plotly_chart(fig)
            
            # Insights for bank statement categories (more specific insight)
            large_spending_categories = [category for category in data['Category'].unique() 
                                        if data[data['Category'] == category]['Amount'].sum() > 500]
            st.write(display_insights(large_spending_categories))

            # Interactive Bar chart using Plotly
            fig = px.bar(data, x='Category', y='Amount', title="Bank Statement Category vs Amount",
                         color='Category', color_discrete_sequence=px.colors.sequential.Plasma)
            st.plotly_chart(fig)

        elif doc_type == "Bank Transactions":
            excel_path = r"C:\Users\namee\OneDrive\Desktop\bfsi-ocr-project\unsupervised\yes.xlsx"  # Path to stored Excel file
            data = pd.read_excel(excel_path)
            st.subheader("Extracted Transactions")
            st.dataframe(data)
            
            st.subheader("Visualizations")
            # Interactive Bar chart using Plotly
            fig = px.bar(data, x='Category', y='Balance', title="Bank Transactions Category vs Balance",
                         color='Category', color_discrete_sequence=px.colors.sequential.Plasma)
            st.plotly_chart(fig)

            # Interactive Pie chart using Plotly with dark background
            fig = px.pie(data, names='Category', values='Balance', title="Bank Transactions Category Distribution",
                         color_discrete_sequence=px.colors.sequential.Plasma)
            st.plotly_chart(fig)

        elif doc_type == "Stock Market":
            if compare_stocks:
                if stock_choice1 and stock_choice2:
                    st.title(f"📈 {stock_choice1} vs {stock_choice2} Stock Market Data")

                    # Fetch data for both selected stocks
                    collection = db_semi_supervised["api"]

                    # Fetch stock data for Stock 1 (Microsoft)
                    stock_data1 = collection.find_one({"symbol": stock_choice1})
                    if stock_data1:
                        stock_dates1 = list(stock_data1["data"].keys())
                        stock_close1 = [float(stock_data1["data"][date]["4. close"]) for date in stock_dates1]
                        stock_volume1 = [int(stock_data1["data"][date]["5. volume"]) for date in stock_dates1]

                        df_stock1 = pd.DataFrame({
                            'Date': pd.to_datetime(stock_dates1),
                            'Close Price': stock_close1,
                            'Volume': stock_volume1
                        })

                    # Fetch stock data for Stock 2 (Google)
                    stock_data2 = collection.find_one({"symbol": stock_choice2})
                    if stock_data2:
                        stock_dates2 = list(stock_data2["data"].keys())
                        stock_close2 = [float(stock_data2["data"][date]["4. close"]) for date in stock_dates2]
                        stock_volume2 = [int(stock_data2["data"][date]["5. volume"]) for date in stock_dates2]

                        df_stock2 = pd.DataFrame({
                            'Date': pd.to_datetime(stock_dates2),
                            'Close Price': stock_close2,
                            'Volume': stock_volume2
                        })

                    # Combine both stock dataframes for plotting together
                    combined_df = pd.concat([df_stock1.assign(Stock=stock_choice1), df_stock2.assign(Stock=stock_choice2)])
                    
                    # Time Series Line Plot with Colors for MSFT (Red) and GOOGL (Blue)
                    fig_stock = px.line(combined_df, x='Date', y='Close Price', color='Stock',
                                        title="Stock Comparison: Price Trend",
                                        color_discrete_map={stock_choice1: 'red', stock_choice2: 'blue'})
                    st.plotly_chart(fig_stock)

                    # Trading Volume Bar Plot with Colors for MSFT (Red) and GOOGL (Blue)
                    fig_vol = px.bar(combined_df, x='Date', y='Volume', color='Stock',
                                     title="Stock Comparison: Trading Volume",
                                     color_discrete_map={stock_choice1: 'red', stock_choice2: 'blue'})
                    st.plotly_chart(fig_vol)

                    # Display Quirky Insights for both stocks
                    st.write(f"Insights for {stock_choice1}: {stock_insights(stock_choice1)}")
                    st.write(f"Insights for {stock_choice2}: {stock_insights(stock_choice2)}")
                else:
                    st.write("Please select two stock symbols first!")
            else:
                if stock_choice1:
                    st.title(f"📈 {stock_choice1} Stock Market Data")

                    # Fetch data for the selected stock
                    collection = db_semi_supervised["api"]

                    # Fetch stock data for Stock 1
                    stock_data1 = collection.find_one({"symbol": stock_choice1})
                    if stock_data1:
                        stock_dates1 = list(stock_data1["data"].keys())
                        stock_close1 = [float(stock_data1["data"][date]["4. close"]) for date in stock_dates1]
                        stock_volume1 = [int(stock_data1["data"][date]["5. volume"]) for date in stock_dates1]

                        df_stock1 = pd.DataFrame({
                            'Date': pd.to_datetime(stock_dates1),
                            'Close Price': stock_close1,
                            'Volume': stock_volume1
                        })

                    # Time Series Line Plot for Single Stock
                    fig_stock = px.line(df_stock1, x='Date', y='Close Price', title=f"{stock_choice1} Time Series")
                    st.plotly_chart(fig_stock)

                    # Trading Volume Bar Plot for Single Stock
                    fig_vol = px.bar(df_stock1, x='Date', y='Volume', title=f"{stock_choice1} Trading Volume")
                    st.plotly_chart(fig_vol)

                    # Display Quirky Insights for the selected stock
                    st.write(f"Insights for {stock_choice1}: {stock_insights(stock_choice1)}")
                else:
                    st.write("Please select a stock symbol first!")

st.sidebar.info("Limit: 200MB per file • PNG, JPG, JPEG, PDF")
