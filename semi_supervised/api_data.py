import requests
import pandas as pd
from pymongo import MongoClient
import matplotlib.pyplot as plt

# Database connection
def create_connection(uri, db_name):
    """ Create a connection to the MongoDB database """
    try:
        client = MongoClient(uri)
        db = client[db_name]
        return db
    except Exception as e:
        print(f"Error connecting to MongoDB: {e}")
        return None

def store_data_in_mongodb(data, db, collection_name):
    """ Store the data into a MongoDB collection """
    collection = db[collection_name]
    result = collection.insert_one(data)
    print(f"Data inserted with ID: {result.inserted_id}")

def fetch_stock_data(symbol, api_key):
    """ Fetch stock data using Alpha Vantage API """
    url = f"https://www.alphavantage.co/query"
    params = {
        "function": "TIME_SERIES_DAILY",
        "symbol": symbol,
        "apikey": api_key
    }

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        if "Time Series (Daily)" in data:
            return data["Time Series (Daily)"]
        else:
            print("Error: Unexpected response format.")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Error fetching stock data: {e}")
        return None

def save_data_to_csv(data, file_path):
    """ Save the stock data to a CSV file """
    df = pd.DataFrame.from_dict(data, orient="index")
    df.columns = ["Open", "High", "Low", "Close", "Volume"]
    df.index.name = "Date"
    df.reset_index(inplace=True)
    df.to_csv(file_path, index=False)
    print(f"Data saved to {file_path}")
    return df

def visualize_stock_data(df):
    """ Visualize the stock data """
    df["Date"] = pd.to_datetime(df["Date"])
    df.sort_values("Date", inplace=True)

    # Plot closing prices
    plt.figure(figsize=(10, 6))
    plt.plot(df["Date"], df["Close"].astype(float), label="Closing Price", color="darkblue")
    plt.title("Stock Closing Prices Over Time", fontsize=16, color="white")
    plt.xlabel("Date", fontsize=12, color="white")
    plt.ylabel("Closing Price", fontsize=12, color="white")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.gca().set_facecolor("#333333")
    plt.gcf().set_facecolor("#222222")
    plt.tick_params(colors="white")
    plt.tight_layout()
    plt.show()

    # Additional Visualization: Volume Bar Chart
    plt.figure(figsize=(10, 6))
    plt.bar(df["Date"], df["Volume"].astype(float), color="darkgreen", alpha=0.7)
    plt.title("Stock Volume Traded Over Time", fontsize=16, color="white")
    plt.xlabel("Date", fontsize=12, color="white")
    plt.ylabel("Volume", fontsize=12, color="white")
    plt.grid(alpha=0.3)
    plt.gca().set_facecolor("#333333")
    plt.gcf().set_facecolor("#222222")
    plt.tick_params(colors="white")
    plt.tight_layout()
    plt.show()

def main():
    # API and database setup
    api_key = "57SSBQR23S8RB8V3"
    symbols = ["AAPL", "GOOGL", "MSFT"]  # Automating with predefined stock ticker symbols
    mongo_uri = "mongodb://localhost:27017/"
    db_name = "semi_supervised"
    collection_name = "api"

    # Connect to MongoDB
    db = create_connection(mongo_uri, db_name)

    for symbol in symbols:
        print(f"Fetching data for {symbol}...")
        stock_data = fetch_stock_data(symbol, api_key)
        if stock_data:
            # Store raw data in MongoDB
            store_data_in_mongodb({"symbol": symbol, "data": stock_data}, db, collection_name)

            # Save data to CSV
            csv_file = f"{symbol}_stock_data.csv"
            df = save_data_to_csv(stock_data, csv_file)

            # Visualize the stock data
            visualize_stock_data(df)

if __name__ == "__main__":
    main()
