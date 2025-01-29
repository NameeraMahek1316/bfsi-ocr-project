import os
import cv2
import pytesseract
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, pipeline
import torch
from pymongo import MongoClient

# MongoDB connection setup
def create_connection(uri, db_name):
    """ Create a connection to the MongoDB database """
    try:
        # Connect to the MongoDB client and get the database
        db = db = MongoClient(uri)[db_name]
        
        # Check if db is not None (MongoClient will return a valid object)
        if db is not None:
            print("Connected to MongoDB successfully!")
            return db
        else:
            print(f"Failed to connect to database {db_name}")
            return None
    except Exception as e:
        print(f"Error connecting to MongoDB: {e}")
        return None


def upload_to_mongodb(df, db, collection_name):
    """ Upload DataFrame to MongoDB """
    try:
        if df is not None and not df.empty:
            records = df.to_dict(orient='records')
            collection = db[collection_name]
            result = collection.insert_many(records)
            print(f"Inserted {len(result.inserted_ids)} records into MongoDB collection '{collection_name}'.")
        else:
            print("No data to upload: DataFrame is empty or None.")
    except Exception as e:
        print(f"Error uploading data to MongoDB: {e}")

# Set up Tesseract path (update according to your system)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Paths to images for OCR processing
IMAGE_FOLDER = r"C:\Users\namee\OneDrive\Desktop\infosys_project\Supervised_images\bank_statements"

# Preprocessing function
def preprocess_image(image_path):
    try:
        print(f"Preprocessing: {image_path}")
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Image not found: {image_path}")

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary
    except Exception as e:
        print(f"Error during preprocessing: {e}")
        return None

# OCR function
def extract_text_from_image(image):
    try:
        text = pytesseract.image_to_string(image)
        print("OCR completed successfully.")
        return text
    except Exception as e:
        print(f"Error during OCR: {e}")
        return ""

# Transaction extraction function
def extract_transactions(text):
    rows = text.split("\n")
    transactions = []
    for row in rows:
        row = re.sub(r'[^A-Za-z0-9\s.,-]', '', row).strip()

        # Match rows with valid descriptions and amounts
        match = re.search(r'(.+?)\s+([\d,.]+\.\d{2})$', row)
        if match:
            description = match.group(1).strip()
            amount = match.group(2).replace(",", "").strip()
            transactions.append((description, amount))
    return transactions

# Save transactions to CSV
def save_transactions_to_csv(transactions, output_csv):
    try:
        if transactions:
            df = pd.DataFrame(transactions, columns=["Description", "Amount"])
            df.to_csv(output_csv, index=False)
            print(f"Transactions saved to {output_csv}.")
            return df
        else:
            print("No transactions to save.")
            return pd.DataFrame()
    except Exception as e:
        print(f"Error saving transactions to CSV: {e}")
        return pd.DataFrame()

# Load fine-tuned DistilBERT model
def load_distilbert_model(model_path):
    try:
        tokenizer = DistilBertTokenizer.from_pretrained(model_path)
        model = DistilBertForSequenceClassification.from_pretrained(model_path, use_auth_token="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")
        print("DistilBERT model loaded successfully.")
        return tokenizer, model
    except Exception as e:
        print(f"Error loading DistilBERT model: {e}")
        return None, None

# Define categories for classification
categories = [
    'Bill Payments', 'Entertainment', 'Food & Restaurants', 'Funds Transfer', 
    'General Store', 'Grocery Stores', 'Health & Medical', 'Loan Payment', 
    'Retail Business', 'Shopping', 'Tax Payments'
]

# Define MCC-to-Category Mapping
mcc_to_category = {
    5812: 'Food & Restaurants', 7832: 'Entertainment', 5411: 'Grocery Stores',
    4900: 'Bill Payments', 6012: 'Funds Transfer', 6013: 'Loan Payment',
    5813: 'Shopping', 5999: 'General Store', 8011: 'Health & Medical',
    8049: 'Health & Medical', 8062: 'Health & Medical', 8099: 'Health & Medical',
    5912: 'Health & Medical', 4511: 'Funds Transfer', 9399: 'Tax Payments',
    6010: 'Funds Transfer', 5541: 'General Store', 5732: 'Software & Subscriptions',
    5211: 'Building', 5734: 'Entertainment', 5699: 'Shopping', 5999: 'Retail Store'
}

# Define label-to-ID and ID-to-label mappings
label2id = {category: idx for idx, category in enumerate(categories)}
id2label = {idx: category for idx, category in enumerate(categories)}

# Debugging: Print the id2label and label2id mappings
print("id2label:", id2label)
print("label2id:", label2id)

# Create a pipeline for classification
classifier = pipeline('text-classification', model=DistilBertForSequenceClassification.from_pretrained("./finetuned_distilbert"), tokenizer=DistilBertTokenizer.from_pretrained("./finetuned_distilbert"), top_k=None)

# Function to predict categories using MCC or DistilBERT
def predict(description, amount, mcc_code=None):
    # Check if MCC code exists and map to a category
    if mcc_code in mcc_to_category:
        return mcc_to_category[mcc_code], 1.0  # Full confidence for MCC-based categorization

    # If no MCC code or invalid MCC, use DistilBERT
    input_text = f"Description: {description}, Amount: {amount}"
    predictions = classifier(input_text, top_k=None)[0]  # Get first prediction dict

    # Debugging: Print the raw prediction output
    print(f"Raw prediction output: {predictions}")

    category = predictions['label']  # e.g. 'LABEL_0'
    confidence = predictions['score']  # e.g. 0.164128

    # Convert 'LABEL_0' to index and map to category
    label_index = int(category.replace('LABEL_', ''))
    category_name = id2label.get(label_index, category)  # Map index to category
    return category_name, confidence

# Classify transactions using DistilBERT
def classify_transactions(input_csv, output_csv, tokenizer, model, db, collection_name):
    try:
        data = pd.read_csv(input_csv)

        if data.empty:
            print("Input CSV is empty. No transactions to classify.")
            return

        if "Description" not in data.columns or "Amount" not in data.columns:
            raise ValueError("CSV must contain 'Description' and 'Amount' columns.")

        predictions = []
        for _, row in data.iterrows():
            description = row['Description']
            amount = row['Amount']
            mcc_code = row['MCC Code'] if 'MCC Code' in row and not pd.isna(row['MCC Code']) else None
            
            category, confidence = predict(description, amount, mcc_code)
            predictions.append({
                'Description': description,
                'Amount': amount,
                'MCC Code': mcc_code,
                'Category': category,
                'Confidence': confidence
            })

        # Convert predictions to DataFrame
        predictions_df = pd.DataFrame(predictions)
        predictions_df.to_csv(output_csv, index=False)
        print(f"Classified transactions saved to {output_csv}.")

        # Save classified transactions to MongoDB
        upload_to_mongodb(predictions_df, db, collection_name)
    except Exception as e:
        print(f"Error classifying transactions: {e}")

# Visualization function
def visualize_transactions(classified_csv):
    try:
        # Read the classified CSV
        df = pd.read_csv(classified_csv)

        if df.empty:
            print("Classified transactions CSV is empty.")
            return

        # Count the frequency of each category
        category_counts = df['Category'].value_counts()

        # Create a bar plot for the categories
        plt.figure(figsize=(12, 6))

        # Bar chart
        plt.subplot(1, 2, 1)
        category_counts.plot(kind='bar', color='skyblue')
        plt.title('Distribution of Classified Transaction Categories (Bar Chart)')
        plt.xlabel('Categories')
        plt.ylabel('Count')
        plt.xticks(rotation=45, ha='right')

        # Pie chart
        plt.subplot(1, 2, 2)
        category_counts.plot(kind='pie', autopct='%1.1f%%', colors=plt.cm.Paired.colors, figsize=(8, 8), legend=False)
        plt.title('Distribution of Classified Transaction Categories (Pie Chart)')

        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Error visualizing transactions: {e}")


# Updated pipeline to save classified transactions
def process_images_pipeline(image_folder, output_csv, classified_csv, model_path, db_uri, db_name, extracted_collection, classified_collection):
    try:
        all_transactions = []
        for file_name in os.listdir(image_folder):
            image_path = os.path.join(image_folder, file_name)
            processed_image = preprocess_image(image_path)
            if processed_image is not None:
                extracted_text = extract_text_from_image(processed_image)
                transactions = extract_transactions(extracted_text)
                all_transactions.extend(transactions)

        # Save extracted transactions to CSV
        df = save_transactions_to_csv(all_transactions, output_csv)

        # Connect to MongoDB and save extracted transactions
        db = create_connection(db_uri, db_name)
        if db is not None:
            upload_to_mongodb(df, db, extracted_collection)

            # Load the fine-tuned model
            tokenizer, model = load_distilbert_model(model_path)
            if tokenizer and model:
                # Classify transactions and save them to MongoDB
                classify_transactions(output_csv, classified_csv, tokenizer, model, db, classified_collection)

                # Visualize classified transactions
                visualize_transactions(classified_csv)
    except Exception as e:
        print(f"Error in pipeline: {e}")

# Example usage
if __name__ == "__main__":
    OUTPUT_CSV = "text3.csv"
    CLASSIFIED_CSV = "classified_transactions.csv"
    MODEL_PATH = r'C:\Users\namee\OneDrive\Desktop\infosys_project\finetuned_distilbert'
    MONGO_URI = "mongodb://localhost:27017/?serverSelectionTimeoutMS=50000"
    DB_NAME = "supervised"
    EXTRACTED_COLLECTION = "bank_statements"
    CLASSIFIED_COLLECTION = "classified_bankstatements"

    process_images_pipeline(
        IMAGE_FOLDER,
        OUTPUT_CSV,
        CLASSIFIED_CSV,
        MODEL_PATH,
        MONGO_URI,
        DB_NAME,
        EXTRACTED_COLLECTION,
        CLASSIFIED_COLLECTION
    )
