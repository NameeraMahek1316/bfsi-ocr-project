import cv2
import pytesseract
import re
import csv
import os
import pandas as pd
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, pipeline
import torch
import matplotlib.pyplot as plt
import sys
import argparse
from pymongo import MongoClient

# MongoDB URI and Database Configuration
MONGO_URI = "mongodb://localhost:27017"  # Update with your MongoDB URI
client = MongoClient(MONGO_URI)
db = client['supervised']  # MongoDB database name
transactions_collection = db['invoice']  # MongoDB collection for transactions
classified_collection = db['invoice_classified']  # MongoDB collection for classified data

# Path to Tesseract-OCR executable
pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

# List of image paths
INPUT_IMAGES = [
    r"C:\Users\namee\OneDrive\Desktop\infosys_project\Supervised_images\dummy_invoices\Invoice1.png",
    r"C:\Users\namee\OneDrive\Desktop\infosys_project\Supervised_images\dummy_invoices\Invoice2.png",
    r"C:\Users\namee\OneDrive\Desktop\infosys_project\Supervised_images\dummy_invoices\Invoice3.png",
    r"C:\Users\namee\OneDrive\Desktop\infosys_project\Supervised_images\dummy_invoices\Invoice4.png",
    r"C:\Users\namee\OneDrive\Desktop\infosys_project\Supervised_images\dummy_invoices\Invoice5.png",
    r"C:\Users\namee\OneDrive\Desktop\infosys_project\Supervised_images\dummy_invoices\Invoice6.png",
    # Add other paths...
]

TEMP_DIR = "temp_data_images"
EXTRACTED_CSV = os.path.join(TEMP_DIR, "text2.csv")
CLASSIFIED_CSV = os.path.join(TEMP_DIR, "classified1_output.csv")

# Ensure temp directory exists
os.makedirs(TEMP_DIR, exist_ok=True)

# --- Image Preprocessing ---
def preprocess_image(image_path):
    try:
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary
    except Exception as e:
        print(f"Error preprocessing image {image_path}: {e}")
        return None

# --- Transaction Extraction ---
def extract_transactions(text):
    transactions = []
    rows = text.split("\n")
    for row in rows:
        row = re.sub(r'[^A-Za-z0-9\s.,$-]', '', row).strip()
        match = re.search(r'([A-Za-z\s]+)\s+(\d+)\s?\$\s?([\d,]+)\s?\$\s?([\d,]+)', row)
        if match:
            transactions.append((
                match.group(1).strip(),  # Description
                int(match.group(2)),    # Qty
                float(match.group(3).replace(",", "")),  # Price
                float(match.group(4).replace(",", ""))   # Total
            ))
    return transactions

# --- Image Processing and CSV Generation ---
def process_images(image_paths, output_csv):
    try:
        with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Description", "Qty", "Price", "Total"])

            for image_path in image_paths:
                processed_image = preprocess_image(image_path)
                if processed_image is not None:
                    text = pytesseract.image_to_string(processed_image, lang='eng')
                    transactions = extract_transactions(text)
                    if transactions:
                        writer.writerows(transactions)

                        # Insert transactions into MongoDB
                        for transaction in transactions:
                            transaction_data = {
                                "description": transaction[0],
                                "qty": transaction[1],
                                "price": transaction[2],
                                "total": transaction[3]
                            }
                            transactions_collection.insert_one(transaction_data)
                    else:
                        print(f"No transactions found in {image_path}")
    except Exception as e:
        print(f"Error processing images: {e}")

# Define categories for classification
categories = [
    'Bill Payments', 'Entertainment', 'Food & Restaurants', 'Funds Transfer',
    'General Store', 'Grocery Stores', 'Health & Medical', 'Loan Payment',
    'Retail Business', 'Shopping', 'Tax Payments'
]

# Define label-to-ID and ID-to-label mappings
label2id = {category: idx for idx, category in enumerate(categories)}
id2label = {idx: category for idx, category in enumerate(categories)}

# Debugging: Print mappings
print("Label-to-ID mapping:", label2id)
print("ID-to-Label mapping:", id2label)

# --- Updated Classification Function ---
def classify_data(input_csv, output_csv, model_path):
    data = pd.read_csv(input_csv)

    # Load model and tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained(model_path)
    model = DistilBertForSequenceClassification.from_pretrained(model_path)
    classifier = pipeline('text-classification', model=model, tokenizer=tokenizer)

    predictions = []
    for _, row in data.iterrows():
        description = row['Description']
        price = row['Price']
        input_text = f"Description: {description}, Price: {price}"
        
        # Classify the input text
        result = classifier(input_text, top_k=1)[0]  # Get top prediction
        label = result['label']  # e.g., 'LABEL_0'
        confidence = result['score']  # e.g., 0.95
        
        # Convert label (e.g., 'LABEL_0') to category name
        label_index = int(label.replace('LABEL_', ''))  # Extract numeric index
        category_name = id2label.get(label_index, "Unknown Category")  # Map index to category

        # Append results
        predictions.append({
            "Description": description,
            "Price": price,
            "Category": category_name,
            "Confidence": confidence
        })

        # Insert classified data into MongoDB
        classified_data = {
            "description": description,
            "price": price,
            "category": category_name,
            "confidence": confidence
        }
        classified_collection.insert_one(classified_data)

    # Save predictions to CSV
    pd.DataFrame(predictions).to_csv(output_csv, index=False)
    print(f"Classified data saved to {output_csv}")

# --- Visualization ---
def visualize_data(csv_path):
    try:
        data = pd.read_csv(csv_path)

        # Pie chart: Total spent per category
        category_totals = data.groupby('Category')['Price'].sum()
        plt.figure(figsize=(8, 8))
        plt.pie(category_totals, labels=category_totals.index, autopct='%1.1f%%', startangle=140, colors=plt.cm.Paired.colors)
        plt.title('Total Spent Per Category')
        plt.axis('equal')
        plt.show()

        # Bar chart: Total price per description
        sorted_data = data.sort_values(by='Price', ascending=False)
        plt.figure(figsize=(12, 6))
        plt.bar(sorted_data['Description'], sorted_data['Price'], color='red', alpha=0.8)
        plt.xticks(rotation=45, ha='right')
        plt.xlabel('Description')
        plt.ylabel('Price')
        plt.title('Total Price Per Description')
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Error visualizing data: {e}")

# --- Main Workflow ---
if __name__ == "__main__":
    try:
        print("Step 1: Processing images...")
        process_images(INPUT_IMAGES, EXTRACTED_CSV)

        print("Step 2: Classifying data...")
        classify_data(EXTRACTED_CSV, CLASSIFIED_CSV, r'C:\Users\namee\OneDrive\Desktop\infosys_project\finetuned_distilbert')

        print("Step 3: Visualizing data...")
        visualize_data(CLASSIFIED_CSV)
    except Exception as e:
        print(f"Error in workflow: {e}")
