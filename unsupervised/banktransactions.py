import fitz  # PyMuPDF
import pytesseract
import pdfplumber
import re

from pymongo import MongoClient
import os

import numpy as np
import pandas as pd
from PIL import Image

import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
from adjustText import adjust_text

from transformers import pipeline, TrainingArguments, Trainer
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.model_selection import train_test_split
import torch
import gc
gc.collect()

# Set Tesseract OCR Path (update this to your Tesseract installation path)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

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

def insert_file(db, collection_name, file_path):
    """ Insert a file into a MongoDB collection """
    with open(file_path, 'rb') as file:
        file_data = file.read()
    result = db[collection_name].insert_one({
        "name": os.path.basename(file_path),
        "data": file_data
    })
    return result.inserted_id

def retrieve_file(db, collection_name, file_id):
    """ Retrieve a file from a MongoDB collection """
    document = db[collection_name].find_one({"_id": file_id})
    if document:
        with open("retrieved_file.pdf", 'wb') as file:
            file.write(document["data"])
        return "retrieved_file.pdf"
    return None

def insert_result(db, collection_name, file_id, result_path):
    """ Insert a result into a MongoDB collection """
    with open(result_path, 'rb') as file:
        result_data = file.read()
    db[collection_name].insert_one({
        "file_id": file_id,
        "result": result_data
    })

def read_pdf(pdf_path):
    """
    Reads the first page of a PDF and converts it to an image.
    """
    try:
        doc = fitz.open(pdf_path)
        page = doc.load_page(0)  # number of page
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        return img
    except Exception as e:
        print(f"Error reading PDF file: {e}")
        return None

def ocr_image(image):
    """
    Extracts text from an image using Tesseract OCR.
    """
    return pytesseract.image_to_string(image, lang='eng')

def extract_transactions(text,pdf_path):
    """
    Parses the extracted text to find transactions and returns them as a DataFrame.
    """
    data = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            lines = text.split('\n')
            for line in lines:
                data.append(line.split())

    # Determine the number of columns dynamically
    max_columns = max(len(row) for row in data)
    columns = ['Transaction Date', 'Value Date', 'Description', 'Debit', 'Credit', 'Balance']
    if max_columns > len(columns):
        columns.extend([f'Extra Column {i}' for i in range(max_columns - len(columns))])            

    df = pd.DataFrame(data, columns=columns)

    print(df.columns)

    # Ensure 'Transaction Date' column exists before accessing it
    if 'Transaction Date' in df.columns:
        for i in df.index:
            if type(df["Transaction Date"][i]) == float:
                df["Transaction Date"][i] = df["Transaction Date"][i]
        df["Transaction Date"] = df["Transaction Date"]

    delete = []
    headers = ["Date", "Description", "Credit", "Debit", "Balance"]
    print(df.columns)
    for i in df.index:
        row = df.iloc[i, :].tolist()
        nan_c = 0
        # For checking empty rows
        for j in row:
            try:
                if np.isnan(j):
                    nan_c += 1
            except:
                continue
        if nan_c == len(df.columns):
            delete.append(i)

        # For checking headers in between
        for j in headers:
            if j in row:
                delete.append(i)
    df = df.drop(delete, axis=0)

    # For merging multiple lines in one
    last = 0
    delete = []
    for i in df.index:
        if type(df["Value Date"][i]) == float and type(df["Description"][i]) == str:
            buff = df["Description"][last] + df["Description"][i]
            df["Description"][last] = buff
            delete.append(i)
        else:
            last = i
    df = df.drop(delete, axis=0)

    print(df.columns)  # Debugging line to check the columns


    df["Credit"] = df["Credit"].apply(lambda x: str(x).replace(",", "") if x is not None else '0')
    df["Debit"] = df["Debit"].apply(lambda x: str(x).replace(",", "") if x is not None else '0')

    df["Credit"] = pd.to_numeric(df["Credit"].apply(lambda x: str(x).replace(",", "") if x is not None else '0'), errors='coerce').fillna(0)
    df["Debit"] = pd.to_numeric(df["Debit"].apply(lambda x: str(x).replace(",", "") if x is not None else '0'), errors='coerce').fillna(0)
    df["Value Date"] = df["Value Date"].apply(lambda x: x[3:] if x is not None else x)

    df = df[["Transaction Date", "Value Date", "Description", "Debit", "Credit", "Balance"]]
    return df

# Define labels
labels = {
    "restaurant": "FOOD", "cafe": "FOOD", "food": "FOOD", "dining": "FOOD", "swiggy": "FOOD", "faasos": "FOOD", "zomato": "FOOD",
    "mall": "SHOPPING", "store": "SHOPPING", "shopping": "SHOPPING", "retail": "SHOPPING", "amazon": "SHOPPING", "flipkart": "SHOPPING",
    "atm": "ATM", "atd": "ATM",
    "upi": "UPI", "paytm": "UPI", "funds trf": "UPI", "imps": "UPI", "rrn": "UPI", "pos": "UPI", "neft": "UPI", "rtgs": "UPI", "txn paytm": "UPI",
    "loan": "Other", "emi": "Other", "mutualfund": "Other", 
    "net txn": "Other", "cash": "Other", "interest": "Other",
    "metro": "Other", "ola": "Other", "refund": "Other", "charge": "Other", "pca": "Other"
}

# Create a mapping from string labels to integers
label_mapping = {label: idx for idx, label in enumerate(set(labels.values()))}

# Prepare data for training
data = []
for key, value in labels.items():
    data.append({"text": key, "label": label_mapping[value]})

df = pd.DataFrame(data)
train_texts, val_texts, train_labels, val_labels = train_test_split(df['text'], df['label'], test_size=0.2)

# Load pre-trained model and tokenizer
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(label_mapping))

# Tokenize data
train_encodings = tokenizer(train_texts.tolist(), truncation=True, padding=True)
val_encodings = tokenizer(val_texts.tolist(), truncation=True, padding=True)

# Convert to torch dataset
class TransactionDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = TransactionDataset(train_encodings, train_labels.tolist())
val_dataset = TransactionDataset(val_encodings, val_labels.tolist())

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,)

# Train the model
trainer.train()

# Save the model
model.save_pretrained('./transaction_model')
tokenizer.save_pretrained('./transaction_model')

# Load the trained model for inference
classifier = pipeline('text-classification', model='./transaction_model', tokenizer='./transaction_model')


def classify_transactions(transactions):
    """
    Classifies the transactions into different categories using a pre-trained model.
    """
    t = transactions["Description"].apply(lambda x: x.lower() if x is not None else x)

    # Removing numbers and special characters
    text = t.replace(to_replace="[0-9]", value="", regex=True).apply(
        lambda x: x.replace("/", "").replace("\\", "").replace(":", "").replace("\n", " ").replace("-", " ").replace("/", " ") if x is not None else x)

    # Removing extra spaces created due to the above step
    for i in range(len(text)):
        if i >= len(text):
            print(f"Index {i} out of bounds for text with length {len(text)}")
            continue
        x = text.iloc[i].split() if text.iloc[i] is not None else []
        for j in range(len(x)):
            x[j] = x[j].strip()
        text.iloc[i] = " ".join(x)

    labs = []

    # Labelling the transaction according to the dictionary defined
    for i in text:
        f = 0
        for j in list(labels.keys()):
            if j in i:
                labs.append(labels[j])
                f = 1
                break
        if f == 0:
            labs.append("Other")
    transactions["Category"] = pd.DataFrame(labs)

    x = transactions.Description.apply(lambda x: re.findall(r'[\w\.-]+@[\w\.-]+', x) if x is not None else [])
    transactions["Remark"] = pd.DataFrame(x)

    return transactions

def visualize_data(transactions):
    """
    Visualizes the classified transactions as both a pie chart and a bar chart.
    """
    category_counts = transactions['Category'].value_counts()

    # Pie Chart
    plt.figure(figsize=(10, 6))
    wedges, texts, autotexts = plt.pie(
        category_counts,
        labels=category_counts.index,
        autopct='%1.1f%%',
        labeldistance=1.1,
        colors=plt.cm.Dark2.colors
    )

    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
    plt.title('Transaction Classification - Pie Chart', fontsize=14)
    plt.show()

    # Bar Chart
    plt.figure(figsize=(10, 6))
    category_counts.plot(kind='bar', color=plt.cm.Dark2.colors, alpha=0.8)
    plt.title('Transaction Classification - Bar Chart', fontsize=14)
    plt.xlabel('Category', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.xticks(rotation=45, fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def main(pdf_path):
    # Database setup
    mongo_uri = "mongodb://localhost:27017/"
    db_name = "unsupervised"
    files_collection = "transactions_pdf"
    results_collection = "classified_pdf"

    db = create_connection(mongo_uri, db_name)

    # Insert PDF file into database
    file_id = insert_file(db, files_collection, pdf_path)

    # Retrieve PDF file from database
    retrieved_pdf_path = retrieve_file(db, files_collection, file_id)
    if retrieved_pdf_path:
        # Process the retrieved PDF file
        image = read_pdf(retrieved_pdf_path)
        if image is None:
            print("Failed to read the PDF file.")
            return

        text = ocr_image(image)
        transactions = extract_transactions(text,pdf_path)
        transactions = classify_transactions(transactions)
        transactions.to_excel(pdf_path.replace('.pdf', '.xlsx'), index=False)
        visualize_data(transactions)

        # Save the output figure
        output_figure_path = "output_figure.png"
        plt.savefig(output_figure_path)

        # Insert the output figure into the database
        insert_result(db, results_collection, file_id, output_figure_path)

if __name__ == "__main__":
    main(r"C:\Users\namee\OneDrive\Desktop\infosys_project\backend\unsupervised\yes.pdf")
