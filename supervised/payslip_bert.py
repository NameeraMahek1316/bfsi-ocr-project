from PIL import Image, ImageEnhance, ImageFilter
import pytesseract
import re
import pandas as pd
import os
import matplotlib.pyplot as plt
from pymongo import MongoClient
import streamlit as st


# MongoDB URI and Database Configuration
MONGO_URI = "mongodb://localhost:27017"  # Update with your MongoDB URI
client = MongoClient(MONGO_URI)
db = client['supervised']  # MongoDB database name
transactions_collection = db['payslip']  # MongoDB collection for transactions
classified_collection = db['payslip_classified']  # MongoDB collection for classified data

pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

# Load the uploaded image
image_path = r'C:\Users\namee\OneDrive\Desktop\infosys_project\Supervised_images\Pyaslips\image.png'
image = Image.open(image_path)

# Preprocess the image
# Convert to grayscale
gray_image = image.convert('L')

# Increase contrast
enhancer = ImageEnhance.Contrast(gray_image)
enhanced_image = enhancer.enhance(2)

# Optional: Apply binarization (converting to black and white)
bw_image = enhanced_image.point(lambda p: p > 128 and 255)

# Apply a slight blur to reduce noise
processed_image = bw_image.filter(ImageFilter.MedianFilter(size=3))

# Perform OCR using Tesseract with additional config for table recognition
extracted_text = pytesseract.image_to_string(processed_image, config='--psm 6')

# Debugging: Print the raw OCR output to inspect text
print("OCR Output:\n", extracted_text)

# Clean the extracted text and isolate the table rows
lines = extracted_text.split('\n')

print("\nRaw lines extracted:")
for line in lines:
    print(f"Line: '{line}'")

table_data = []

# Updated regex to capture descriptions (e.g., "Basic Salary") and amounts (e.g., "4,833")
# This assumes the description is the first part of the line and the price is the last part of the line
table_pattern = re.compile(r'([a-zA-Z\s]+)\s+\d+\s+[\d,]+\s+([\d,]+\.\d{2}|\d{1,3}(?:,\d{3})*)')

# Loop through each line to extract the relevant data
for line in lines:
    # Skip lines that are just headers or non-relevant text like "Earnings", "Quantity"
    if 'Earnings' in line or 'Quantity' in line or 'Rate' in line or 'Amount' in line or not line.strip():
        continue
    
    print(f"Processing line: '{line}'")  # Debug print each line being processed

    # Preprocess line: Remove extra spaces and normalize the formatting
    line = re.sub(r'\s+', ' ', line).strip()  # Replace multiple spaces with a single space

    match = table_pattern.match(line)
    if match:
        description = match.group(1).strip()  # Capture the description (e.g., "Basic Salary")
        price = match.group(2).strip()  # Capture the amount (e.g., "4,833")
        table_data.append({'Description': description, 'Price': price})
    else:
        print(f"No match for line: '{line}'") 

# Debugging: Print the parsed table data
print("Parsed Table Data:\n", table_data)

# Check current directory to ensure the file will be saved correctly
print("Current working directory:", os.getcwd())

# If there is data to save, write to MongoDB
if table_data:
    # Save the data to MongoDB collection
    classified_collection.insert_many(table_data)  # Save directly to MongoDB
    print(f"Data saved to MongoDB in 'payslip' collection.")

    # Optionally, save the DataFrame to a CSV as well
    output_path = r'C:\Users\namee\OneDrive\Desktop\infosys_project\text4.csv'  # Change path as needed
    df = pd.DataFrame(table_data)
    df.to_csv(output_path, index=False)

    print(f"Data also saved to '{output_path}'")
else:
    print("No data to save.")


# Sample data (replace with the file path if reading from a file)
data = pd.read_csv('classified_output.csv')
# Create a DataFrame
df = pd.DataFrame(data)

# Group data by 'Category' and calculate the total 'Price' per category
category_totals = df.groupby('Category')['Price'].sum()

# --- Pie Chart: Total Spent per Category ---
plt.figure(figsize=(8, 8))
dark_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']  # Dark and vibrant colors
plt.pie(
    category_totals,
    labels=category_totals.index,
    autopct='%1.1f%%',
    startangle=140,
    colors=dark_colors[:len(category_totals)]
)
plt.title('Total Spent Per Category', fontsize=14, color='#333')  # Dark title
plt.axis('equal')  # Equal aspect ratio ensures a circular pie chart
plt.show()

# --- Bar Chart: Total Price per Description ---
# Sort by total price for better visualization
sorted_df = df.sort_values(by='Price', ascending=False)

plt.figure(figsize=(10, 6))
plt.bar(
    sorted_df['Description'],
    sorted_df['Price'],
    color='#2c3e50',  # Dark slate color
    alpha=0.85,
    edgecolor='#34495e',  # Darker edge for bars
    linewidth=1.5
)
plt.xticks(rotation=45, ha='right', fontsize=10, color='#333')  # Rotate x-axis labels for readability
plt.xlabel('Description', fontsize=12, color='#333')  # Dark label
plt.ylabel('Total Price', fontsize=12, color='#333')  # Dark label
plt.title('Total Price per Description', fontsize=14, color='#333')  # Dark title
plt.grid(axis='y', linestyle='--', alpha=0.6)  # Subtle gridlines for better readability
plt.tight_layout()
plt.show()
