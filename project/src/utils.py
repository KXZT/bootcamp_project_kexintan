# src/utils.py
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv

load_dotenv()

def get_data_path():
    """Get data path from environment variable with fallback."""
    return os.getenv('DATA_PATH', '../data/raw/coffee_sales.csv')

def load_data():
    """
    Loads the raw coffee sales data.
    Returns a pandas DataFrame.
    """
    data_path = get_data_path()
    try:
        df = pd.read_csv(data_path)
        print(f"Successfully loaded data from {data_path}")
        print(f"Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {data_path}")
        print("Please check your DATA_PATH in .env file or run generate_data.ipynb")
        return None

def save_processed_data(df, filename):
    """
    Save processed data to the processed folder.
    """
    processed_path = f'../data/processed/{filename}'
    df.to_csv(processed_path, index=False)
    print(f"Data saved to {processed_path}")