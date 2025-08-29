# src/utils.py
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv

load_dotenv()

def load_data():
    """
    Loads the raw coffee sales data from the path specified in .env file.
    Returns a pandas DataFrame.
    """
    data_path = os.getenv('DATA_PATH', '../data/raw/coffee_sales.csv')
    try:
        df = pd.read_csv(data_path)
        print(f"Successfully loaded data with {len(df)} rows and {len(df.columns)} columns")
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {data_path}")
        return None

def summarize_data(df):
    """
    Provides a basic summary of the DataFrame.
    """
    print("=== DATA SUMMARY ===")
    print(f"Shape: {df.shape}")
    print("\n=== COLUMN TYPES ===")
    print(df.dtypes)
    print("\n=== MISSING VALUES ===")
    print(df.isnull().sum())
    print("\n=== BASIC STATISTICS ===")
    print(df.describe())