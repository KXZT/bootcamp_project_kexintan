# src/cleaning.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def preprocess_data(df):
    """
    Preprocess the coffee sales data.
    Returns cleaned DataFrame and preprocessing pipeline.
    """
    # Create a copy to avoid modifying original
    df_clean = df.copy()
    
    # Convert date to datetime and extract features
    df_clean['date'] = pd.to_datetime(df_clean['date'])
    df_clean['month'] = df_clean['date'].dt.month
    df_clean['day_of_month'] = df_clean['date'].dt.day
    
    # Define preprocessing pipeline
    numeric_features = ['average_temperature', 'marketing_spend', 'month', 'day_of_month']
    categorical_features = ['day_of_week', 'is_weekend', 'is_holiday']
    
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    return df_clean, preprocessor

def handle_missing_values(df):
    """Handle any missing values in the dataset."""
    # Check for missing values
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(f"Missing values found: {missing[missing > 0]}")
        # For this synthetic data, we'll just drop any missing rows
        df_clean = df.dropna()
        print(f"Dropped {len(df) - len(df_clean)} rows with missing values")
        return df_clean
    return df