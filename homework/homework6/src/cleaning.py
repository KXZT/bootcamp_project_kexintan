import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def fill_missing_median(df, columns):
    df_copy = df.copy()
    for col in columns:
        if col in df_copy.columns:
            df_copy[col] = df_copy[col].fillna(df_copy[col].median())
    return df_copy

def drop_missing(df, threshold=0.5):
    df_copy = df.copy()
    missing_percent = df_copy.isnull().mean()
    columns_to_drop = missing_percent[missing_percent > threshold].index
    df_copy = df_copy.drop(columns=columns_to_drop)
    return df_copy

def normalize_data(df, columns):
    df_copy = df.copy()
    scaler = StandardScaler()
    
    for col in columns:
        if col in df_copy.columns:
            if not df_copy[col].dropna().empty:
                df_copy[col] = scaler.fit_transform(df_copy[[col]])
    
    return df_copy