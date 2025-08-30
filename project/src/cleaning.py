# src/cleaning.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

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

def create_features(df):
    """
    Create additional features for the coffee sales data.
    """
    df_featured = df.copy()
    
    df_featured['date'] = pd.to_datetime(df_featured['date'])
    df_featured['month'] = df_featured['date'].dt.month
    df_featured['day_of_month'] = df_featured['date'].dt.day
    df_featured['day_of_year'] = df_featured['date'].dt.dayofyear
    df_featured['week_of_year'] = df_featured['date'].dt.isocalendar().week
    df_featured['is_month_start'] = df_featured['date'].dt.is_month_start.astype(int)
    df_featured['is_month_end'] = df_featured['date'].dt.is_month_end.astype(int)
    

    df_featured['season'] = df_featured['month'].apply(lambda x: 
        'Winter' if x in [12, 1, 2] else
        'Spring' if x in [3, 4, 5] else
        'Summer' if x in [6, 7, 8] else 'Fall')
    

    df_featured['temp_category'] = pd.cut(df_featured['average_temperature'],
                                         bins=[0, 55, 65, 75, 100],
                                         labels=['Cold', 'Cool', 'Warm', 'Hot'])

    df_featured['marketing_efficiency'] = df_featured['revenue'] / (df_featured['marketing_spend'] + 1)
    
    df_featured['prev_day_revenue'] = df_featured['revenue'].shift(1)
    df_featured['revenue_3day_avg'] = df_featured['revenue'].rolling(window=3).mean()
    
    df_featured = df_featured.dropna()
    
    return df_featured

def get_feature_names(preprocessor, df):
    """Get feature names after preprocessing."""
    # Numeric features
    numeric_features = ['average_temperature', 'marketing_spend', 'month', 
                       'day_of_month', 'day_of_year', 'week_of_year',
                       'prev_day_revenue', 'revenue_3day_avg', 'marketing_efficiency']

    categorical_features = ['day_of_week', 'is_weekend', 'is_holiday', 
                           'season', 'temp_category', 'is_month_start', 'is_month_end']
    
    categorical_transformer = preprocessor.named_transformers_['cat']
    if hasattr(categorical_transformer, 'get_feature_names_out'):
        cat_features = categorical_transformer.get_feature_names_out(categorical_features)
    else:
        cat_features = []
    
    return numeric_features + list(cat_features)