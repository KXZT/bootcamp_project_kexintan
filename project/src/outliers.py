# src/outliers.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def detect_outliers_iqr(df, column, threshold=1.5):
    """
    Detect outliers using the Interquartile Range (IQR) method.
    Returns a DataFrame with outlier information.
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR
    
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    
    print(f"Outlier detection for '{column}':")
    print(f"Q1: {Q1:.2f}, Q3: {Q3:.2f}, IQR: {IQR:.2f}")
    print(f"Bounds: [{lower_bound:.2f}, {upper_bound:.2f}]")
    print(f"Found {len(outliers)} outliers ({len(outliers)/len(df)*100:.2f}% of data)")
    
    return outliers, lower_bound, upper_bound

def handle_outliers(df, column, method='clip', threshold=1.5):
    """
    Handle outliers using various methods.
    Methods: 'clip', 'remove', 'transform'
    """
    df_clean = df.copy()
    outliers, lower_bound, upper_bound = detect_outliers_iqr(df, column, threshold)
    
    if method == 'clip':
        df_clean[column] = df_clean[column].clip(lower_bound, upper_bound)
        print(f"Clipped {len(outliers)} outliers to bounds")
    
    elif method == 'remove':
        df_clean = df_clean[~df_clean.index.isin(outliers.index)]
        print(f"Removed {len(outliers)} outlier rows")
    
    elif method == 'transform':
        if df_clean[column].min() > 0:
            df_clean[column] = np.log1p(df_clean[column])
            print("Applied log transformation")
        else:
            print("Data contains non-positive values, cannot apply log transform")
    
    return df_clean

def plot_outlier_analysis(df, column):
    """Create visualization for outlier analysis."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    axes[0, 0].hist(df[column], bins=30, alpha=0.7, edgecolor='black')
    axes[0, 0].set_title(f'Original {column} Distribution')
    axes[0, 0].set_xlabel(column)
    axes[0, 0].set_ylabel('Frequency')
    
    axes[0, 1].boxplot(df[column])
    axes[0, 1].set_title(f'{column} Boxplot')
    axes[0, 1].set_ylabel(column)
    
    outliers, lower_bound, upper_bound = detect_outliers_iqr(df, column)
    
    axes[1, 0].scatter(range(len(df)), df[column], alpha=0.6, label='Normal')
    axes[1, 0].scatter(outliers.index, outliers[column], color='red', alpha=0.8, label='Outliers')
    axes[1, 0].axhline(y=upper_bound, color='r', linestyle='--', alpha=0.7, label='Upper Bound')
    axes[1, 0].axhline(y=lower_bound, color='r', linestyle='--', alpha=0.7, label='Lower Bound')
    axes[1, 0].set_title(f'{column} - Outlier Detection')
    axes[1, 0].set_xlabel('Index')
    axes[1, 0].set_ylabel(column)
    axes[1, 0].legend()
    
    from scipy import stats
    stats.probplot(df[column], dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title(f'Q-Q Plot of {column}')
    
    plt.tight_layout()
    return fig