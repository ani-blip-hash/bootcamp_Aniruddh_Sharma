"""
Data Cleaning Functions for Stock Market Data

This module provides reusable functions for cleaning and preprocessing stock market data,
including handling missing values, dropping incomplete records, and normalizing features.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import List, Optional, Union


def fill_missing_median(df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Fill missing values in specified columns with their median values.
    
    This function is particularly useful for numerical columns where the median
    provides a robust central tendency measure that's less sensitive to outliers
    than the mean.
    
    Args:
        df (pd.DataFrame): Input dataframe with potential missing values
        columns (Optional[List[str]]): List of column names to process. 
                                     If None, processes all numeric columns.
    
    Returns:
        pd.DataFrame: Dataframe with missing values filled with median values
        
    Assumptions:
        - Missing values are represented as NaN
        - Specified columns contain numeric data suitable for median calculation
        - Median is an appropriate imputation strategy for the data distribution
    """
    df_cleaned = df.copy()
    
    if columns is None:
        # Automatically detect numeric columns
        columns = df_cleaned.select_dtypes(include=[np.number]).columns.tolist()
    
    for col in columns:
        if col in df_cleaned.columns:
            if df_cleaned[col].isnull().any():
                median_value = df_cleaned[col].median()
                df_cleaned[col].fillna(median_value, inplace=True)
                print(f"Filled {df_cleaned[col].isnull().sum()} missing values in '{col}' with median: {median_value:.4f}")
    
    return df_cleaned


def drop_missing(df: pd.DataFrame, columns: Optional[List[str]] = None, 
                threshold: Optional[float] = None) -> pd.DataFrame:
    """
    Drop rows with missing values in specified columns.
    
    This function removes rows that have missing values in critical columns
    where imputation might not be appropriate or where complete data is required.
    
    Args:
        df (pd.DataFrame): Input dataframe
        columns (Optional[List[str]]): List of column names to check for missing values.
                                     If None, checks all columns.
        threshold (Optional[float]): If provided, drop rows where the proportion of 
                                   missing values exceeds this threshold (0.0 to 1.0).
    
    Returns:
        pd.DataFrame: Dataframe with rows containing missing values removed
        
    Assumptions:
        - Complete data is required for the specified columns
        - Dropping rows will not significantly reduce dataset size
        - Missing data is not informative (MCAR - Missing Completely at Random)
    """
    df_cleaned = df.copy()
    initial_rows = len(df_cleaned)
    
    if threshold is not None:
        # Drop rows based on missing value threshold
        missing_ratio = df_cleaned.isnull().sum(axis=1) / len(df_cleaned.columns)
        df_cleaned = df_cleaned[missing_ratio <= threshold]
        rows_dropped = initial_rows - len(df_cleaned)
        print(f"Dropped {rows_dropped} rows with missing value ratio > {threshold}")
    
    if columns is not None:
        # Drop rows with missing values in specific columns
        df_cleaned = df_cleaned.dropna(subset=columns)
        rows_dropped = initial_rows - len(df_cleaned)
        print(f"Dropped {rows_dropped} rows with missing values in specified columns: {columns}")
    else:
        # Drop rows with any missing values
        df_cleaned = df_cleaned.dropna()
        rows_dropped = initial_rows - len(df_cleaned)
        print(f"Dropped {rows_dropped} rows with any missing values")
    
    return df_cleaned


def normalize_data(df: pd.DataFrame, columns: Optional[List[str]] = None, 
                  method: str = 'standard') -> tuple[pd.DataFrame, object]:
    """
    Normalize numerical columns using specified scaling method.
    
    This function applies feature scaling to numerical columns to ensure
    all features contribute equally to model training and to improve
    convergence in optimization algorithms.
    
    Args:
        df (pd.DataFrame): Input dataframe
        columns (Optional[List[str]]): List of column names to normalize.
                                     If None, normalizes all numeric columns.
        method (str): Normalization method - 'standard' (z-score) or 'minmax' (0-1 scaling)
    
    Returns:
        tuple: (normalized_dataframe, fitted_scaler)
            - normalized_dataframe: Dataframe with normalized columns
            - fitted_scaler: Fitted scaler object for inverse transformation
            
    Assumptions:
        - Numerical columns follow approximately normal distribution (for standard scaling)
        - Features should be on similar scales for model performance
        - Original scale information may need to be preserved for interpretation
    """
    df_normalized = df.copy()
    
    if columns is None:
        # Automatically detect numeric columns, excluding ID-like columns
        numeric_cols = df_normalized.select_dtypes(include=[np.number]).columns.tolist()
        # Exclude columns that shouldn't be normalized (IDs, timestamps, etc.)
        exclude_patterns = ['id', 'timestamp', 'date', 'symbol']
        columns = [col for col in numeric_cols 
                  if not any(pattern in col.lower() for pattern in exclude_patterns)]
    
    # Initialize scaler based on method
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    else:
        raise ValueError("Method must be 'standard' or 'minmax'")
    
    # Apply normalization
    if columns:
        # Fit and transform the specified columns
        df_normalized[columns] = scaler.fit_transform(df_normalized[columns])
        print(f"Normalized {len(columns)} columns using {method} scaling: {columns}")
        
        # Print scaling statistics
        if method == 'standard':
            print("Scaling statistics (mean, std):")
            for i, col in enumerate(columns):
                print(f"  {col}: mean={scaler.mean_[i]:.4f}, std={scaler.scale_[i]:.4f}")
        elif method == 'minmax':
            print("Scaling statistics (min, max):")
            for i, col in enumerate(columns):
                print(f"  {col}: min={scaler.data_min_[i]:.4f}, max={scaler.data_max_[i]:.4f}")
    
    return df_normalized, scaler


def get_data_summary(df: pd.DataFrame) -> dict:
    """
    Generate a comprehensive summary of the dataset for documentation purposes.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        dict: Summary statistics including shape, missing values, data types
    """
    summary = {
        'shape': df.shape,
        'columns': df.columns.tolist(),
        'dtypes': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict(),
        'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
        'categorical_columns': df.select_dtypes(include=['object']).columns.tolist()
    }
    return summary


def print_cleaning_report(original_df: pd.DataFrame, cleaned_df: pd.DataFrame) -> None:
    """
    Print a comprehensive report comparing original and cleaned datasets.
    
    Args:
        original_df (pd.DataFrame): Original dataset before cleaning
        cleaned_df (pd.DataFrame): Dataset after cleaning operations
    """
    print("=" * 60)
    print("DATA CLEANING REPORT")
    print("=" * 60)
    
    print(f"\nOriginal Dataset:")
    print(f"  Shape: {original_df.shape}")
    print(f"  Missing values: {original_df.isnull().sum().sum()}")
    
    print(f"\nCleaned Dataset:")
    print(f"  Shape: {cleaned_df.shape}")
    print(f"  Missing values: {cleaned_df.isnull().sum().sum()}")
    
    rows_removed = original_df.shape[0] - cleaned_df.shape[0]
    cols_removed = original_df.shape[1] - cleaned_df.shape[1]
    
    print(f"\nChanges:")
    print(f"  Rows removed: {rows_removed} ({rows_removed/original_df.shape[0]*100:.1f}%)")
    print(f"  Columns removed: {cols_removed}")
    
    print(f"\nMissing Values by Column (Original vs Cleaned):")
    original_missing = original_df.isnull().sum()
    cleaned_missing = cleaned_df.isnull().sum()
    
    for col in original_df.columns:
        if col in cleaned_df.columns:
            orig_miss = original_missing[col]
            clean_miss = cleaned_missing[col]
            if orig_miss > 0 or clean_miss > 0:
                print(f"  {col}: {orig_miss} -> {clean_miss}")
    
    print("=" * 60)
