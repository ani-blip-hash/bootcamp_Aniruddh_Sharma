"""
Outlier Detection and Handling Module

This module provides reusable functions for detecting, flagging, and removing outliers
from financial datasets. Supports multiple detection methods and sensitivity analysis.

Author: Financial Engineering Bootcamp
Date: 2024
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Optional, Union
import warnings


def detect_outliers_iqr(data: pd.Series, multiplier: float = 1.5) -> pd.Series:
    """
    Detect outliers using Interquartile Range (IQR) method.
    
    Parameters:
    -----------
    data : pd.Series
        Input data series
    multiplier : float, default=1.5
        IQR multiplier for outlier threshold
        
    Returns:
    --------
    pd.Series
        Boolean series indicating outliers (True = outlier)
    """
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    return (data < lower_bound) | (data > upper_bound)


def detect_outliers_zscore(data: pd.Series, threshold: float = 3.0) -> pd.Series:
    """
    Detect outliers using Z-score method.
    
    Parameters:
    -----------
    data : pd.Series
        Input data series
    threshold : float, default=3.0
        Z-score threshold for outlier detection
        
    Returns:
    --------
    pd.Series
        Boolean series indicating outliers (True = outlier)
    """
    z_scores = np.abs(stats.zscore(data, nan_policy='omit'))
    return z_scores > threshold


def detect_outliers_modified_zscore(data: pd.Series, threshold: float = 3.5) -> pd.Series:
    """
    Detect outliers using Modified Z-score method (using median).
    More robust to outliers than standard Z-score.
    
    Parameters:
    -----------
    data : pd.Series
        Input data series
    threshold : float, default=3.5
        Modified Z-score threshold for outlier detection
        
    Returns:
    --------
    pd.Series
        Boolean series indicating outliers (True = outlier)
    """
    median = data.median()
    mad = np.median(np.abs(data - median))
    
    # Avoid division by zero
    if mad == 0:
        mad = np.std(data)
    
    modified_z_scores = 0.6745 * (data - median) / mad
    return np.abs(modified_z_scores) > threshold


def detect_outliers_isolation_forest(data: pd.DataFrame, 
                                    contamination: float = 0.1,
                                    random_state: int = 42) -> pd.Series:
    """
    Detect outliers using Isolation Forest method.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Input dataframe with features
    contamination : float, default=0.1
        Expected proportion of outliers
    random_state : int, default=42
        Random state for reproducibility
        
    Returns:
    --------
    pd.Series
        Boolean series indicating outliers (True = outlier)
    """
    try:
        from sklearn.ensemble import IsolationForest
        
        iso_forest = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1
        )
        
        # Handle missing values
        data_clean = data.fillna(data.median())
        
        outlier_labels = iso_forest.fit_predict(data_clean)
        return pd.Series(outlier_labels == -1, index=data.index)
        
    except ImportError:
        warnings.warn("sklearn not available. Using IQR method as fallback.")
        # Fallback to IQR on first numeric column
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            return detect_outliers_iqr(data[numeric_cols[0]])
        else:
            return pd.Series(False, index=data.index)


def winsorize_outliers(data: pd.Series, 
                      limits: Tuple[float, float] = (0.05, 0.05)) -> pd.Series:
    """
    Winsorize outliers by capping extreme values at specified percentiles.
    
    Parameters:
    -----------
    data : pd.Series
        Input data series
    limits : tuple, default=(0.05, 0.05)
        Lower and upper percentile limits for winsorization
        
    Returns:
    --------
    pd.Series
        Winsorized data series
    """
    try:
        from scipy.stats import mstats
        return pd.Series(mstats.winsorize(data, limits=limits), index=data.index)
    except ImportError:
        # Manual winsorization
        lower_limit = data.quantile(limits[0])
        upper_limit = data.quantile(1 - limits[1])
        return data.clip(lower=lower_limit, upper=upper_limit)


def remove_outliers(df: pd.DataFrame, 
                   columns: List[str],
                   method: str = 'iqr',
                   **kwargs) -> pd.DataFrame:
    """
    Remove outliers from specified columns using chosen method.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    columns : list
        List of column names to check for outliers
    method : str, default='iqr'
        Outlier detection method ('iqr', 'zscore', 'modified_zscore', 'isolation_forest')
    **kwargs
        Additional parameters for outlier detection methods
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with outliers removed
    """
    outlier_mask = pd.Series(False, index=df.index)
    
    for col in columns:
        if col not in df.columns:
            warnings.warn(f"Column '{col}' not found in dataframe")
            continue
            
        if method == 'iqr':
            col_outliers = detect_outliers_iqr(df[col], **kwargs)
        elif method == 'zscore':
            col_outliers = detect_outliers_zscore(df[col], **kwargs)
        elif method == 'modified_zscore':
            col_outliers = detect_outliers_modified_zscore(df[col], **kwargs)
        elif method == 'isolation_forest':
            col_outliers = detect_outliers_isolation_forest(df[[col]], **kwargs)
        else:
            raise ValueError(f"Unknown method: {method}")
            
        outlier_mask = outlier_mask | col_outliers
    
    return df[~outlier_mask].copy()


def flag_outliers(df: pd.DataFrame,
                 columns: List[str],
                 method: str = 'iqr',
                 flag_column: str = 'outlier_flag',
                 **kwargs) -> pd.DataFrame:
    """
    Flag outliers in dataframe without removing them.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    columns : list
        List of column names to check for outliers
    method : str, default='iqr'
        Outlier detection method
    flag_column : str, default='outlier_flag'
        Name of column to store outlier flags
    **kwargs
        Additional parameters for outlier detection methods
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with outlier flag column added
    """
    df_flagged = df.copy()
    outlier_mask = pd.Series(False, index=df.index)
    
    for col in columns:
        if col not in df.columns:
            warnings.warn(f"Column '{col}' not found in dataframe")
            continue
            
        if method == 'iqr':
            col_outliers = detect_outliers_iqr(df[col], **kwargs)
        elif method == 'zscore':
            col_outliers = detect_outliers_zscore(df[col], **kwargs)
        elif method == 'modified_zscore':
            col_outliers = detect_outliers_modified_zscore(df[col], **kwargs)
        elif method == 'isolation_forest':
            col_outliers = detect_outliers_isolation_forest(df[[col]], **kwargs)
        else:
            raise ValueError(f"Unknown method: {method}")
            
        outlier_mask = outlier_mask | col_outliers
    
    df_flagged[flag_column] = outlier_mask
    return df_flagged


def outlier_summary(df: pd.DataFrame, 
                   columns: List[str],
                   methods: List[str] = ['iqr', 'zscore', 'modified_zscore']) -> pd.DataFrame:
    """
    Generate summary of outliers detected by different methods.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    columns : list
        List of column names to analyze
    methods : list, default=['iqr', 'zscore', 'modified_zscore']
        List of outlier detection methods to compare
        
    Returns:
    --------
    pd.DataFrame
        Summary dataframe with outlier counts by method and column
    """
    summary_data = []
    
    for col in columns:
        if col not in df.columns:
            continue
            
        col_data = df[col].dropna()
        total_count = len(col_data)
        
        for method in methods:
            try:
                if method == 'iqr':
                    outliers = detect_outliers_iqr(col_data)
                elif method == 'zscore':
                    outliers = detect_outliers_zscore(col_data)
                elif method == 'modified_zscore':
                    outliers = detect_outliers_modified_zscore(col_data)
                elif method == 'isolation_forest':
                    outliers = detect_outliers_isolation_forest(df[[col]])
                else:
                    continue
                
                outlier_count = outliers.sum()
                outlier_pct = (outlier_count / total_count) * 100
                
                summary_data.append({
                    'Column': col,
                    'Method': method,
                    'Total_Count': total_count,
                    'Outlier_Count': outlier_count,
                    'Outlier_Percentage': outlier_pct
                })
                
            except Exception as e:
                warnings.warn(f"Error with method {method} on column {col}: {str(e)}")
                continue
    
    return pd.DataFrame(summary_data)


def sensitivity_analysis(df: pd.DataFrame,
                        target_column: str,
                        feature_columns: List[str],
                        outlier_methods: List[str] = ['iqr', 'zscore'],
                        analysis_func: callable = None) -> Dict:
    """
    Perform sensitivity analysis comparing results with and without outliers.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    target_column : str
        Name of target variable column
    feature_columns : list
        List of feature column names
    outlier_methods : list, default=['iqr', 'zscore']
        Outlier detection methods to test
    analysis_func : callable, optional
        Custom analysis function to apply (e.g., correlation, regression)
        
    Returns:
    --------
    dict
        Dictionary containing results for each outlier treatment
    """
    results = {}
    
    # Original data (no outlier removal)
    results['original'] = {
        'data_shape': df.shape,
        'target_stats': df[target_column].describe().to_dict(),
        'feature_stats': {col: df[col].describe().to_dict() for col in feature_columns}
    }
    
    if analysis_func:
        results['original']['custom_analysis'] = analysis_func(df)
    
    # Test each outlier method
    for method in outlier_methods:
        try:
            # Remove outliers
            df_clean = remove_outliers(df, feature_columns + [target_column], method=method)
            
            results[f'{method}_removed'] = {
                'data_shape': df_clean.shape,
                'rows_removed': df.shape[0] - df_clean.shape[0],
                'removal_percentage': ((df.shape[0] - df_clean.shape[0]) / df.shape[0]) * 100,
                'target_stats': df_clean[target_column].describe().to_dict(),
                'feature_stats': {col: df_clean[col].describe().to_dict() for col in feature_columns}
            }
            
            if analysis_func:
                results[f'{method}_removed']['custom_analysis'] = analysis_func(df_clean)
                
        except Exception as e:
            warnings.warn(f"Error in sensitivity analysis with method {method}: {str(e)}")
            continue
    
    return results


# Financial-specific outlier detection functions
def detect_return_outliers(returns: pd.Series, 
                          method: str = 'modified_zscore',
                          threshold: float = 3.5) -> pd.Series:
    """
    Detect outliers in financial return series.
    Uses modified Z-score by default as it's more robust for financial data.
    
    Parameters:
    -----------
    returns : pd.Series
        Return series
    method : str, default='modified_zscore'
        Detection method
    threshold : float, default=3.5
        Threshold for outlier detection
        
    Returns:
    --------
    pd.Series
        Boolean series indicating outliers
    """
    if method == 'modified_zscore':
        return detect_outliers_modified_zscore(returns, threshold)
    elif method == 'zscore':
        return detect_outliers_zscore(returns, threshold)
    elif method == 'iqr':
        return detect_outliers_iqr(returns, threshold)
    else:
        raise ValueError(f"Unknown method: {method}")


def detect_volume_outliers(volume: pd.Series, 
                          multiplier: float = 2.0) -> pd.Series:
    """
    Detect volume outliers using IQR method with higher multiplier.
    Volume data often has extreme values that are legitimate.
    
    Parameters:
    -----------
    volume : pd.Series
        Volume series
    multiplier : float, default=2.0
        IQR multiplier (higher than default to be less aggressive)
        
    Returns:
    --------
    pd.Series
        Boolean series indicating outliers
    """
    return detect_outliers_iqr(volume, multiplier=multiplier)


if __name__ == "__main__":
    # Example usage
    print("Outlier Detection Module")
    print("Available functions:")
    print("- detect_outliers_iqr()")
    print("- detect_outliers_zscore()")
    print("- detect_outliers_modified_zscore()")
    print("- remove_outliers()")
    print("- flag_outliers()")
    print("- outlier_summary()")
    print("- sensitivity_analysis()")
