"""
Feature Engineering Module for Bike Demand Prediction

This module contains functions to create meaningful features from raw bike demand data
to improve model performance. Features are designed based on domain knowledge and 
insights from exploratory data analysis.

Author: Financial Engineering Bootcamp Team
Date: 2024-08-24
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


def create_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create temporal features from hour and day_of_week columns.
    
    Features created:
    - Hour categories (morning, afternoon, evening, night)
    - Weekend indicator
    - Cyclical encoding for hour (sin/cos transformation)
    - Rush hour indicators
    
    Args:
        df: DataFrame with 'hour' and 'day_of_week' columns
        
    Returns:
        DataFrame with additional temporal features
    """
    df = df.copy()
    
    # Hour categories based on typical bike usage patterns
    def categorize_hour(hour):
        if 6 <= hour <= 11:
            return 'morning'
        elif 12 <= hour <= 17:
            return 'afternoon'
        elif 18 <= hour <= 22:
            return 'evening'
        else:
            return 'night'
    
    df['hour_category'] = df['hour'].apply(categorize_hour)
    
    # Weekend indicator (assuming 6=Saturday, 0=Sunday in day_of_week)
    df['is_weekend'] = df['day_of_week'].isin([0, 6]).astype(int)
    
    # Cyclical encoding for hour (captures circular nature of time)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    
    # Rush hour indicators (morning: 7-9, evening: 17-19)
    df['is_morning_rush'] = df['hour'].between(7, 9).astype(int)
    df['is_evening_rush'] = df['hour'].between(17, 19).astype(int)
    
    # Work hours indicator (9-17)
    df['is_work_hours'] = df['hour'].between(9, 17).astype(int)
    
    return df


def create_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create weather-related features and interactions.
    
    Features created:
    - Temperature-humidity interaction
    - Weather comfort index
    - Temperature categories
    - Humidity categories
    - Weather extremes indicators
    
    Args:
        df: DataFrame with 'temperature' and 'humidity' columns
        
    Returns:
        DataFrame with additional weather features
    """
    df = df.copy()
    
    # Temperature-humidity interaction (discomfort increases with both)
    df['temp_humidity_interaction'] = df['temperature'] * df['humidity'] / 100
    
    # Weather comfort index (higher is more comfortable)
    # Based on temperature being optimal around 20-25°C and humidity around 40-60%
    temp_comfort = 1 - np.abs(df['temperature'] - 22.5) / 22.5
    humidity_comfort = 1 - np.abs(df['humidity'] - 50) / 50
    df['weather_comfort_index'] = (temp_comfort + humidity_comfort) / 2
    
    # Temperature categories
    def categorize_temperature(temp):
        if temp < 10:
            return 'cold'
        elif temp < 20:
            return 'cool'
        elif temp < 30:
            return 'warm'
        else:
            return 'hot'
    
    df['temperature_category'] = df['temperature'].apply(categorize_temperature)
    
    # Humidity categories
    def categorize_humidity(humidity):
        if humidity < 40:
            return 'low'
        elif humidity < 70:
            return 'moderate'
        else:
            return 'high'
    
    df['humidity_category'] = df['humidity'].apply(categorize_humidity)
    
    # Weather extremes (very hot/cold or very humid/dry)
    df['is_temp_extreme'] = ((df['temperature'] < 5) | (df['temperature'] > 35)).astype(int)
    df['is_humidity_extreme'] = ((df['humidity'] < 30) | (df['humidity'] > 80)).astype(int)
    
    # Ideal biking weather (moderate temp and humidity)
    df['is_ideal_weather'] = ((df['temperature'].between(15, 25)) & 
                              (df['humidity'].between(40, 65))).astype(int)
    
    return df


def create_derived_features(df: pd.DataFrame, target_col: str = 'demand') -> pd.DataFrame:
    """
    Create derived features including rolling statistics and lag features.
    
    Features created:
    - Rolling averages for demand
    - Lag features for demand
    - Rate of change features
    - Moving statistics
    
    Args:
        df: DataFrame with time series data
        target_col: Name of target column for lag features
        
    Returns:
        DataFrame with additional derived features
    """
    df = df.copy()
    df = df.sort_values('hour').reset_index(drop=True)  # Ensure proper ordering
    
    # Rolling averages for demand (3-hour and 6-hour windows)
    if len(df) >= 3:
        df['demand_rolling_3h'] = df[target_col].rolling(window=3, min_periods=1).mean()
        df['demand_rolling_std_3h'] = df[target_col].rolling(window=3, min_periods=1).std()
    
    if len(df) >= 6:
        df['demand_rolling_6h'] = df[target_col].rolling(window=6, min_periods=1).mean()
    
    # Lag features (previous observations)
    df['demand_lag_1'] = df[target_col].shift(1)
    df['demand_lag_2'] = df[target_col].shift(2)
    
    # Rate of change features
    df['temperature_change'] = df['temperature'].diff()
    df['humidity_change'] = df['humidity'].diff()
    df['demand_change'] = df[target_col].diff()
    
    # Moving statistics for weather
    if len(df) >= 3:
        df['temp_rolling_3h'] = df['temperature'].rolling(window=3, min_periods=1).mean()
        df['humidity_rolling_3h'] = df['humidity'].rolling(window=3, min_periods=1).mean()
    
    # Demand momentum (acceleration)
    df['demand_momentum'] = df['demand_change'].diff()
    
    # Fill NaN values created by lag and diff operations
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(method='bfill').fillna(method='ffill')
    
    return df


def create_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create interaction features between different variable types.
    
    Features created:
    - Hour-temperature interactions
    - Weekend-weather interactions
    - Complex feature combinations
    
    Args:
        df: DataFrame with engineered features
        
    Returns:
        DataFrame with interaction features
    """
    df = df.copy()
    
    # Hour-temperature interaction (temperature effect varies by time of day)
    df['hour_temp_interaction'] = df['hour'] * df['temperature']
    
    # Weekend-weather interaction (weather impact may differ on weekends)
    df['weekend_temp_interaction'] = df['is_weekend'] * df['temperature']
    df['weekend_humidity_interaction'] = df['is_weekend'] * df['humidity']
    
    # Rush hour weather interaction
    df['rush_hour_weather'] = (df['is_morning_rush'] + df['is_evening_rush']) * df['weather_comfort_index']
    
    # Temperature and time of day categories
    df['temp_time_combo'] = df['temperature_category'] + '_' + df['hour_category']
    
    return df


def encode_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode categorical features for machine learning models.
    
    Args:
        df: DataFrame with categorical features
        
    Returns:
        DataFrame with encoded categorical features
    """
    df = df.copy()
    
    # One-hot encode categorical features
    categorical_features = ['hour_category', 'temperature_category', 'humidity_category', 'temp_time_combo']
    
    for feature in categorical_features:
        if feature in df.columns:
            dummies = pd.get_dummies(df[feature], prefix=feature)
            df = pd.concat([df, dummies], axis=1)
            df = df.drop(columns=[feature])
    
    return df


def scale_features(df: pd.DataFrame, features_to_scale: List[str] = None) -> Tuple[pd.DataFrame, Dict]:
    """
    Scale numerical features using standardization.
    
    Args:
        df: DataFrame with features to scale
        features_to_scale: List of feature names to scale. If None, scales all numeric features.
        
    Returns:
        Tuple of (scaled DataFrame, scaling parameters dictionary)
    """
    df = df.copy()
    
    if features_to_scale is None:
        features_to_scale = df.select_dtypes(include=[np.number]).columns.tolist()
        # Exclude binary features and target variable
        features_to_scale = [col for col in features_to_scale 
                           if not col.startswith('is_') and col != 'demand']
    
    scaling_params = {}
    
    for feature in features_to_scale:
        if feature in df.columns:
            mean_val = df[feature].mean()
            std_val = df[feature].std()
            
            if std_val > 0:  # Avoid division by zero
                df[feature] = (df[feature] - mean_val) / std_val
                scaling_params[feature] = {'mean': mean_val, 'std': std_val}
    
    return df, scaling_params


def feature_engineering_pipeline(df: pd.DataFrame, target_col: str = 'demand') -> pd.DataFrame:
    """
    Complete feature engineering pipeline that applies all transformations.
    
    Args:
        df: Raw DataFrame with basic features
        target_col: Name of target column
        
    Returns:
        DataFrame with all engineered features
    """
    print("🔧 Starting Feature Engineering Pipeline...")
    
    # Step 1: Create temporal features
    print("   ⏰ Creating temporal features...")
    df = create_temporal_features(df)
    
    # Step 2: Create weather features
    print("   🌡️ Creating weather features...")
    df = create_weather_features(df)
    
    # Step 3: Create derived features
    print("   📈 Creating derived features...")
    df = create_derived_features(df, target_col)
    
    # Step 4: Create interaction features
    print("   🔗 Creating interaction features...")
    df = create_interaction_features(df)
    
    # Step 5: Encode categorical features
    print("   🏷️ Encoding categorical features...")
    df = encode_categorical_features(df)
    
    print(f"✅ Feature engineering complete! Shape: {df.shape}")
    print(f"   Original features: 5")
    print(f"   Engineered features: {df.shape[1] - 5}")
    print(f"   Total features: {df.shape[1]}")
    
    return df


def get_feature_importance_summary(df: pd.DataFrame, target_col: str = 'demand') -> pd.DataFrame:
    """
    Calculate basic feature importance using correlation with target.
    
    Args:
        df: DataFrame with engineered features
        target_col: Name of target column
        
    Returns:
        DataFrame with feature importance summary
    """
    numeric_features = df.select_dtypes(include=[np.number]).columns
    numeric_features = [col for col in numeric_features if col != target_col]
    
    correlations = []
    for feature in numeric_features:
        corr = abs(df[feature].corr(df[target_col]))
        correlations.append({'feature': feature, 'abs_correlation': corr})
    
    importance_df = pd.DataFrame(correlations)
    importance_df = importance_df.sort_values('abs_correlation', ascending=False)
    importance_df['importance_rank'] = range(1, len(importance_df) + 1)
    
    return importance_df


# Example usage and testing
if __name__ == "__main__":
    # Load sample data
    data = pd.read_csv('../data/sample-data.csv')
    
    print("🧪 Testing Feature Engineering Pipeline")
    print("=" * 50)
    print(f"Original data shape: {data.shape}")
    
    # Apply feature engineering
    engineered_data = feature_engineering_pipeline(data)
    
    # Get feature importance
    importance = get_feature_importance_summary(engineered_data)
    
    print("\n🏆 Top 10 Most Important Features:")
    print(importance.head(10))
    
    print(f"\n💾 Saving engineered features to '../data/processed/engineered_features.csv'")
    engineered_data.to_csv('../data/processed/engineered_features.csv', index=False)
    
    print("✅ Feature engineering pipeline test complete!")
