"""
Time Series Modeling Pipeline for Bike Demand Prediction

This module provides automated time series modeling with sklearn Pipeline,
lag/rolling features, and temporal validation specifically designed for
sequential data with strong temporal dependencies.

Author: Financial Engineering Bootcamp Team
Date: 2024-08-24
"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')


class TimeSeriesFeatureTransformer(BaseEstimator, TransformerMixin):
    """
    Custom transformer for time series feature engineering.
    Creates lag features, rolling statistics, temporal features, and differencing.
    """
    
    def __init__(self, target_col='demand', lags=[1, 2, 3, 6], windows=[3, 6, 12]):
        self.target_col = target_col
        self.lags = lags
        self.windows = windows
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_new = X.copy()
        
        # Lag features - previous time periods
        for lag in self.lags:
            X_new[f'{self.target_col}_lag_{lag}'] = X_new[self.target_col].shift(lag)
        
        # Rolling features - moving averages and statistics
        for window in self.windows:
            X_new[f'{self.target_col}_rolling_{window}h'] = (
                X_new[self.target_col].rolling(window=window, min_periods=1).mean()
            )
            X_new[f'{self.target_col}_rolling_std_{window}h'] = (
                X_new[self.target_col].rolling(window=window, min_periods=1).std()
            )
            
            # Weather rolling features
            if 'temperature' in X_new.columns:
                X_new[f'temperature_rolling_{window}h'] = (
                    X_new['temperature'].rolling(window=window, min_periods=1).mean()
                )
            if 'humidity' in X_new.columns:
                X_new[f'humidity_rolling_{window}h'] = (
                    X_new['humidity'].rolling(window=window, min_periods=1).mean()
                )
        
        # Temporal features from datetime index
        if hasattr(X_new.index, 'hour'):
            X_new['hour_of_day'] = X_new.index.hour
            X_new['day_of_week'] = X_new.index.dayofweek
            
            # Cyclical encoding
            X_new['hour_sin'] = np.sin(2 * np.pi * X_new['hour_of_day'] / 24)
            X_new['hour_cos'] = np.cos(2 * np.pi * X_new['hour_of_day'] / 24)
            X_new['dow_sin'] = np.sin(2 * np.pi * X_new['day_of_week'] / 7)
            X_new['dow_cos'] = np.cos(2 * np.pi * X_new['day_of_week'] / 7)
            
            # Weekend indicator
            X_new['is_weekend'] = (X_new['day_of_week'] >= 5).astype(int)
        
        # Differencing for stationarity
        X_new[f'{self.target_col}_diff'] = X_new[self.target_col].diff()
        if 'temperature' in X_new.columns:
            X_new['temperature_diff'] = X_new['temperature'].diff()
        if 'humidity' in X_new.columns:
            X_new['humidity_diff'] = X_new['humidity'].diff()
        
        return X_new


class TimeSeriesModelPipeline:
    """
    Automated time series modeling pipeline with sklearn Pipeline integration.
    """
    
    def __init__(self, target_col='demand', random_state=42):
        self.target_col = target_col
        self.random_state = random_state
        self.pipelines = {}
        self.results = {}
        self.best_pipeline = None
        self.best_model_name = None
        
    def prepare_time_series_data(self, data, test_size=0.2):
        """
        Prepare time series data with proper temporal ordering.
        
        Args:
            data: DataFrame with datetime index
            test_size: Proportion for test set
        """
        # Ensure datetime index
        if not isinstance(data.index, pd.DatetimeIndex):
            if 'datetime' in data.columns:
                data = data.set_index('datetime')
            else:
                # Create datetime index from row numbers (hourly)
                data.index = pd.to_datetime('2024-01-01') + pd.to_timedelta(data.index, unit='h')
        
        # Apply time series feature engineering
        ts_transformer = TimeSeriesFeatureTransformer(target_col=self.target_col)
        data_transformed = ts_transformer.transform(data)
        
        # Remove NaN values created by lag and diff operations
        data_clean = data_transformed.dropna()
        
        # Separate features and target
        feature_cols = [col for col in data_clean.columns if col != self.target_col]
        X = data_clean[feature_cols]
        y = data_clean[self.target_col]
        
        # Time series split (maintain temporal order)
        split_idx = int((1 - test_size) * len(data_clean))
        
        self.X_train = X.iloc[:split_idx]
        self.X_test = X.iloc[split_idx:]
        self.y_train = y.iloc[:split_idx]
        self.y_test = y.iloc[split_idx:]
        
        # Feature selection - prioritize time series features
        self.selected_features = self._select_time_series_features(self.X_train, self.y_train)
        
        self.X_train_sel = self.X_train[self.selected_features]
        self.X_test_sel = self.X_test[self.selected_features]
        
        print(f"📊 Time Series Data Prepared:")
        print(f"   • Original shape: {data.shape}")
        print(f"   • After feature engineering: {data_transformed.shape}")
        print(f"   • After cleaning: {data_clean.shape}")
        print(f"   • Selected features: {len(self.selected_features)}")
        print(f"   • Train samples: {len(self.X_train)} | Test samples: {len(self.X_test)}")
        
    def _select_time_series_features(self, X, y, max_features=12):
        """Select most relevant features prioritizing time series features."""
        correlations = X.corrwith(y).abs().sort_values(ascending=False)
        
        # Prioritize time series features
        priority_features = []
        other_features = []
        
        for feature in correlations.index:
            if any(keyword in feature for keyword in ['lag', 'rolling', 'diff', 'hour', 'weekend']):
                priority_features.append(feature)
            else:
                other_features.append(feature)
        
        # Select balanced mix
        selected = (priority_features[:max_features//2] + 
                   other_features[:max_features//2])[:max_features]
        
        return selected
    
    def define_pipelines(self):
        """Define sklearn pipelines for different time series models."""
        self.pipelines = {
            'Linear_TS': Pipeline([
                ('scaler', StandardScaler()),
                ('model', LinearRegression())
            ]),
            
            'Ridge_TS': Pipeline([
                ('scaler', StandardScaler()),
                ('model', Ridge(alpha=1.0, random_state=self.random_state))
            ]),
            
            'Ridge_Strong': Pipeline([
                ('scaler', StandardScaler()),
                ('model', Ridge(alpha=10.0, random_state=self.random_state))
            ]),
            
            'RandomForest_TS': Pipeline([
                ('model', RandomForestRegressor(
                    n_estimators=50,
                    max_depth=8,
                    min_samples_split=5,
                    random_state=self.random_state
                ))
            ])
        }
    
    def train_pipelines(self):
        """Train all defined pipelines."""
        self.results = {}
        
        print("🤖 Training Time Series Pipelines:")
        
        for name, pipeline in self.pipelines.items():
            # Train pipeline
            pipeline.fit(self.X_train_sel, self.y_train)
            
            # Make predictions
            y_train_pred = pipeline.predict(self.X_train_sel)
            y_test_pred = pipeline.predict(self.X_test_sel)
            
            # Calculate metrics
            self.results[name] = {
                'train_r2': r2_score(self.y_train, y_train_pred),
                'test_r2': r2_score(self.y_test, y_test_pred),
                'test_rmse': np.sqrt(mean_squared_error(self.y_test, y_test_pred)),
                'test_mae': mean_absolute_error(self.y_test, y_test_pred),
                'predictions': y_test_pred,
                'pipeline': pipeline
            }
            
            print(f"   {name}: R²={self.results[name]['test_r2']:.3f}, "
                  f"RMSE={self.results[name]['test_rmse']:.2f}")
        
        # Identify best model
        self.best_model_name = max(self.results.keys(), key=lambda k: self.results[k]['test_r2'])
        self.best_pipeline = self.results[self.best_model_name]['pipeline']
        
        print(f"\n🏆 Best Time Series Model: {self.best_model_name}")
        print(f"   • Test R²: {self.results[self.best_model_name]['test_r2']:.3f}")
        print(f"   • Test RMSE: {self.results[self.best_model_name]['test_rmse']:.2f}")
    
    def time_series_cross_validation(self, n_splits=4):
        """Perform time series cross-validation."""
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        # Combine train and test for full time series CV
        X_full = pd.concat([self.X_train_sel, self.X_test_sel])
        y_full = pd.concat([self.y_train, self.y_test])
        
        cv_scores = []
        
        print(f"📊 Time Series Cross-Validation ({self.best_model_name}):")
        
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X_full), 1):
            # Split data
            X_cv_train = X_full.iloc[train_idx]
            X_cv_test = X_full.iloc[test_idx]
            y_cv_train = y_full.iloc[train_idx]
            y_cv_test = y_full.iloc[test_idx]
            
            # Train and predict
            cv_pipeline = Pipeline(self.best_pipeline.steps)
            cv_pipeline.fit(X_cv_train, y_cv_train)
            y_cv_pred = cv_pipeline.predict(X_cv_test)
            
            # Calculate score
            cv_score = r2_score(y_cv_test, y_cv_pred)
            cv_scores.append(cv_score)
            
            print(f"   Fold {fold}: R² = {cv_score:.3f}")
        
        cv_results = {
            'mean_score': np.mean(cv_scores),
            'std_score': np.std(cv_scores),
            'scores': cv_scores
        }
        
        print(f"\n📈 CV Results: {cv_results['mean_score']:.3f} ± {cv_results['std_score']:.3f}")
        return cv_results
    
    def get_performance_summary(self):
        """Get performance summary DataFrame."""
        return pd.DataFrame({
            'Model': list(self.results.keys()),
            'Test_R2': [self.results[m]['test_r2'] for m in self.results.keys()],
            'Test_RMSE': [self.results[m]['test_rmse'] for m in self.results.keys()],
            'Test_MAE': [self.results[m]['test_mae'] for m in self.results.keys()],
            'Overfitting': [self.results[m]['train_r2'] - self.results[m]['test_r2'] 
                           for m in self.results.keys()]
        }).sort_values('Test_R2', ascending=False)
    
    def get_residual_analysis(self):
        """Get residual analysis for best model."""
        best_predictions = self.results[self.best_model_name]['predictions']
        residuals = self.y_test - best_predictions
        
        return {
            'residuals': residuals,
            'predictions': best_predictions,
            'mean_residual': residuals.mean(),
            'std_residual': residuals.std(),
            'max_error': abs(residuals).max(),
            'percentile_95': np.percentile(abs(residuals), 95)
        }
    
    def predict_future(self, steps_ahead=6):
        """
        Make future predictions (simplified version).
        Note: This is a basic implementation. For production, would need
        more sophisticated handling of lag features.
        """
        if self.best_pipeline is None:
            raise ValueError("No trained pipeline available. Train models first.")
        
        # Use last known values for prediction
        last_features = self.X_test_sel.iloc[-1:].copy()
        
        predictions = []
        for step in range(steps_ahead):
            pred = self.best_pipeline.predict(last_features)[0]
            predictions.append(pred)
            
            # Update lag features (simplified)
            # In practice, would need more sophisticated lag feature updating
            
        return np.array(predictions)


# Example usage and testing
if __name__ == "__main__":
    # Load data
    data = pd.read_csv('../data/sample-data.csv')
    
    print("🧪 Testing Time Series Pipeline")
    print("=" * 50)
    
    # Initialize pipeline
    ts_pipeline = TimeSeriesModelPipeline(target_col='demand', random_state=42)
    
    # Prepare data
    ts_pipeline.prepare_time_series_data(data, test_size=0.2)
    
    # Define and train pipelines
    ts_pipeline.define_pipelines()
    ts_pipeline.train_pipelines()
    
    # Cross-validation
    cv_results = ts_pipeline.time_series_cross_validation(n_splits=4)
    
    # Performance summary
    performance = ts_pipeline.get_performance_summary()
    print("\n📊 Performance Summary:")
    print(performance)
    
    # Residual analysis
    residual_analysis = ts_pipeline.get_residual_analysis()
    print(f"\n📈 Residual Analysis:")
    print(f"   • Mean residual: {residual_analysis['mean_residual']:.6f}")
    print(f"   • Std residual: {residual_analysis['std_residual']:.3f}")
    print(f"   • Max error: {residual_analysis['max_error']:.2f}")
    print(f"   • 95th percentile error: {residual_analysis['percentile_95']:.2f}")
    
    # Future predictions
    try:
        future_preds = ts_pipeline.predict_future(steps_ahead=6)
        print(f"\n🔮 Future Predictions (next 6 hours): {future_preds.round(1)}")
    except Exception as e:
        print(f"\n⚠️ Future prediction error: {e}")
    
    print("\n✅ Time Series Pipeline test complete!")
