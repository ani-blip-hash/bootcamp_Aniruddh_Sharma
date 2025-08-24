"""
Automated Modeling Pipeline for Bike Demand Prediction

This module provides automated regression modeling capabilities with multiple
model variations, feature selection, and comprehensive evaluation.

Author: Financial Engineering Bootcamp Team
Date: 2024-08-24
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


class RegressionModelPipeline:
    """
    Automated regression modeling pipeline with multiple model variations.
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.results = {}
        self.scaler = StandardScaler()
        self.best_model = None
        self.best_model_name = None
        
    def prepare_data(self, data, target_col='demand', test_size=0.2, 
                    remove_leakage=True, feature_selection=True, k_features=15):
        """
        Prepare data for modeling with feature selection and train-test split.
        
        Args:
            data: DataFrame with features and target
            target_col: Name of target column
            test_size: Proportion for test set
            remove_leakage: Whether to remove target leakage features
            feature_selection: Whether to apply feature selection
            k_features: Number of top features to select
        """
        # Separate target and features
        self.target_col = target_col
        y = data[target_col].copy()
        
        # Remove leakage features
        if remove_leakage:
            leakage_features = [
                target_col, f'{target_col}_rolling_3h', f'{target_col}_rolling_6h',
                f'{target_col}_lag_1', f'{target_col}_lag_2', f'{target_col}_change',
                f'{target_col}_momentum', f'{target_col}_rolling_std_3h'
            ]
            X = data.drop(columns=[col for col in leakage_features if col in data.columns])
        else:
            X = data.drop(columns=[target_col])
        
        # Handle missing values
        if X.isnull().sum().sum() > 0:
            X = X.fillna(X.median())
        
        # Feature selection
        if feature_selection and X.shape[1] > k_features:
            selector = SelectKBest(score_func=f_regression, k=k_features)
            X_selected = selector.fit_transform(X, y)
            self.selected_features = X.columns[selector.get_support()].tolist()
            X = X[self.selected_features]
        else:
            self.selected_features = X.columns.tolist()
        
        # Time-aware train-test split
        n_samples = len(X)
        train_size = int((1 - test_size) * n_samples)
        
        self.X_train = X.iloc[:train_size].copy()
        self.X_test = X.iloc[train_size:].copy()
        self.y_train = y.iloc[:train_size].copy()
        self.y_test = y.iloc[train_size:].copy()
        
        # Scale features
        self.X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(self.X_train),
            columns=self.X_train.columns,
            index=self.X_train.index
        )
        self.X_test_scaled = pd.DataFrame(
            self.scaler.transform(self.X_test),
            columns=self.X_test.columns,
            index=self.X_test.index
        )
        
        print(f"📊 Data prepared: {X.shape[1]} features, {len(self.X_train)} train, {len(self.X_test)} test")
        
    def define_models(self, include_ensemble=True):
        """Define regression models to train."""
        self.models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0, random_state=self.random_state),
            'Lasso Regression': Lasso(alpha=0.1, random_state=self.random_state),
            'Elastic Net': ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=self.random_state)
        }
        
        if include_ensemble:
            self.models.update({
                'Random Forest': RandomForestRegressor(
                    n_estimators=100, random_state=self.random_state, max_depth=10
                ),
                'Gradient Boosting': GradientBoostingRegressor(
                    n_estimators=100, random_state=self.random_state, max_depth=6
                )
            })
    
    def train_models(self):
        """Train all defined models."""
        self.results = {}
        self.trained_models = {}
        
        print("🤖 Training models...")
        
        for name, model in self.models.items():
            # Use scaled features for linear models, original for tree-based
            if any(tree_name in name for tree_name in ['Forest', 'Boosting']):
                X_train_model = self.X_train
                X_test_model = self.X_test
            else:
                X_train_model = self.X_train_scaled
                X_test_model = self.X_test_scaled
            
            # Train model
            model.fit(X_train_model, self.y_train)
            self.trained_models[name] = model
            
            # Make predictions
            y_train_pred = model.predict(X_train_model)
            y_test_pred = model.predict(X_test_model)
            
            # Calculate metrics
            self.results[name] = {
                'train_r2': r2_score(self.y_train, y_train_pred),
                'test_r2': r2_score(self.y_test, y_test_pred),
                'train_rmse': np.sqrt(mean_squared_error(self.y_train, y_train_pred)),
                'test_rmse': np.sqrt(mean_squared_error(self.y_test, y_test_pred)),
                'test_mae': mean_absolute_error(self.y_test, y_test_pred),
                'overfitting': r2_score(self.y_train, y_train_pred) - r2_score(self.y_test, y_test_pred),
                'predictions': y_test_pred
            }
            
            print(f"   ✅ {name}: R²={self.results[name]['test_r2']:.3f}")
        
        # Identify best model
        self.best_model_name = max(self.results.keys(), key=lambda k: self.results[k]['test_r2'])
        self.best_model = self.trained_models[self.best_model_name]
        
        print(f"🏆 Best model: {self.best_model_name} (R²={self.results[self.best_model_name]['test_r2']:.3f})")
    
    def get_performance_summary(self):
        """Get performance summary DataFrame."""
        return pd.DataFrame({
            'Model': list(self.results.keys()),
            'Test_R2': [self.results[m]['test_r2'] for m in self.results.keys()],
            'Test_RMSE': [self.results[m]['test_rmse'] for m in self.results.keys()],
            'Test_MAE': [self.results[m]['test_mae'] for m in self.results.keys()],
            'Overfitting': [self.results[m]['overfitting'] for m in self.results.keys()]
        }).sort_values('Test_R2', ascending=False)
    
    def get_feature_importance(self, model_name=None):
        """Get feature importance for specified model."""
        if model_name is None:
            model_name = self.best_model_name
        
        model = self.trained_models[model_name]
        
        if hasattr(model, 'coef_'):
            # Linear model coefficients
            return pd.DataFrame({
                'feature': self.selected_features,
                'importance': np.abs(model.coef_),
                'coefficient': model.coef_
            }).sort_values('importance', ascending=False)
        
        elif hasattr(model, 'feature_importances_'):
            # Tree-based model importances
            return pd.DataFrame({
                'feature': self.selected_features,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
        
        else:
            return None
    
    def validate_assumptions(self, model_name=None):
        """Validate regression assumptions for linear models."""
        if model_name is None:
            # Find best linear model
            linear_models = [m for m in self.results.keys() if 'Forest' not in m and 'Boosting' not in m]
            model_name = max(linear_models, key=lambda k: self.results[k]['test_r2'])
        
        predictions = self.results[model_name]['predictions']
        residuals = self.y_test - predictions
        
        # Statistical tests
        assumptions = {}
        
        # Normality test
        if len(residuals) <= 5000:
            stat, p_value = stats.shapiro(residuals)
            assumptions['normality'] = {
                'test': 'Shapiro-Wilk',
                'statistic': stat,
                'p_value': p_value,
                'normal': p_value > 0.05
            }
        else:
            stat, p_value = stats.jarque_bera(residuals)
            assumptions['normality'] = {
                'test': 'Jarque-Bera',
                'statistic': stat,
                'p_value': p_value,
                'normal': p_value > 0.05
            }
        
        # Homoscedasticity (constant variance)
        # Simple test: correlation between fitted values and absolute residuals
        fitted_abs_resid_corr = np.corrcoef(predictions, np.abs(residuals))[0, 1]
        assumptions['homoscedasticity'] = {
            'correlation': fitted_abs_resid_corr,
            'homoscedastic': abs(fitted_abs_resid_corr) < 0.3
        }
        
        # Independence (Durbin-Watson test approximation)
        residuals_diff = np.diff(residuals)
        dw_stat = np.sum(residuals_diff**2) / np.sum(residuals**2)
        assumptions['independence'] = {
            'durbin_watson': dw_stat,
            'independent': 1.5 < dw_stat < 2.5
        }
        
        return assumptions, residuals, predictions
    
    def cross_validate(self, cv_folds=5):
        """Perform cross-validation on best model."""
        if self.best_model is None:
            raise ValueError("No models trained yet. Call train_models() first.")
        
        # Use appropriate features for best model
        if any(tree_name in self.best_model_name for tree_name in ['Forest', 'Boosting']):
            X_cv = pd.concat([self.X_train, self.X_test])
        else:
            X_cv = pd.concat([self.X_train_scaled, self.X_test_scaled])
        
        y_cv = pd.concat([self.y_train, self.y_test])
        
        cv_scores = cross_val_score(self.best_model, X_cv, y_cv, 
                                   cv=cv_folds, scoring='r2')
        
        return {
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'cv_scores': cv_scores
        }


# Example usage and testing
if __name__ == "__main__":
    # Load data
    try:
        data = pd.read_csv('../data/processed/engineered_features.csv')
    except FileNotFoundError:
        print("Engineered features not found. Run feature engineering first.")
        exit()
    
    print("🧪 Testing Automated Modeling Pipeline")
    print("=" * 50)
    
    # Initialize pipeline
    pipeline = RegressionModelPipeline(random_state=42)
    
    # Prepare data
    pipeline.prepare_data(data, target_col='demand', test_size=0.2)
    
    # Define and train models
    pipeline.define_models(include_ensemble=True)
    pipeline.train_models()
    
    # Get results
    performance = pipeline.get_performance_summary()
    print("\n📊 Performance Summary:")
    print(performance)
    
    # Feature importance
    importance = pipeline.get_feature_importance()
    if importance is not None:
        print(f"\n🏆 Top 5 Features ({pipeline.best_model_name}):")
        print(importance.head())
    
    # Validate assumptions
    assumptions, residuals, predictions = pipeline.validate_assumptions()
    print(f"\n🔍 Regression Assumptions:")
    print(f"   • Normality: {'✅' if assumptions['normality']['normal'] else '❌'}")
    print(f"   • Homoscedasticity: {'✅' if assumptions['homoscedasticity']['homoscedastic'] else '❌'}")
    print(f"   • Independence: {'✅' if assumptions['independence']['independent'] else '❌'}")
    
    # Cross-validation
    cv_results = pipeline.cross_validate()
    print(f"\n📈 Cross-Validation (5-fold):")
    print(f"   • Mean R²: {cv_results['cv_mean']:.3f} ± {cv_results['cv_std']:.3f}")
    
    print("\n✅ Automated modeling pipeline test complete!")
