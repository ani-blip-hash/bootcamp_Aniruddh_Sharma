# Portfolio Risk Management System

# Intelligent Portfolio Risk Assessment and Optimization

## Project Scoping Paragraph

Institutional investors and portfolio managers face significant challenges in managing risk across diversified portfolios, especially during volatile market conditions. Traditional risk metrics often fail to capture complex interdependencies between assets, leading to unexpected losses and suboptimal portfolio performance. Current risk management systems lack real-time adaptability and fail to integrate alternative data sources that could provide early warning signals.

The primary stakeholders are **portfolio managers**, **risk officers**, and **institutional investors** who need actionable insights to optimize risk-adjusted returns. This project will create a **comprehensive risk management system** that provides real-time portfolio risk assessment, stress testing capabilities, and optimization recommendations. Outputs will be **risk metrics** (VaR, CVaR, Sharpe ratios), **stress test scenarios**, and an **interactive risk dashboard** for portfolio monitoring and decision-making.

---

## Goals → Lifecycle → Deliverables Mapping

| Goals                                     | Lifecycle Stage      | Deliverables                           |
| ----------------------------------------- | -------------------- | -------------------------------------- |
| Assess portfolio risk exposure            | Data Exploration     | Risk analysis report, correlation matrices |
| Build risk prediction models             | Modeling             | VaR/CVaR models, stress testing framework |
| Enable real-time risk monitoring         | Deployment           | Risk dashboard, automated alerts       |
| Optimize portfolio allocation             | Optimization         | Portfolio optimization engine          |

## Repository Structure

- /data/ → datasets
- /src/ → source code
- /notebooks/ → Jupyter notebooks
- /docs/ → documentation

---

## Data Cleaning Strategy

### Overview
Our data preprocessing pipeline implements a systematic approach to handle missing values, normalize features, and ensure data quality for downstream analysis and modeling.

### Cleaning Operations

#### 1. Missing Value Treatment
- **Technical Indicators**: Missing values in `volatility_20d`, `sma_20`, and `sma_50` are filled with median values
  - **Rationale**: These indicators require historical data windows (20/50 days) and are missing at the beginning of time series
  - **Method**: Median imputation preserves distribution characteristics while being robust to outliers

- **Core Metrics**: Missing values in `daily_return` are filled with median values
  - **Rationale**: Maintains continuity in return calculations while avoiding bias from extreme values

#### 2. Data Normalization
- **Price Data**: OHLC (Open, High, Low, Close) prices normalized using standard scaling (z-score)
- **Volume Data**: Trading volume normalized using standard scaling
- **Method**: `(x - μ) / σ` where μ is mean and σ is standard deviation
- **Benefits**: Ensures all features contribute equally to model training and improves convergence

#### 3. Data Quality Assurance
- **Completeness**: All critical columns (date, symbol, OHLCV) maintained without data loss
- **Consistency**: Standardized data types and formats across all features
- **Validation**: Zero missing values in final cleaned dataset

### Implementation
- **Modular Functions**: Reusable cleaning functions in `src/cleaning.py`
- **Documentation**: Comprehensive docstrings and assumption documentation
- **Reproducibility**: All operations parameterized and version-controlled

### Files
- **Raw Data**: `homework/data/raw/stock_data_raw.csv`
- **Processed Data**: `homework/data/processed/stock_data_cleaned.csv`
- **Cleaning Code**: `src/cleaning.py`
- **Analysis Notebook**: `homework/hw06_data_preprocessing.ipynb`

---

## Feature Engineering Documentation

### Overview
Comprehensive feature engineering pipeline for bike demand prediction, transforming raw data into meaningful predictors. The pipeline creates 45+ engineered features from 5 original features, improving model performance through domain knowledge and temporal patterns.

### Engineered Features

#### 1. Temporal Features
- **Hour Categories**: `hour_category` - Morning, afternoon, evening, night classifications
- **Weekend Indicator**: `is_weekend` - Binary flag for weekend vs weekday
- **Cyclical Encoding**: `hour_sin`, `hour_cos` - Circular time representation (24-hour cycle)
- **Rush Hours**: `is_morning_rush`, `is_evening_rush` - Peak commuting periods (7-9 AM, 5-7 PM)
- **Work Hours**: `is_work_hours` - Business hours indicator (9 AM - 5 PM)

#### 2. Weather Features
- **Temperature-Humidity Interaction**: `temp_humidity_interaction` - Combined comfort effect
- **Weather Comfort Index**: `weather_comfort_index` - Optimal biking conditions (temp: 20-25°C, humidity: 40-60%)
- **Temperature Categories**: `temperature_category` - Cold, cool, warm, hot classifications
- **Humidity Categories**: `humidity_category` - Low, moderate, high classifications
- **Weather Extremes**: `is_temp_extreme`, `is_humidity_extreme` - Adverse conditions flags
- **Ideal Weather**: `is_ideal_weather` - Perfect biking conditions indicator

#### 3. Derived Features (Time Series)
- **Rolling Averages**: `demand_rolling_3h`, `demand_rolling_6h` - Smoothed demand trends
- **Rolling Statistics**: `demand_rolling_std_3h` - Demand variability measures
- **Lag Features**: `demand_lag_1`, `demand_lag_2` - Previous hour demand values
- **Rate of Change**: `temperature_change`, `humidity_change`, `demand_change` - Momentum indicators
- **Weather Rolling**: `temp_rolling_3h`, `humidity_rolling_3h` - Smoothed weather patterns
- **Demand Momentum**: `demand_momentum` - Acceleration in demand changes

#### 4. Interaction Features
- **Hour-Temperature**: `hour_temp_interaction` - Time-dependent temperature effects
- **Weekend-Weather**: `weekend_temp_interaction`, `weekend_humidity_interaction` - Weekend weather patterns
- **Rush Hour Weather**: `rush_hour_weather` - Weather impact during peak hours
- **Complex Combinations**: `temp_time_combo` - Temperature-time category combinations

#### 5. Encoded Features
- **One-Hot Encoded**: Categorical features converted to binary indicators
- **Hour Categories**: `hour_category_afternoon`, `hour_category_evening`, etc.
- **Temperature Categories**: `temperature_category_cold`, `temperature_category_warm`, etc.
- **Humidity Categories**: `humidity_category_high`, `humidity_category_low`, etc.

### Feature Engineering Pipeline

The complete pipeline processes data through five stages:
1. **Temporal Features**: Time-based patterns and cyclical encoding
2. **Weather Features**: Environmental conditions and comfort indices
3. **Derived Features**: Rolling statistics, lags, and rate of change
4. **Interaction Features**: Complex relationships between variables
5. **Categorical Encoding**: One-hot encoding for machine learning compatibility

### Feature Importance Rankings

**Top 10 Most Predictive Features** (by correlation with demand):
1. `demand_rolling_3h` (0.xxx) - 3-hour rolling average demand
2. `temperature` (0.xxx) - Raw temperature reading
3. `temp_rolling_3h` (0.xxx) - 3-hour rolling average temperature
4. `humidity_rolling_3h` (0.xxx) - 3-hour rolling average humidity
5. `humidity` (0.xxx) - Raw humidity reading
6. `demand_rolling_6h` (0.xxx) - 6-hour rolling average demand
7. `demand_lag_1` (0.xxx) - Previous hour demand
8. `temp_humidity_interaction` (0.xxx) - Temperature-humidity interaction
9. `demand_lag_2` (0.xxx) - Two hours ago demand
10. `hour_temp_interaction` (0.xxx) - Hour-temperature interaction

### Domain Knowledge Rationale

**Temporal Features**: Bike sharing exhibits strong daily patterns with morning/evening commute peaks, weekend leisure usage, and seasonal variations. Cyclical encoding captures the circular nature of time.

**Weather Features**: Temperature and humidity significantly impact biking comfort. The comfort index combines both factors, while extreme weather conditions deter usage regardless of other factors.

**Derived Features**: Time series patterns in demand are crucial - previous hours predict current demand, while rolling averages smooth out noise and capture trends.

**Interaction Features**: Weather effects vary by time (temperature matters more during day hours) and usage patterns differ between weekdays and weekends.

### Implementation Files

- **Core Pipeline**: `src/feature_engineering.py` - Complete feature engineering functions
- **Notebook Analysis**: `notebooks/09_feature_engineering.ipynb` - Interactive feature exploration
- **Processed Data**: `data/processed/engineered_features.csv` - Final feature set
- **Feature Metadata**: `data/processed/feature_importance.csv` - Importance rankings

### Usage Examples

```python
# Complete pipeline
from src.feature_engineering import feature_engineering_pipeline
engineered_data = feature_engineering_pipeline(raw_data)

# Individual feature groups
from src.feature_engineering import create_temporal_features, create_weather_features
temporal_data = create_temporal_features(raw_data)
weather_data = create_weather_features(temporal_data)

# Feature importance analysis
from src.feature_engineering import get_feature_importance_summary
importance = get_feature_importance_summary(engineered_data)
```

### Performance Impact

- **Feature Count**: 5 → 50 features (45 engineered)
- **Data Quality**: No missing values, all numeric features
- **Memory Usage**: ~2KB for 56 observations
- **Processing Time**: <1 second for pipeline execution
- **Model Ready**: Scaled and encoded for ML algorithms

---

## Outlier Analysis Documentation

### Overview
Comprehensive outlier detection and handling framework for financial data analysis. Outliers can significantly impact statistical analysis and machine learning model performance, making their proper identification and treatment crucial for reliable results.

### Outlier Detection Methods

#### 1. IQR Method
- **Threshold**: 1.5 × IQR (Interquartile Range) - standard, 2.0 - conservative
- **Use Case**: Symmetric or near-symmetric distributions
- **Advantages**: Robust to distribution shape, easy to interpret
- **Implementation**: `outliers.detect_outliers_iqr(data, multiplier=1.5)`

#### 2. Z-Score Method
- **Threshold**: |z| > 3.0 (captures ~99.7% of normal distribution)
- **Use Case**: Large datasets with approximately normal distributions
- **Advantages**: Based on statistical theory, works well for normal distributions
- **Implementation**: `outliers.detect_outliers_zscore(data, threshold=3.0)`

#### 3. Modified Z-Score Method
- **Threshold**: |modified_z| > 3.5
- **Use Case**: Financial returns (more robust than standard Z-score)
- **Advantages**: Uses median and MAD, more robust to extreme outliers
- **Implementation**: `outliers.detect_outliers_modified_zscore(data, threshold=3.5)`

#### 4. Isolation Forest
- **Contamination**: 10% expected outliers (configurable)
- **Use Case**: High-dimensional data, complex outlier patterns
- **Advantages**: Handles multivariate outliers, no distribution assumptions
- **Implementation**: `outliers.detect_outliers_isolation_forest(data, contamination=0.1)`

### Outlier Treatment Strategies

1. **Removal**: Complete removal of outlier observations (`remove_outliers()`)
   - **When to use**: Clear data quality issues, measurement errors
   - **Risk**: Loss of information, reduced sample size

2. **Winsorization**: Capping extreme values at percentile thresholds (`winsorize_outliers()`)
   - **When to use**: Extreme values that may be legitimate but distort analysis
   - **Benefit**: Preserves sample size while reducing extreme influence

3. **Flagging**: Marking outliers without removal for analysis (`flag_outliers()`)
   - **When to use**: Uncertain about outlier legitimacy, need to preserve all data
   - **Benefit**: Maintains data integrity while enabling conditional analysis

### Financial Data Considerations

- **Return Data**: Modified Z-Score recommended (threshold: 3.5) - more robust to fat tails
- **Volume Data**: IQR with higher multiplier (2.0) - volume spikes often legitimate
- **Price Data**: IQR or Modified Z-Score - validate against corporate actions

### Sensitivity Analysis Framework

The `sensitivity_analysis()` function compares model performance with and without outliers:
- Tests multiple detection methods simultaneously
- Provides statistical summaries and custom analysis results
- Enables data-driven outlier treatment decisions

### Risk Assessment

**Data Loss Risk**: IQR method typically removes 3-7% of data, Z-score removes 1-3%
**Bias Risk**: Outlier removal can shift mean returns by 0.1-0.5%
**Model Impact**: Outlier treatment generally improves model stability but may reduce R² by 0.01-0.03

### Implementation Files

- **Core Functions**: `src/outliers.py` - Complete outlier detection and treatment toolkit
- **Sensitivity Analysis**: `notebooks/sensitivity_outliers.ipynb` - Comparative analysis framework
- **Documentation**: `docs/outliers.md` - Comprehensive assumptions and guidelines
- **Project Analysis**: `project/07_outlier_analysis.ipynb` - Project-specific outlier analysis

### Usage Examples

```python
# Basic outlier detection
import sys; sys.path.append('src')
import outliers
summary = outliers.outlier_summary(data, ['returns', 'volume'])

# Sensitivity analysis
results = outliers.sensitivity_analysis(
    data, target_column='returns', 
    feature_columns=['volume'], 
    outlier_methods=['iqr', 'modified_zscore']
)

# Treatment options
cleaned_data = outliers.remove_outliers(data, ['returns'], method='iqr')
flagged_data = outliers.flag_outliers(data, ['returns'], method='modified_zscore')
```

---

## Model Assumptions and Risks

### Linear Regression Assumptions
1. **Linearity**: Linear relationship between features and target
2. **Independence**: Residuals are independent (Durbin-Watson test)
3. **Homoscedasticity**: Constant variance of residuals
4. **Normality**: Residuals follow normal distribution
5. **No Multicollinearity**: Features are not perfectly correlated

### Time Series Model Assumptions
1. **Stationarity**: Statistical properties constant over time
2. **Temporal Dependencies**: Past values influence future values
3. **No Look-Ahead Bias**: Only historical data used for predictions
4. **Regime Stability**: Model parameters stable across market conditions

### Risk Factors
1. **Overfitting**: High complexity models may not generalize
2. **Regime Changes**: Market shifts can invalidate historical patterns
3. **Data Quality**: Missing or incorrect data affects model performance
4. **Feature Stability**: Engineered features may lose predictive power
5. **Survivorship Bias**: Analysis limited to currently active stocks

### Mitigation Strategies
- Time series cross-validation for temporal data
- Regular model retraining and validation
- Ensemble methods to reduce overfitting
- Robust feature engineering with domain knowledge
- Comprehensive backtesting across different market periods

---

## Stakeholder Context

See [/docs/stakeholder_memo.md](docs/stakeholder_memo.md).
