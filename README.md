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

### Engineered Features

#### 1. Momentum Features
- **Price Momentum**: `price_momentum_5`, `price_momentum_10`, `price_momentum_20` - Rate of change over different periods
- **Return Momentum**: `return_momentum_5`, `return_momentum_10`, `return_momentum_20` - Rolling average returns
- **RSI**: `rsi_14`, `rsi_30` - Relative Strength Index for overbought/oversold conditions
- **MACD**: `macd`, `macd_signal`, `macd_histogram` - Moving Average Convergence Divergence
- **Price Position**: `price_vs_sma20`, `price_vs_sma50` - Relative position to moving averages

#### 2. Volatility Features
- **Volatility Ratios**: `vol_ratio_5_20`, `vol_ratio_10_20` - Short vs long-term volatility
- **Volatility-Adjusted Returns**: `vol_adj_return_5`, `vol_adj_return_20` - Risk-adjusted performance
- **Bollinger Bands**: `bb_position` - Position within Bollinger Bands
- **Average True Range**: `atr`, `atr_ratio` - Volatility measure and normalization

#### 3. Volume Features
- **Volume Ratios**: `volume_ratio_5`, `volume_ratio_10`, `volume_ratio_20` - Current vs average volume
- **Volume Momentum**: `volume_momentum_5`, `volume_momentum_10` - Volume rate of change
- **Price-Volume**: `pv_trend` - Combined price and volume signal
- **On-Balance Volume**: `obv`, `obv_signal` - Cumulative volume indicator

#### 4. Cross-Asset Features
- **Market Metrics**: `market_return_mean`, `market_return_std`, `market_volatility` - Market-wide indicators
- **Relative Performance**: `relative_return`, `relative_volatility` - Asset vs market performance
- **Beta**: `beta_20` - 20-day rolling correlation with market
- **Market Regime**: `market_stress`, `market_direction` - Market condition indicators

#### 5. Lag Features
- **Return Lags**: `daily_return_lag_1`, `daily_return_lag_2`, `daily_return_lag_3`, `daily_return_lag_5`
- **Volatility Lags**: `volatility_20_lag_1`, `volatility_20_lag_2`, `volatility_20_lag_3`
- **Volume Lags**: `volume_ratio_20_lag_1`, `volume_ratio_20_lag_2`, `volume_ratio_20_lag_3`
- **Rolling Statistics**: `daily_return_roll_mean_3`, `daily_return_roll_std_7`, etc.

### Feature Rationale

**Momentum Features**: Capture trend-following and mean-reversion patterns in financial markets. RSI and MACD are widely used technical indicators that signal potential reversal points.

**Volatility Features**: Risk-adjusted returns normalize performance by volatility, crucial for portfolio optimization. Bollinger Bands identify overbought/oversold conditions.

**Volume Features**: Volume confirms price movements and indicates institutional interest. High volume with price movement suggests stronger trends.

**Cross-Asset Features**: Market-wide indicators help identify systematic risk and regime changes that affect all assets.

**Lag Features**: Essential for time series modeling, capturing temporal dependencies and autocorrelation in financial data.

---

## Outlier Analysis Documentation

### Outlier Detection Methods

#### 1. IQR Method
- **Threshold**: 1.5 × IQR (Interquartile Range)
- **Use Case**: General outlier detection, robust to distribution shape
- **Implementation**: `outliers.detect_outliers_iqr()`

#### 2. Z-Score Method
- **Threshold**: |z| > 3.0
- **Use Case**: Normally distributed data
- **Implementation**: `outliers.detect_outliers_zscore()`

#### 3. Modified Z-Score Method
- **Threshold**: |modified_z| > 3.5
- **Use Case**: Financial returns (more robust than standard Z-score)
- **Implementation**: `outliers.detect_outliers_modified_zscore()`

#### 4. Isolation Forest
- **Contamination**: 10% expected outliers
- **Use Case**: Multivariate outlier detection
- **Implementation**: `outliers.detect_outliers_isolation_forest()`

### Outlier Treatment Strategies

1. **Removal**: Complete removal of outlier observations
2. **Winsorization**: Capping extreme values at percentile thresholds
3. **Flagging**: Marking outliers without removal for analysis

### Risk Assessment

**Data Loss Risk**: IQR method typically removes 3-7% of data, Z-score removes 1-3%
**Bias Risk**: Outlier removal can shift mean returns by 0.1-0.5%
**Model Impact**: Outlier treatment generally improves model stability but may reduce R² by 0.01-0.03

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
