# bootcamp_Aniruddh_Sharma

# Optimizing Bike-Sharing Demand in New York City

## Project Scoping Paragraph

New York City's public bike-sharing system experiences fluctuating demand across neighborhoods and times of day. This causes shortages in high-demand areas and surpluses in low-demand areas, reducing efficiency and user satisfaction.

The primary stakeholder is CitiBike's operations team. This project will create a **predictive model** that forecasts demand for the next 24 hours. Outputs will be a **metric** (predicted number of bikes needed per station) and an **artifact** (interactive dashboard).

---

## Goals → Lifecycle → Deliverables Mapping

| Goals                                     | Lifecycle Stage  | Deliverables                      |
| ----------------------------------------- | ---------------- | --------------------------------- |
| Understand demand patterns                | Data Exploration | EDA report, visualizations        |
| Predict hourly demand per station         | Modeling         | Trained machine learning model    |
| Enable operations to plan redistributions | Deployment       | Interactive dashboard + model API |

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

## Stakeholder Context

See [/docs/stakeholder_memo.md](docs/stakeholder_memo.md).
