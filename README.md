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

## Stakeholder Context

See [/docs/stakeholder_memo.md](docs/stakeholder_memo.md).
