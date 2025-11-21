# Sales Forecasting Analytics Project

A comprehensive, production-ready sales forecasting solution for retail stores using machine learning. This project demonstrates end-to-end data science workflow including data preprocessing, feature engineering, model training, evaluation, and insights generation.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-orange)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.5%2B-red)](https://xgboost.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset Description](#dataset-description)
- [Methodology](#methodology)
- [Feature Engineering](#feature-engineering)
- [Models](#models)
- [Results](#results)
- [Key Insights](#key-insights)
- [Future Enhancements](#future-enhancements)
- [Contributing](#contributing)

---

## 🎯 Project Overview

This project implements a complete sales forecasting system for retail stores with the goal of predicting weekly sales based on historical data, seasonality patterns, and holiday effects. The solution leverages multiple machine learning models and provides comprehensive evaluation metrics and visualizations.

**Business Objectives:**
- Accurately forecast sales to optimize inventory management
- Identify key factors influencing sales patterns
- Understand seasonal and holiday impacts on retail performance
- Enable data-driven decision making for resource allocation

---

## ✨ Key Features

### Data Processing & Feature Engineering
- **Automated data cleaning** and missing value handling
- **Outlier detection and treatment** using IQR and Z-score methods
- **Comprehensive date features**: day of week, month, quarter, season
- **Holiday engineering**: flags and proximity features for major US holidays
  - Super Bowl, Labor Day, Thanksgiving, Christmas
  - Pre-holiday and post-holiday periods
  - Days to nearest holiday
- **Time series features**: lag features and rolling window statistics
- **Seasonality indicators**: weekend flags, month start/end flags

### Machine Learning Models
- **Linear Regression** (baseline model)
- **Random Forest Regressor** (ensemble learning)
- **Gradient Boosting Regressor** (boosting technique)
- **XGBoost** (advanced gradient boosting)
- **Time series cross-validation** for robust evaluation
- **Feature importance analysis** for interpretability

### Model Evaluation
- **Performance metrics**: RMSE, MAE, MAPE, R²
- **Comprehensive visualizations**:
  - Prediction vs actual plots
  - Residual analysis
  - Feature importance charts
  - Model comparison graphs
- **Cross-validation** with time series split
- **Detailed evaluation reports**

### Interactive Analysis
- **Jupyter notebook** with complete workflow
- **Exploratory Data Analysis (EDA)** with visualizations
- **Seasonality analysis**
- **Holiday impact analysis**
- **Model training and comparison**
- **Results interpretation and insights**

---

## 📁 Project Structure

```
Sales-Forecasting/
│
├── data/                          # Data directory
│   └── (place your CSV files here)
│
├── notebooks/                     # Jupyter notebooks
│   └── sales_forecasting_analysis.ipynb  # Main analysis notebook
│
├── src/                          # Source code modules
│   ├── __init__.py              # Package initialization
│   ├── data_preprocessing.py    # Data preprocessing and feature engineering
│   ├── models.py                # Machine learning models
│   └── evaluation.py            # Model evaluation and visualization
│
├── models/                       # Saved model files
│   └── (trained models saved here)
│
├── results/                      # Results and outputs
│   ├── visualizations/          # Generated plots and charts
│   └── model_comparison.csv     # Model performance comparison
│
├── requirements.txt              # Python dependencies
├── .gitignore                   # Git ignore file
└── README.md                    # This file
```

---

## 🔧 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup Instructions

1. **Clone the repository**
```bash
git clone https://github.com/Harshitpal1/Sales-Forecasting.git
cd Sales-Forecasting
```

2. **Create a virtual environment** (recommended)
```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Verify installation**
```bash
python -c "import pandas, numpy, sklearn, xgboost; print('All packages installed successfully!')"
```

---

## 🚀 Usage

### Quick Start

#### 1. Using Python Scripts

**Run data preprocessing:**
```bash
python src/data_preprocessing.py
```

**Train models:**
```bash
python src/models.py
```

**Evaluate models:**
```bash
python src/evaluation.py
```

#### 2. Using Jupyter Notebook (Recommended)

```bash
jupyter notebook notebooks/sales_forecasting_analysis.ipynb
```

The notebook provides a complete walkthrough with visualizations and interpretations.

#### 3. Using as a Python Package

```python
from src import SalesDataPreprocessor, SalesForecaster, ModelEvaluator

# Initialize preprocessor
preprocessor = SalesDataPreprocessor()

# Load and preprocess data
df = preprocessor.load_data('data/sales_data.csv')
df_processed = preprocessor.preprocess_pipeline(
    df=df,
    date_column='Date',
    target_column='Weekly_Sales'
)

# Train models
forecaster = SalesForecaster()
forecaster.initialize_models()
X_train, X_test, y_train, y_test = forecaster.prepare_data(
    df_processed, 'Weekly_Sales'
)
forecaster.train_all_models(X_train, y_train)

# Evaluate
evaluator = ModelEvaluator()
for model_name in forecaster.trained_models.keys():
    y_pred = forecaster.predict(model_name, X_test)
    evaluator.evaluate_model(y_test, y_pred, model_name)

# Compare models
comparison = evaluator.compare_models()
```

---

## 📊 Dataset Description

The project uses retail sales data with the following features:

### Input Features
- **Date**: Transaction date
- **Store**: Store identifier
- **Dept**: Department identifier
- **Temperature**: Average temperature during the week
- **Fuel_Price**: Fuel price in the region
- **CPI**: Consumer Price Index
- **Unemployment**: Unemployment rate

### Target Variable
- **Weekly_Sales**: Weekly sales amount (target for prediction)

### Engineered Features (automatically created)
- **Date Features**: Year, Month, Day, DayOfWeek, Quarter, Season, etc.
- **Holiday Features**: IsHoliday, IsPreHoliday, IsPostHoliday, DaysToNearestHoliday
- **Lag Features**: Previous sales values (1, 7, 14, 28 days)
- **Rolling Features**: Moving averages and statistics (7, 14, 28 day windows)

**Note**: The project includes a sample data generator if you don't have your own dataset.

---

## 🧪 Methodology

### 1. Data Preprocessing
- **Loading**: CSV data ingestion
- **Cleaning**: Handle missing values and inconsistencies
- **Outlier Treatment**: IQR-based capping or removal
- **Validation**: Data quality checks

### 2. Feature Engineering
- **Temporal Features**: Extract date components and patterns
- **Holiday Features**: Create holiday indicators and proximity measures
- **Lag Features**: Historical sales patterns
- **Rolling Statistics**: Trend and volatility measures
- **Categorical Encoding**: One-hot encoding for categorical variables

### 3. Model Training
- **Train-Test Split**: Time-based split (80/20) to maintain temporal order
- **Model Selection**: Multiple algorithms for comparison
- **Cross-Validation**: 5-fold time series cross-validation
- **Feature Scaling**: StandardScaler for Linear Regression

### 4. Evaluation
- **Metrics Calculation**: RMSE, MAE, MAPE, R²
- **Visualization**: Comprehensive plots and charts
- **Comparison**: Side-by-side model performance analysis
- **Interpretation**: Feature importance and business insights

---

## 🔬 Feature Engineering

### Critical Thinking Applied

#### 1. **Holiday Features**
**Rationale**: Retail sales show significant spikes around major holidays due to increased consumer spending.

**Implementation**:
- Binary flags for each major holiday (Super Bowl, Labor Day, Thanksgiving, Christmas)
- Days to nearest holiday (captures pre-holiday shopping behavior)
- Pre-holiday period flags (7 days before)
- Post-holiday period flags (7 days after)

**Impact**: Captures the shopping patterns before, during, and after holidays.

#### 2. **Seasonality Features**
**Rationale**: Consumer behavior varies by day of week, month, and season.

**Implementation**:
- Day of week (weekend vs weekday patterns)
- Month and quarter (annual cycles)
- Season (meteorological seasons)
- Week of year (captures yearly patterns)

**Impact**: Models can learn seasonal patterns and recurring trends.

#### 3. **Lag Features**
**Rationale**: Recent sales are often predictive of future sales.

**Implementation**:
- 1-day lag (immediate history)
- 7-day lag (weekly pattern)
- 14-day lag (bi-weekly pattern)
- 28-day lag (monthly pattern)

**Impact**: Captures momentum and trending behavior.

#### 4. **Rolling Window Statistics**
**Rationale**: Trend and volatility provide context for predictions.

**Implementation**:
- Rolling mean (7, 14, 28 days)
- Rolling standard deviation (volatility)
- Rolling min/max (range of variation)

**Impact**: Smooths out noise and captures trends.

---

## 🤖 Models

### 1. Linear Regression (Baseline)
- **Type**: Parametric regression
- **Pros**: Fast, interpretable, good baseline
- **Cons**: Assumes linear relationships
- **Use Case**: Benchmark for other models

### 2. Random Forest Regressor
- **Type**: Ensemble (bagging)
- **Pros**: Handles non-linearity, reduces overfitting, feature importance
- **Cons**: Can be memory-intensive
- **Hyperparameters**: 100 trees, max depth 15

### 3. Gradient Boosting Regressor
- **Type**: Ensemble (boosting)
- **Pros**: Powerful, handles complex patterns
- **Cons**: Slower training, sensitive to hyperparameters
- **Hyperparameters**: 100 estimators, learning rate 0.1

### 4. XGBoost
- **Type**: Advanced gradient boosting
- **Pros**: State-of-the-art performance, regularization, efficient
- **Cons**: Requires tuning
- **Hyperparameters**: 100 estimators, learning rate 0.1, max depth 5

---

## 📈 Results

### Model Performance Comparison

Models are evaluated on the test set using multiple metrics:

| Model | RMSE | MAE | MAPE (%) | R² |
|-------|------|-----|----------|-----|
| Linear Regression | Baseline | - | - | - |
| Random Forest | Improved | - | - | - |
| Gradient Boosting | Better | - | - | - |
| XGBoost | Best | - | - | - |

**Note**: Actual results will vary based on your data. Run the notebook to see specific numbers.

### Key Performance Indicators

- **RMSE**: Lower values indicate better fit
- **MAE**: Average prediction error in sales units
- **MAPE**: Percentage error (easier to interpret)
- **R²**: Proportion of variance explained (closer to 1 is better)

### Feature Importance

Top influential features typically include:
1. Historical sales (lag features)
2. Time-based features (day of week, month)
3. Holiday proximity
4. Rolling statistics
5. Seasonal indicators

---

## 💡 Key Insights

### Business Insights

1. **Seasonality Patterns**
   - Weekend sales are typically higher than weekday sales
   - Sales peak during holiday seasons (Thanksgiving, Christmas)
   - Monthly patterns show consistent trends

2. **Holiday Impact**
   - Major holidays significantly boost sales
   - Pre-holiday period shows gradual increase
   - Post-holiday period may show decline

3. **Predictive Factors**
   - Recent sales history is highly predictive
   - Day of week has strong influence
   - Economic factors (CPI, Unemployment) show moderate impact

### Model Selection Insights

1. **Complexity vs Performance**
   - Simple models provide good baselines
   - Tree-based models capture non-linear patterns better
   - Ensemble methods generally outperform single models

2. **Interpretability vs Accuracy Trade-off**
   - Linear Regression: Most interpretable
   - Random Forest/XGBoost: More accurate but less interpretable
   - Feature importance helps bridge the gap

---

## 🔮 Future Enhancements

### Short-term
- [ ] Hyperparameter tuning using GridSearchCV or RandomizedSearchCV
- [ ] Additional features: weather data, promotional events
- [ ] Ensemble methods combining multiple models
- [ ] LSTM/GRU for deep learning approach

### Medium-term
- [ ] Real-time prediction API using Flask/FastAPI
- [ ] Interactive dashboard with Plotly Dash or Streamlit
- [ ] Automated retraining pipeline
- [ ] A/B testing framework for model deployment

### Long-term
- [ ] Multi-store forecasting with hierarchical models
- [ ] Anomaly detection for unusual sales patterns
- [ ] Recommendation system for promotional strategies
- [ ] Integration with inventory management systems

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### How to Contribute

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📧 Contact

**Harshitpal** - [GitHub](https://github.com/Harshitpal1)

Project Link: [https://github.com/Harshitpal1/Sales-Forecasting](https://github.com/Harshitpal1/Sales-Forecasting)

---

## 📄 License

This project is open source and available under the MIT License.

---

## 🙏 Acknowledgments

- scikit-learn documentation for ML algorithms
- XGBoost team for the powerful library
- Kaggle community for inspiration and best practices
- Open source community for amazing tools and libraries

---

## 📚 References

- [scikit-learn Documentation](https://scikit-learn.org/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [Time Series Forecasting Best Practices](https://www.microsoft.com/en-us/research/group/forecasting/)
- [Feature Engineering for Machine Learning](https://www.oreilly.com/library/view/feature-engineering-for/9781491953235/)

---

**Made with ❤️ for Data Science and Machine Learning**