"""
Sales Forecasting Package

A comprehensive sales forecasting toolkit with:
- Data preprocessing and feature engineering
- Multiple regression models
- Model evaluation and visualization
"""

__version__ = "1.0.0"

from .data_preprocessing import SalesDataPreprocessor, create_sample_data
from .models import SalesForecaster
from .evaluation import ModelEvaluator

__all__ = [
    'SalesDataPreprocessor',
    'SalesForecaster',
    'ModelEvaluator',
    'create_sample_data'
]
