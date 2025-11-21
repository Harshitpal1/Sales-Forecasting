"""
Machine Learning Models Module for Sales Forecasting

This module provides implementations of various regression models for sales forecasting:
- Linear Regression (baseline)
- Random Forest Regressor
- Gradient Boosting Regressor
- XGBoost Regressor

It includes functionality for:
- Model training with cross-validation
- Hyperparameter tuning
- Model comparison
- Feature importance analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
import warnings
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import joblib
from datetime import datetime

warnings.filterwarnings('ignore')


class SalesForecaster:
    """
    Sales forecasting model trainer and predictor.
    Handles multiple models and provides comprehensive evaluation.
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the forecaster.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.models = {}
        self.scaler = StandardScaler()
        self.feature_names = None
        self.trained_models = {}
        self.training_history = {}
        
    def initialize_models(self) -> Dict[str, Any]:
        """
        Initialize all regression models.
        
        Returns:
            Dictionary of initialized models
        """
        self.models = {
            'Linear Regression': LinearRegression(),
            
            'Random Forest': RandomForestRegressor(
                n_estimators=100,
                max_depth=15,
                min_samples_split=10,
                min_samples_leaf=4,
                random_state=self.random_state,
                n_jobs=-1
            ),
            
            'Gradient Boosting': GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                min_samples_split=10,
                min_samples_leaf=4,
                subsample=0.8,
                random_state=self.random_state
            ),
            
            'XGBoost': xgb.XGBRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                min_child_weight=3,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_state,
                n_jobs=-1
            )
        }
        
        print(f"Initialized {len(self.models)} models")
        return self.models
    
    def prepare_data(self, df: pd.DataFrame, target_column: str,
                    feature_columns: List[str] = None,
                    test_size: float = 0.2) -> Tuple[np.ndarray, np.ndarray, 
                                                      np.ndarray, np.ndarray]:
        """
        Prepare data for training with proper time series split.
        
        Args:
            df: Input DataFrame
            target_column: Name of target column
            feature_columns: List of feature column names (if None, use all except target)
            test_size: Proportion of data for testing
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        # Select features
        if feature_columns is None:
            # Exclude target and date columns
            feature_columns = [col for col in df.columns 
                             if col != target_column and 
                             not pd.api.types.is_datetime64_any_dtype(df[col])]
        
        self.feature_names = feature_columns
        
        X = df[feature_columns].copy()
        y = df[target_column].copy()
        
        # Handle any remaining NaN values
        X = X.fillna(0)
        y = y.fillna(y.mean())
        
        # Time series split (no shuffling to maintain temporal order)
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        print(f"\nData prepared:")
        print(f"  Features: {len(feature_columns)}")
        print(f"  Training samples: {len(X_train)}")
        print(f"  Testing samples: {len(X_test)}")
        print(f"  Train date range: {split_idx} samples")
        
        return X_train, X_test, y_train, y_test
    
    def train_model(self, model_name: str, X_train: np.ndarray, 
                   y_train: np.ndarray, scale_features: bool = False) -> Any:
        """
        Train a specific model.
        
        Args:
            model_name: Name of the model to train
            X_train: Training features
            y_train: Training target
            scale_features: Whether to scale features (recommended for Linear Regression)
            
        Returns:
            Trained model
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found. Available: {list(self.models.keys())}")
        
        print(f"\nTraining {model_name}...")
        
        model = self.models[model_name]
        
        # Scale features if requested
        if scale_features and model_name == 'Linear Regression':
            X_train_processed = self.scaler.fit_transform(X_train)
        else:
            X_train_processed = X_train
        
        # Train model
        start_time = datetime.now()
        model.fit(X_train_processed, y_train)
        training_time = (datetime.now() - start_time).total_seconds()
        
        self.trained_models[model_name] = model
        self.training_history[model_name] = {
            'training_time': training_time,
            'scaled': scale_features
        }
        
        print(f"  Training completed in {training_time:.2f} seconds")
        
        return model
    
    def train_all_models(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
        """
        Train all initialized models.
        
        Args:
            X_train: Training features
            y_train: Training target
            
        Returns:
            Dictionary of trained models
        """
        print("\n" + "="*50)
        print("Training All Models")
        print("="*50)
        
        if not self.models:
            self.initialize_models()
        
        for model_name in self.models.keys():
            # Scale only for Linear Regression
            scale = (model_name == 'Linear Regression')
            self.train_model(model_name, X_train, y_train, scale_features=scale)
        
        print("\n" + "="*50)
        print("All Models Trained Successfully")
        print("="*50)
        
        return self.trained_models
    
    def predict(self, model_name: str, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using a trained model.
        
        Args:
            model_name: Name of the model to use
            X: Features for prediction
            
        Returns:
            Predictions array
        """
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} not trained yet")
        
        model = self.trained_models[model_name]
        
        # Scale if needed (Linear Regression)
        if model_name == 'Linear Regression' and self.training_history[model_name].get('scaled'):
            X_processed = self.scaler.transform(X)
        else:
            X_processed = X
        
        predictions = model.predict(X_processed)
        
        return predictions
    
    def cross_validate_model(self, model_name: str, X: np.ndarray, 
                           y: np.ndarray, cv_folds: int = 5) -> Dict[str, float]:
        """
        Perform time series cross-validation on a model.
        
        Args:
            model_name: Name of the model
            X: Features
            y: Target
            cv_folds: Number of cross-validation folds
            
        Returns:
            Dictionary with cross-validation scores
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.models[model_name]
        
        # Use TimeSeriesSplit for proper time series CV
        tscv = TimeSeriesSplit(n_splits=cv_folds)
        
        # Calculate cross-validation scores
        cv_scores = cross_val_score(
            model, X, y, 
            cv=tscv, 
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )
        
        # Convert to RMSE
        cv_rmse = np.sqrt(-cv_scores)
        
        results = {
            'mean_rmse': cv_rmse.mean(),
            'std_rmse': cv_rmse.std(),
            'scores': cv_rmse.tolist()
        }
        
        print(f"\n{model_name} Cross-Validation:")
        print(f"  Mean RMSE: {results['mean_rmse']:.2f} (+/- {results['std_rmse']:.2f})")
        
        return results
    
    def get_feature_importance(self, model_name: str, top_n: int = 20) -> pd.DataFrame:
        """
        Get feature importance for tree-based models.
        
        Args:
            model_name: Name of the model
            top_n: Number of top features to return
            
        Returns:
            DataFrame with feature importances
        """
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} not trained yet")
        
        model = self.trained_models[model_name]
        
        # Check if model has feature_importances_
        if not hasattr(model, 'feature_importances_'):
            print(f"{model_name} does not support feature importance")
            return None
        
        importances = model.feature_importances_
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': importances
        })
        
        # Sort by importance
        importance_df = importance_df.sort_values('Importance', ascending=False)
        
        # Return top N
        return importance_df.head(top_n)
    
    def save_model(self, model_name: str, filepath: str):
        """
        Save a trained model to disk.
        
        Args:
            model_name: Name of the model to save
            filepath: Path to save the model
        """
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} not trained yet")
        
        model = self.trained_models[model_name]
        joblib.dump(model, filepath)
        print(f"Model {model_name} saved to {filepath}")
    
    def load_model(self, model_name: str, filepath: str):
        """
        Load a trained model from disk.
        
        Args:
            model_name: Name to assign to the loaded model
            filepath: Path to load the model from
        """
        model = joblib.load(filepath)
        self.trained_models[model_name] = model
        print(f"Model loaded from {filepath} as {model_name}")
        
        return model
    
    def get_model_summary(self) -> pd.DataFrame:
        """
        Get summary of all trained models.
        
        Returns:
            DataFrame with model summary
        """
        if not self.trained_models:
            print("No models trained yet")
            return None
        
        summary_data = []
        for model_name in self.trained_models.keys():
            history = self.training_history.get(model_name, {})
            summary_data.append({
                'Model': model_name,
                'Training Time (s)': history.get('training_time', 'N/A'),
                'Feature Scaling': history.get('scaled', False)
            })
        
        summary_df = pd.DataFrame(summary_data)
        return summary_df


def demonstrate_models():
    """
    Demonstrate model training with sample data.
    """
    print("Sales Forecasting Models Demo")
    print("="*50)
    
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    n_features = 10
    
    X = np.random.randn(n_samples, n_features)
    y = (X[:, 0] * 10 + X[:, 1] * 5 + 
         np.random.randn(n_samples) * 2 + 100)
    
    # Create DataFrame for easier handling
    feature_names = [f'Feature_{i}' for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)
    df['Sales'] = y
    
    print(f"\nSample data created: {df.shape}")
    
    # Initialize forecaster
    forecaster = SalesForecaster(random_state=42)
    forecaster.initialize_models()
    
    # Prepare data
    X_train, X_test, y_train, y_test = forecaster.prepare_data(
        df, target_column='Sales', test_size=0.2
    )
    
    # Train all models
    forecaster.train_all_models(X_train, y_train)
    
    # Get model summary
    print("\n\nModel Summary:")
    print(forecaster.get_model_summary())
    
    # Feature importance for tree-based models
    print("\n\nFeature Importance (Random Forest):")
    importance_df = forecaster.get_feature_importance('Random Forest', top_n=5)
    if importance_df is not None:
        print(importance_df)


if __name__ == "__main__":
    demonstrate_models()
