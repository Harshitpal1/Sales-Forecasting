"""
Data Preprocessing and Feature Engineering Module for Sales Forecasting

This module provides functionality for:
- Data loading and cleaning
- Missing value handling
- Outlier detection and treatment
- Feature engineering including date-based features, holidays, and seasonality
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Tuple, List, Dict
import warnings

warnings.filterwarnings('ignore')


class SalesDataPreprocessor:
    """
    Comprehensive data preprocessing and feature engineering for retail sales data.
    """
    
    def __init__(self):
        """Initialize the preprocessor with holiday dates."""
        self.holidays = self._define_holidays()
        
    def _define_holidays(self) -> Dict[str, List[str]]:
        """
        Define major US holidays that impact retail sales.
        
        Returns:
            Dict mapping holiday names to their dates for multiple years
        """
        holidays = {
            'Super Bowl': [
                '2010-02-07', '2011-02-06', '2012-02-05', '2013-02-03',
                '2014-02-02', '2015-02-01', '2016-02-07', '2017-02-05',
                '2018-02-04', '2019-02-03', '2020-02-02', '2021-02-07',
                '2022-02-13', '2023-02-12'
            ],
            'Labor Day': [
                '2010-09-06', '2011-09-05', '2012-09-03', '2013-09-02',
                '2014-09-01', '2015-09-07', '2016-09-05', '2017-09-04',
                '2018-09-03', '2019-09-02', '2020-09-07', '2021-09-06',
                '2022-09-05', '2023-09-04'
            ],
            'Thanksgiving': [
                '2010-11-25', '2011-11-24', '2012-11-22', '2013-11-28',
                '2014-11-27', '2015-11-26', '2016-11-24', '2017-11-23',
                '2018-11-22', '2019-11-28', '2020-11-26', '2021-11-25',
                '2022-11-24', '2023-11-23'
            ],
            'Christmas': [
                '2010-12-25', '2011-12-25', '2012-12-25', '2013-12-25',
                '2014-12-25', '2015-12-25', '2016-12-25', '2017-12-25',
                '2018-12-25', '2019-12-25', '2020-12-25', '2021-12-25',
                '2022-12-25', '2023-12-25'
            ]
        }
        return holidays
    
    def load_data(self, filepath: str) -> pd.DataFrame:
        """
        Load sales data from CSV file.
        
        Args:
            filepath: Path to the CSV file
            
        Returns:
            DataFrame with loaded data
        """
        try:
            df = pd.read_csv(filepath)
            print(f"Data loaded successfully. Shape: {df.shape}")
            return df
        except Exception as e:
            print(f"Error loading data: {e}")
            raise
    
    def handle_missing_values(self, df: pd.DataFrame, strategy: str = 'forward_fill') -> pd.DataFrame:
        """
        Handle missing values in the dataset.
        
        Args:
            df: Input DataFrame
            strategy: Strategy for handling missing values
                     ('forward_fill', 'backward_fill', 'interpolate', 'drop')
            
        Returns:
            DataFrame with missing values handled
        """
        df_copy = df.copy()
        
        # Report missing values
        missing_counts = df_copy.isnull().sum()
        if missing_counts.sum() > 0:
            print("\nMissing values found:")
            print(missing_counts[missing_counts > 0])
        
        if strategy == 'forward_fill':
            df_copy = df_copy.ffill()
        elif strategy == 'backward_fill':
            df_copy = df_copy.bfill()
        elif strategy == 'interpolate':
            numeric_cols = df_copy.select_dtypes(include=[np.number]).columns
            df_copy[numeric_cols] = df_copy[numeric_cols].interpolate(method='linear')
        elif strategy == 'drop':
            df_copy = df_copy.dropna()
        
        # Fill any remaining NaN with 0 for numeric columns
        numeric_cols = df_copy.select_dtypes(include=[np.number]).columns
        df_copy[numeric_cols] = df_copy[numeric_cols].fillna(0)
        
        return df_copy
    
    def detect_outliers(self, df: pd.DataFrame, column: str, method: str = 'iqr', 
                       threshold: float = 1.5) -> pd.Series:
        """
        Detect outliers in a specific column.
        
        Args:
            df: Input DataFrame
            column: Column name to check for outliers
            method: Method for outlier detection ('iqr' or 'zscore')
            threshold: Threshold value (1.5 for IQR, 3 for z-score typically)
            
        Returns:
            Boolean Series indicating outliers
        """
        if method == 'iqr':
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            outliers = (df[column] < lower_bound) | (df[column] > upper_bound)
        elif method == 'zscore':
            z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
            outliers = z_scores > threshold
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return outliers
    
    def handle_outliers(self, df: pd.DataFrame, column: str, method: str = 'cap',
                       detection_method: str = 'iqr') -> pd.DataFrame:
        """
        Handle outliers in a specific column.
        
        Args:
            df: Input DataFrame
            column: Column name to handle outliers
            method: Method for handling ('cap', 'remove', 'transform')
            detection_method: Method for detecting outliers
            
        Returns:
            DataFrame with outliers handled
        """
        df_copy = df.copy()
        outliers = self.detect_outliers(df_copy, column, detection_method)
        
        num_outliers = outliers.sum()
        if num_outliers > 0:
            print(f"Found {num_outliers} outliers in {column}")
            
            if method == 'cap':
                # Cap outliers at percentile boundaries
                lower = df_copy[column].quantile(0.01)
                upper = df_copy[column].quantile(0.99)
                df_copy[column] = df_copy[column].clip(lower, upper)
            elif method == 'remove':
                df_copy = df_copy[~outliers]
            elif method == 'transform':
                # Log transform to reduce outlier impact
                df_copy[column] = np.log1p(df_copy[column])
        
        return df_copy
    
    def create_date_features(self, df: pd.DataFrame, date_column: str) -> pd.DataFrame:
        """
        Create comprehensive date-related features.
        
        Args:
            df: Input DataFrame
            date_column: Name of the date column
            
        Returns:
            DataFrame with additional date features
        """
        df_copy = df.copy()
        
        # Ensure date column is datetime
        df_copy[date_column] = pd.to_datetime(df_copy[date_column])
        
        # Basic date features
        df_copy['Year'] = df_copy[date_column].dt.year
        df_copy['Month'] = df_copy[date_column].dt.month
        df_copy['Day'] = df_copy[date_column].dt.day
        df_copy['DayOfWeek'] = df_copy[date_column].dt.dayofweek  # Monday=0, Sunday=6
        df_copy['DayOfYear'] = df_copy[date_column].dt.dayofyear
        df_copy['WeekOfYear'] = df_copy[date_column].dt.isocalendar().week.astype(int)
        df_copy['Quarter'] = df_copy[date_column].dt.quarter
        
        # Weekend flag
        df_copy['IsWeekend'] = (df_copy['DayOfWeek'] >= 5).astype(int)
        
        # Month start/end flags
        df_copy['IsMonthStart'] = df_copy[date_column].dt.is_month_start.astype(int)
        df_copy['IsMonthEnd'] = df_copy[date_column].dt.is_month_end.astype(int)
        
        # Season (meteorological)
        df_copy['Season'] = df_copy['Month'].apply(self._get_season)
        
        print(f"Created {len([c for c in df_copy.columns if c not in df.columns])} date features")
        
        return df_copy
    
    def _get_season(self, month: int) -> str:
        """
        Get season based on month.
        
        Args:
            month: Month number (1-12)
            
        Returns:
            Season name
        """
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        else:
            return 'Fall'
    
    def create_holiday_features(self, df: pd.DataFrame, date_column: str) -> pd.DataFrame:
        """
        Create holiday-related features including holiday flags and days to/from holidays.
        
        Args:
            df: Input DataFrame
            date_column: Name of the date column
            
        Returns:
            DataFrame with holiday features
        """
        df_copy = df.copy()
        
        # Ensure date column is datetime
        df_copy[date_column] = pd.to_datetime(df_copy[date_column])
        
        # Initialize holiday flag columns
        for holiday_name in self.holidays.keys():
            df_copy[f'Is{holiday_name.replace(" ", "")}'] = 0
            df_copy[f'DaysTo{holiday_name.replace(" ", "")}'] = 999  # Large number for days far from holiday
        
        # Create holiday features
        for holiday_name, holiday_dates in self.holidays.items():
            holiday_name_clean = holiday_name.replace(" ", "")
            holiday_dates_dt = pd.to_datetime(holiday_dates)
            
            for idx, row_date in enumerate(df_copy[date_column]):
                # Check if it's a holiday
                if row_date in holiday_dates_dt.values:
                    df_copy.loc[idx, f'Is{holiday_name_clean}'] = 1
                
                # Calculate days to nearest holiday
                days_diff = [(row_date - hd).days for hd in holiday_dates_dt]
                
                # Find nearest upcoming holiday (negative = past, positive = future)
                future_holidays = [d for d in days_diff if d <= 0]
                if future_holidays:
                    df_copy.loc[idx, f'DaysTo{holiday_name_clean}'] = max(future_holidays)
                else:
                    # If no upcoming holiday this year, use the closest one
                    df_copy.loc[idx, f'DaysTo{holiday_name_clean}'] = min(days_diff, key=abs)
        
        # Create general holiday flag (any holiday)
        holiday_cols = [col for col in df_copy.columns if col.startswith('Is') and 
                       any(h.replace(" ", "") in col for h in self.holidays.keys())]
        df_copy['IsHoliday'] = df_copy[holiday_cols].max(axis=1)
        
        # Create proximity to any holiday feature
        days_to_cols = [col for col in df_copy.columns if col.startswith('DaysTo')]
        df_copy['DaysToNearestHoliday'] = df_copy[days_to_cols].apply(
            lambda x: min(x, key=lambda y: abs(y)), axis=1
        )
        
        # Create pre-holiday flag (7 days before any holiday)
        df_copy['IsPreHoliday'] = (df_copy['DaysToNearestHoliday'] >= -7) & \
                                   (df_copy['DaysToNearestHoliday'] < 0)
        df_copy['IsPreHoliday'] = df_copy['IsPreHoliday'].astype(int)
        
        # Create post-holiday flag (7 days after any holiday)
        df_copy['IsPostHoliday'] = (df_copy['DaysToNearestHoliday'] > 0) & \
                                    (df_copy['DaysToNearestHoliday'] <= 7)
        df_copy['IsPostHoliday'] = df_copy['IsPostHoliday'].astype(int)
        
        print(f"Created holiday features for {len(self.holidays)} holidays")
        
        return df_copy
    
    def create_lag_features(self, df: pd.DataFrame, target_column: str, 
                          lags: List[int] = [1, 7, 14, 28]) -> pd.DataFrame:
        """
        Create lag features for time series data.
        
        Args:
            df: Input DataFrame
            target_column: Column to create lags for
            lags: List of lag periods
            
        Returns:
            DataFrame with lag features
        """
        df_copy = df.copy()
        
        for lag in lags:
            df_copy[f'{target_column}_Lag{lag}'] = df_copy[target_column].shift(lag)
        
        print(f"Created {len(lags)} lag features")
        
        return df_copy
    
    def create_rolling_features(self, df: pd.DataFrame, target_column: str,
                               windows: List[int] = [7, 14, 28]) -> pd.DataFrame:
        """
        Create rolling window statistics features.
        
        Args:
            df: Input DataFrame
            target_column: Column to calculate rolling statistics for
            windows: List of window sizes
            
        Returns:
            DataFrame with rolling features
        """
        df_copy = df.copy()
        
        for window in windows:
            df_copy[f'{target_column}_RollingMean{window}'] = \
                df_copy[target_column].rolling(window=window, min_periods=1).mean()
            df_copy[f'{target_column}_RollingStd{window}'] = \
                df_copy[target_column].rolling(window=window, min_periods=1).std()
            df_copy[f'{target_column}_RollingMin{window}'] = \
                df_copy[target_column].rolling(window=window, min_periods=1).min()
            df_copy[f'{target_column}_RollingMax{window}'] = \
                df_copy[target_column].rolling(window=window, min_periods=1).max()
        
        print(f"Created rolling features for {len(windows)} windows")
        
        return df_copy
    
    def encode_categorical_features(self, df: pd.DataFrame, 
                                   categorical_columns: List[str] = None) -> pd.DataFrame:
        """
        Encode categorical features using one-hot encoding.
        
        Args:
            df: Input DataFrame
            categorical_columns: List of categorical columns to encode
            
        Returns:
            DataFrame with encoded categorical features
        """
        df_copy = df.copy()
        
        if categorical_columns is None:
            categorical_columns = df_copy.select_dtypes(include=['object']).columns.tolist()
        
        # Remove date columns from categorical encoding
        categorical_columns = [col for col in categorical_columns 
                             if not pd.api.types.is_datetime64_any_dtype(df_copy[col])]
        
        if categorical_columns:
            df_copy = pd.get_dummies(df_copy, columns=categorical_columns, drop_first=True)
            print(f"Encoded {len(categorical_columns)} categorical features")
        
        return df_copy
    
    def preprocess_pipeline(self, df: pd.DataFrame, date_column: str, 
                          target_column: str = None,
                          handle_outliers_cols: List[str] = None) -> pd.DataFrame:
        """
        Complete preprocessing pipeline.
        
        Args:
            df: Input DataFrame
            date_column: Name of the date column
            target_column: Name of the target column (for lag/rolling features)
            handle_outliers_cols: Columns to handle outliers
            
        Returns:
            Fully preprocessed DataFrame
        """
        print("="*50)
        print("Starting Data Preprocessing Pipeline")
        print("="*50)
        
        # Step 1: Handle missing values
        print("\n1. Handling missing values...")
        df = self.handle_missing_values(df)
        
        # Step 2: Create date features
        print("\n2. Creating date features...")
        df = self.create_date_features(df, date_column)
        
        # Step 3: Create holiday features
        print("\n3. Creating holiday features...")
        df = self.create_holiday_features(df, date_column)
        
        # Step 4: Handle outliers if specified
        if handle_outliers_cols:
            print("\n4. Handling outliers...")
            for col in handle_outliers_cols:
                if col in df.columns:
                    df = self.handle_outliers(df, col, method='cap')
        
        # Step 5: Create lag features if target specified
        if target_column:
            print("\n5. Creating lag features...")
            df = self.create_lag_features(df, target_column)
            
            print("\n6. Creating rolling features...")
            df = self.create_rolling_features(df, target_column)
        
        # Step 7: Encode categorical features
        print(f"\n{7 if target_column else 5}. Encoding categorical features...")
        df = self.encode_categorical_features(df)
        
        print("\n" + "="*50)
        print("Preprocessing Pipeline Complete")
        print(f"Final shape: {df.shape}")
        print("="*50)
        
        return df


def create_sample_data(num_rows: int = 1000, start_date: str = '2010-01-01') -> pd.DataFrame:
    """
    Create sample sales data for demonstration purposes.
    
    Args:
        num_rows: Number of rows to generate
        start_date: Starting date for the data
        
    Returns:
        DataFrame with sample sales data
    """
    np.random.seed(42)
    
    # Generate date range
    dates = pd.date_range(start=start_date, periods=num_rows, freq='D')
    
    # Base sales with trend
    base_sales = 1000 + np.arange(num_rows) * 0.5
    
    # Add seasonality (annual cycle)
    seasonality = 200 * np.sin(2 * np.pi * np.arange(num_rows) / 365)
    
    # Add weekly pattern (weekend boost)
    weekly = np.array([50 if d.dayofweek >= 5 else 0 for d in dates])
    
    # Add random noise
    noise = np.random.normal(0, 50, num_rows)
    
    # Combine components
    sales = base_sales + seasonality + weekly + noise
    sales = np.maximum(sales, 0)  # Ensure non-negative
    
    # Create DataFrame
    df = pd.DataFrame({
        'Date': dates,
        'Weekly_Sales': sales,
        'Store': np.random.randint(1, 11, num_rows),
        'Dept': np.random.randint(1, 21, num_rows),
        'Temperature': np.random.uniform(30, 100, num_rows),
        'Fuel_Price': np.random.uniform(2.5, 4.5, num_rows),
        'CPI': np.random.uniform(150, 230, num_rows),
        'Unemployment': np.random.uniform(4, 10, num_rows)
    })
    
    return df


if __name__ == "__main__":
    # Example usage
    print("Sales Data Preprocessing Module")
    print("="*50)
    
    # Create sample data
    print("\nCreating sample data...")
    sample_df = create_sample_data(num_rows=1000)
    print(f"Sample data created with shape: {sample_df.shape}")
    print("\nFirst few rows:")
    print(sample_df.head())
    
    # Initialize preprocessor
    preprocessor = SalesDataPreprocessor()
    
    # Run preprocessing pipeline
    processed_df = preprocessor.preprocess_pipeline(
        df=sample_df,
        date_column='Date',
        target_column='Weekly_Sales',
        handle_outliers_cols=['Weekly_Sales', 'Temperature']
    )
    
    print("\n\nProcessed data preview:")
    print(processed_df.head())
    print(f"\nFeature columns: {list(processed_df.columns)}")
