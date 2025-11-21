"""
Quick Start Example for Sales Forecasting Project

This script demonstrates how to use the Sales Forecasting package
in just a few lines of code.
"""

from src import SalesDataPreprocessor, SalesForecaster, ModelEvaluator, create_sample_data
import pandas as pd

def quick_example():
    """
    Quick example showing the simplest way to use the package.
    """
    print("Sales Forecasting - Quick Start Example")
    print("=" * 60)
    
    # Step 1: Load or create data
    print("\n1. Creating sample data...")
    df = create_sample_data(num_rows=500)
    print(f"   ✓ Data created: {df.shape}")
    
    # Step 2: Preprocess and engineer features
    print("\n2. Preprocessing and feature engineering...")
    preprocessor = SalesDataPreprocessor()
    df_processed = preprocessor.preprocess_pipeline(
        df=df,
        date_column='Date',
        target_column='Weekly_Sales'
    )
    df_clean = df_processed.dropna()
    print(f"   ✓ Processed data: {df_clean.shape}")
    
    # Step 3: Train models
    print("\n3. Training models...")
    forecaster = SalesForecaster()
    forecaster.initialize_models()
    X_train, X_test, y_train, y_test = forecaster.prepare_data(
        df_clean, 'Weekly_Sales', test_size=0.2
    )
    forecaster.train_all_models(X_train, y_train)
    print(f"   ✓ Trained {len(forecaster.trained_models)} models")
    
    # Step 4: Evaluate and compare
    print("\n4. Evaluating models...")
    evaluator = ModelEvaluator()
    
    for model_name in forecaster.trained_models.keys():
        y_pred = forecaster.predict(model_name, X_test)
        evaluator.evaluate_model(y_test, y_pred, model_name)
    
    # Step 5: Get best model
    print("\n5. Model comparison:")
    comparison = evaluator.compare_models()
    best_model = comparison.index[0]
    
    print(f"\n✅ Best Model: {best_model}")
    print(f"   RMSE: {comparison.loc[best_model, 'RMSE']:.2f}")
    print(f"   R²:   {comparison.loc[best_model, 'R2']:.4f}")
    
    print("\n" + "=" * 60)
    print("Quick example completed!")
    print("For detailed analysis, run: jupyter notebook notebooks/sales_forecasting_analysis.ipynb")
    print("=" * 60)

if __name__ == "__main__":
    quick_example()
