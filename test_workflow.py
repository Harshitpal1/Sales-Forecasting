"""
End-to-end workflow test script for Sales Forecasting project.

This script demonstrates the complete pipeline:
1. Data generation/loading
2. Preprocessing and feature engineering
3. Model training
4. Evaluation and comparison
5. Results saving
"""

import numpy as np
import pandas as pd
import os

# Import from src package
from src.data_preprocessing import SalesDataPreprocessor, create_sample_data
from src.models import SalesForecaster
from src.evaluation import ModelEvaluator

def main():
    print("="*70)
    print("SALES FORECASTING - END-TO-END WORKFLOW TEST")
    print("="*70)
    
    # Set random seed
    np.random.seed(42)
    
    # Step 1: Create sample data
    print("\n[1/6] Creating sample data...")
    df = create_sample_data(num_rows=1000, start_date='2010-01-01')
    print(f"✓ Sample data created: {df.shape}")
    
    # Save sample data
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/sample_sales_data.csv', index=False)
    print(f"✓ Sample data saved to: data/sample_sales_data.csv")
    
    # Step 2: Preprocess data
    print("\n[2/6] Preprocessing data...")
    preprocessor = SalesDataPreprocessor()
    df_processed = preprocessor.preprocess_pipeline(
        df=df.copy(),
        date_column='Date',
        target_column='Weekly_Sales',
        handle_outliers_cols=['Weekly_Sales']
    )
    print(f"✓ Data preprocessed: {df_processed.shape}")
    
    # Remove NaN values from lag features
    df_model = df_processed.dropna().reset_index(drop=True)
    print(f"✓ Data cleaned (removed NaN): {df_model.shape}")
    
    # Step 3: Prepare data for modeling
    print("\n[3/6] Preparing train/test split...")
    forecaster = SalesForecaster(random_state=42)
    X_train, X_test, y_train, y_test = forecaster.prepare_data(
        df=df_model,
        target_column='Weekly_Sales',
        test_size=0.2
    )
    print(f"✓ Train set: {X_train.shape}, Test set: {X_test.shape}")
    
    # Step 4: Train models
    print("\n[4/6] Training models...")
    forecaster.initialize_models()
    trained_models = forecaster.train_all_models(X_train, y_train)
    print(f"✓ Trained {len(trained_models)} models")
    
    # Step 5: Evaluate models
    print("\n[5/6] Evaluating models...")
    evaluator = ModelEvaluator()
    
    results = {}
    for model_name in trained_models.keys():
        y_pred = forecaster.predict(model_name, X_test)
        metrics = evaluator.evaluate_model(y_test, y_pred, model_name)
        results[model_name] = {'predictions': y_pred, 'metrics': metrics}
    
    # Compare models
    print("\n" + "="*70)
    comparison_df = evaluator.compare_models()
    
    # Step 6: Save results
    print("\n[6/6] Saving results...")
    os.makedirs('results', exist_ok=True)
    os.makedirs('results/visualizations', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Save comparison
    comparison_df.to_csv('results/model_comparison.csv')
    print(f"✓ Model comparison saved to: results/model_comparison.csv")
    
    # Save best model
    best_model_name = comparison_df.index[0]
    model_path = f'models/{best_model_name.replace(" ", "_")}_model.pkl'
    forecaster.save_model(best_model_name, model_path)
    print(f"✓ Best model saved to: {model_path}")
    
    # Generate visualizations for best model
    print(f"\n[BONUS] Generating visualizations for best model: {best_model_name}")
    best_predictions = results[best_model_name]['predictions']
    
    # Create comparison plot
    evaluator.plot_model_comparison(
        comparison_df, 
        metric='RMSE',
        save_path='results/visualizations/model_comparison_rmse.png'
    )
    
    # Create prediction plot
    evaluator.plot_predictions(
        y_test,
        best_predictions,
        best_model_name,
        save_path=f'results/visualizations/{best_model_name.replace(" ", "_")}_predictions.png'
    )
    
    # Feature importance for tree-based models
    tree_models = ['Random Forest', 'Gradient Boosting', 'XGBoost']
    if best_model_name in tree_models:
        importance_df = forecaster.get_feature_importance(best_model_name, top_n=15)
        if importance_df is not None:
            evaluator.plot_feature_importance(
                importance_df,
                best_model_name,
                top_n=15,
                save_path=f'results/visualizations/{best_model_name.replace(" ", "_")}_feature_importance.png'
            )
            print(f"✓ Feature importance plot saved")
    
    # Summary
    print("\n" + "="*70)
    print("WORKFLOW TEST COMPLETED SUCCESSFULLY!")
    print("="*70)
    print("\nSummary:")
    print(f"  • Data samples processed: {len(df_model)}")
    print(f"  • Features engineered: {df_processed.shape[1] - df.shape[1]}")
    print(f"  • Models trained: {len(trained_models)}")
    print(f"  • Best model: {best_model_name}")
    print(f"  • Best RMSE: {comparison_df.loc[best_model_name, 'RMSE']:.2f}")
    print(f"  • Best R²: {comparison_df.loc[best_model_name, 'R2']:.4f}")
    print("\nOutput files:")
    print(f"  • data/sample_sales_data.csv")
    print(f"  • results/model_comparison.csv")
    print(f"  • {model_path}")
    print(f"  • results/visualizations/ (multiple PNG files)")
    print("\nNext steps:")
    print("  1. Run the Jupyter notebook for detailed analysis")
    print("  2. Use your own data by placing CSV in data/ directory")
    print("  3. Customize models and hyperparameters as needed")
    print("="*70)

if __name__ == "__main__":
    main()
