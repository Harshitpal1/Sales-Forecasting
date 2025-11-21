"""
Model Evaluation Module for Sales Forecasting

This module provides comprehensive evaluation metrics and visualization functions:
- Performance metrics: RMSE, MAE, MAPE, R²
- Prediction vs actual plots
- Residual analysis
- Model comparison visualizations
- Feature importance plots
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)
import warnings

warnings.filterwarnings('ignore')

# Set style for better-looking plots
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except OSError:
    plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class ModelEvaluator:
    """
    Comprehensive model evaluation and visualization.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (12, 6)):
        """
        Initialize the evaluator.
        
        Args:
            figsize: Default figure size for plots
        """
        self.figsize = figsize
        self.results = {}
        
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                         model_name: str = None) -> Dict[str, float]:
        """
        Calculate comprehensive regression metrics.
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            model_name: Name of the model (for storing results)
            
        Returns:
            Dictionary with all metrics
        """
        # Basic metrics
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # MAPE (Mean Absolute Percentage Error)
        # Avoid division by zero
        mask = y_true != 0
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        
        # Additional metrics
        mse = mean_squared_error(y_true, y_pred)
        
        # Mean Error (bias)
        me = np.mean(y_pred - y_true)
        
        metrics = {
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape,
            'R2': r2,
            'MSE': mse,
            'ME': me
        }
        
        # Store results if model name provided
        if model_name:
            self.results[model_name] = metrics
        
        return metrics
    
    def print_metrics(self, metrics: Dict[str, float], model_name: str = None):
        """
        Print metrics in a formatted way.
        
        Args:
            metrics: Dictionary of metrics
            model_name: Name of the model
        """
        if model_name:
            print(f"\n{'='*50}")
            print(f"Metrics for {model_name}")
            print(f"{'='*50}")
        
        print(f"RMSE:  {metrics['RMSE']:>10.2f}")
        print(f"MAE:   {metrics['MAE']:>10.2f}")
        print(f"MAPE:  {metrics['MAPE']:>10.2f}%")
        print(f"R²:    {metrics['R2']:>10.4f}")
        print(f"MSE:   {metrics['MSE']:>10.2f}")
        print(f"ME:    {metrics['ME']:>10.2f}")
        
    def evaluate_model(self, y_true: np.ndarray, y_pred: np.ndarray, 
                      model_name: str) -> Dict[str, float]:
        """
        Evaluate a model and print metrics.
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            model_name: Name of the model
            
        Returns:
            Dictionary with all metrics
        """
        metrics = self.calculate_metrics(y_true, y_pred, model_name)
        self.print_metrics(metrics, model_name)
        return metrics
    
    def compare_models(self, results: Dict[str, Dict[str, float]] = None) -> pd.DataFrame:
        """
        Compare multiple models side by side.
        
        Args:
            results: Dictionary of model results (if None, use stored results)
            
        Returns:
            DataFrame comparing all models
        """
        if results is None:
            results = self.results
        
        if not results:
            print("No results to compare")
            return None
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame(results).T
        
        # Sort by RMSE (lower is better)
        comparison_df = comparison_df.sort_values('RMSE')
        
        # Highlight best model
        print("\n" + "="*70)
        print("MODEL COMPARISON")
        print("="*70)
        print(comparison_df.to_string())
        print("\n" + "="*70)
        print(f"Best Model (by RMSE): {comparison_df.index[0]}")
        print("="*70)
        
        return comparison_df
    
    def plot_predictions(self, y_true: np.ndarray, y_pred: np.ndarray,
                        model_name: str, save_path: str = None):
        """
        Plot actual vs predicted values.
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            model_name: Name of the model
            save_path: Path to save the plot (optional)
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot 1: Time series comparison
        indices = np.arange(len(y_true))
        ax1.plot(indices, y_true, label='Actual', alpha=0.7, linewidth=2)
        ax1.plot(indices, y_pred, label='Predicted', alpha=0.7, linewidth=2)
        ax1.set_xlabel('Sample Index', fontsize=12)
        ax1.set_ylabel('Sales', fontsize=12)
        ax1.set_title(f'{model_name}: Actual vs Predicted', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Scatter plot
        ax2.scatter(y_true, y_pred, alpha=0.5, s=20)
        
        # Perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        
        ax2.set_xlabel('Actual Sales', fontsize=12)
        ax2.set_ylabel('Predicted Sales', fontsize=12)
        ax2.set_title(f'{model_name}: Prediction Scatter', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()
        plt.close()
    
    def plot_residuals(self, y_true: np.ndarray, y_pred: np.ndarray,
                      model_name: str, save_path: str = None):
        """
        Plot residual analysis.
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            model_name: Name of the model
            save_path: Path to save the plot (optional)
        """
        residuals = y_true - y_pred
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Residuals over time
        axes[0, 0].plot(residuals, alpha=0.7)
        axes[0, 0].axhline(y=0, color='r', linestyle='--', linewidth=2)
        axes[0, 0].set_xlabel('Sample Index', fontsize=11)
        axes[0, 0].set_ylabel('Residuals', fontsize=11)
        axes[0, 0].set_title('Residuals Over Time', fontsize=12, fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Residuals vs Predicted
        axes[0, 1].scatter(y_pred, residuals, alpha=0.5, s=20)
        axes[0, 1].axhline(y=0, color='r', linestyle='--', linewidth=2)
        axes[0, 1].set_xlabel('Predicted Sales', fontsize=11)
        axes[0, 1].set_ylabel('Residuals', fontsize=11)
        axes[0, 1].set_title('Residuals vs Predicted', fontsize=12, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Histogram of residuals
        axes[1, 0].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
        axes[1, 0].axvline(x=0, color='r', linestyle='--', linewidth=2)
        axes[1, 0].set_xlabel('Residuals', fontsize=11)
        axes[1, 0].set_ylabel('Frequency', fontsize=11)
        axes[1, 0].set_title('Distribution of Residuals', fontsize=12, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Q-Q plot
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[1, 1])
        axes[1, 1].set_title('Q-Q Plot', fontsize=12, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
        
        fig.suptitle(f'{model_name}: Residual Analysis', fontsize=16, fontweight='bold', y=1.00)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()
        plt.close()
    
    def plot_model_comparison(self, comparison_df: pd.DataFrame = None,
                            metric: str = 'RMSE', save_path: str = None):
        """
        Plot comparison of multiple models.
        
        Args:
            comparison_df: DataFrame with model comparison (if None, use stored results)
            metric: Metric to compare
            save_path: Path to save the plot (optional)
        """
        if comparison_df is None:
            if not self.results:
                print("No results to plot")
                return
            comparison_df = pd.DataFrame(self.results).T
        
        # Create bar plot
        fig, ax = plt.subplots(figsize=self.figsize)
        
        models = comparison_df.index
        values = comparison_df[metric]
        
        bars = ax.bar(models, values, alpha=0.8, edgecolor='black')
        
        # Color bars by performance (lower is better for RMSE, MAE, MAPE)
        if metric in ['RMSE', 'MAE', 'MAPE', 'MSE']:
            colors = plt.cm.RdYlGn_r(np.linspace(0.3, 0.9, len(models)))
        else:  # R2 - higher is better
            colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(models)))
        
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        # Add value labels on bars
        for i, (model, value) in enumerate(zip(models, values)):
            ax.text(i, value, f'{value:.2f}', 
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax.set_xlabel('Models', fontsize=12, fontweight='bold')
        ax.set_ylabel(metric, fontsize=12, fontweight='bold')
        ax.set_title(f'Model Comparison: {metric}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()
        plt.close()
    
    def plot_multiple_metrics_comparison(self, comparison_df: pd.DataFrame = None,
                                        metrics: List[str] = None, save_path: str = None):
        """
        Plot comparison of multiple metrics for all models.
        
        Args:
            comparison_df: DataFrame with model comparison
            metrics: List of metrics to compare
            save_path: Path to save the plot (optional)
        """
        if comparison_df is None:
            if not self.results:
                print("No results to plot")
                return
            comparison_df = pd.DataFrame(self.results).T
        
        if metrics is None:
            metrics = ['RMSE', 'MAE', 'MAPE', 'R2']
        
        # Normalize metrics for comparison (0-1 scale)
        comparison_normalized = comparison_df[metrics].copy()
        for metric in metrics:
            if metric in ['RMSE', 'MAE', 'MAPE', 'MSE']:
                # Lower is better - invert normalization
                comparison_normalized[metric] = 1 - (
                    (comparison_normalized[metric] - comparison_normalized[metric].min()) /
                    (comparison_normalized[metric].max() - comparison_normalized[metric].min())
                )
            else:
                # Higher is better (R2)
                comparison_normalized[metric] = (
                    (comparison_normalized[metric] - comparison_normalized[metric].min()) /
                    (comparison_normalized[metric].max() - comparison_normalized[metric].min())
                )
        
        # Create grouped bar plot
        fig, ax = plt.subplots(figsize=(14, 6))
        
        x = np.arange(len(comparison_df.index))
        width = 0.8 / len(metrics)
        
        for i, metric in enumerate(metrics):
            offset = (i - len(metrics)/2) * width + width/2
            ax.bar(x + offset, comparison_normalized[metric], width, 
                  label=metric, alpha=0.8, edgecolor='black')
        
        ax.set_xlabel('Models', fontsize=12, fontweight='bold')
        ax.set_ylabel('Normalized Score (higher is better)', fontsize=12, fontweight='bold')
        ax.set_title('Multi-Metric Model Comparison (Normalized)', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(comparison_df.index, rotation=45, ha='right')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([0, 1.1])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()
        plt.close()
    
    def plot_feature_importance(self, importance_df: pd.DataFrame, 
                               model_name: str, top_n: int = 20,
                               save_path: str = None):
        """
        Plot feature importance.
        
        Args:
            importance_df: DataFrame with feature importances
            model_name: Name of the model
            top_n: Number of top features to display
            save_path: Path to save the plot (optional)
        """
        if importance_df is None or len(importance_df) == 0:
            print("No feature importance data available")
            return
        
        # Get top N features
        plot_df = importance_df.head(top_n).sort_values('Importance')
        
        # Create horizontal bar plot
        fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.3)))
        
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(plot_df)))
        bars = ax.barh(plot_df['Feature'], plot_df['Importance'], 
                       alpha=0.8, edgecolor='black', color=colors)
        
        ax.set_xlabel('Importance', fontsize=12, fontweight='bold')
        ax.set_ylabel('Features', fontsize=12, fontweight='bold')
        ax.set_title(f'{model_name}: Top {top_n} Important Features', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()
        plt.close()
    
    def generate_evaluation_report(self, y_true: np.ndarray, y_pred: np.ndarray,
                                  model_name: str, save_dir: str = None) -> Dict[str, float]:
        """
        Generate complete evaluation report with all plots and metrics.
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            model_name: Name of the model
            save_dir: Directory to save plots (optional)
            
        Returns:
            Dictionary with all metrics
        """
        print(f"\n{'='*70}")
        print(f"EVALUATION REPORT: {model_name}")
        print(f"{'='*70}")
        
        # Calculate and print metrics
        metrics = self.evaluate_model(y_true, y_pred, model_name)
        
        # Generate plots
        if save_dir:
            pred_path = f"{save_dir}/{model_name.replace(' ', '_')}_predictions.png"
            resid_path = f"{save_dir}/{model_name.replace(' ', '_')}_residuals.png"
        else:
            pred_path = None
            resid_path = None
        
        print("\nGenerating prediction plots...")
        self.plot_predictions(y_true, y_pred, model_name, save_path=pred_path)
        
        print("\nGenerating residual analysis...")
        self.plot_residuals(y_true, y_pred, model_name, save_path=resid_path)
        
        print(f"\n{'='*70}")
        print("EVALUATION REPORT COMPLETE")
        print(f"{'='*70}")
        
        return metrics


def demonstrate_evaluation():
    """
    Demonstrate evaluation functionality with sample data.
    """
    print("Model Evaluation Demo")
    print("="*50)
    
    # Create sample predictions
    np.random.seed(42)
    n_samples = 200
    
    y_true = np.random.randn(n_samples) * 100 + 1000
    y_pred1 = y_true + np.random.randn(n_samples) * 50  # Model 1
    y_pred2 = y_true + np.random.randn(n_samples) * 30  # Model 2 (better)
    
    # Initialize evaluator
    evaluator = ModelEvaluator()
    
    # Evaluate models
    metrics1 = evaluator.evaluate_model(y_true, y_pred1, "Model 1")
    metrics2 = evaluator.evaluate_model(y_true, y_pred2, "Model 2")
    
    # Compare models
    comparison_df = evaluator.compare_models()
    
    # Plot comparison
    print("\nGenerating comparison plots...")
    evaluator.plot_model_comparison(comparison_df, metric='RMSE')


if __name__ == "__main__":
    demonstrate_evaluation()
