"""
Visualization Module
EDA and result visualization
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

class FraudVisualizer:
    def __init__(self, output_dir='outputs/'):
        self.output_dir = output_dir
        import os
        os.makedirs(output_dir, exist_ok=True)
        
    def plot_class_distribution(self, df):
        """Plot fraud vs non-fraud distribution"""
        plt.figure(figsize=(10, 6))
        
        colors = ['#2ecc71', '#e74c3c']
        labels = ['Normal', 'Fraud']
        
        counts = df['is_fraud'].value_counts()
        plt.bar(labels, counts.values, color=colors, alpha=0.7)
        plt.title('Transaction Class Distribution', fontsize=14, fontweight='bold')
        plt.ylabel('Number of Transactions')
        plt.xlabel('Transaction Type')
        
        # Add value labels on bars
        for i, v in enumerate(counts.values):
            plt.text(i, v + (v*0.01), str(v), ha='center', va='bottom', fontweight='bold')
        
        # Add fraud percentage
        fraud_pct = df['is_fraud'].mean() * 100
        plt.text(0.5, 0.95, f'Fraud Rate: {fraud_pct:.3f}%', 
                 transform=plt.gca().transAxes, ha='center', 
                 fontsize=12, bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/class_distribution.png", dpi=150, bbox_inches='tight')
        plt.show()
        
    def plot_amount_distribution(self, df):
        """Plot transaction amount distributions"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Normal transactions
        normal_amounts = df[df['is_fraud'] == 0]['amount']
        axes[0].hist(normal_amounts, bins=50, color='#2ecc71', alpha=0.7, edgecolor='black')
        axes[0].set_title('Normal Transactions - Amount Distribution', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Transaction Amount ($)')
        axes[0].set_ylabel('Frequency')
        axes[0].axvline(normal_amounts.median(), color='red', linestyle='--', 
                        label=f'Median: ${normal_amounts.median():.2f}')
        axes[0].legend()
        
        # Fraud transactions
        fraud_amounts = df[df['is_fraud'] == 1]['amount']
        axes[1].hist(fraud_amounts, bins=30, color='#e74c3c', alpha=0.7, edgecolor='black')
        axes[1].set_title('Fraud Transactions - Amount Distribution', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Transaction Amount ($)')
        axes[1].set_ylabel('Frequency')
        axes[1].axvline(fraud_amounts.median(), color='blue', linestyle='--', 
                        label=f'Median: ${fraud_amounts.median():.2f}')
        axes[1].legend()
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/amount_distribution.png", dpi=150, bbox_inches='tight')
        plt.show()
        
    def plot_hourly_patterns(self, df):
        """Plot hourly transaction patterns"""
        plt.figure(figsize=(12, 6))
        
        # Normal transactions by hour
        normal_hours = df[df['is_fraud'] == 0]['transaction_hour'].value_counts().sort_index()
        fraud_hours = df[df['is_fraud'] == 1]['transaction_hour'].value_counts().sort_index()
        
        # Normalize by total to compare patterns
        normal_hours_norm = normal_hours / normal_hours.sum() * 100
        fraud_hours_norm = fraud_hours / fraud_hours.sum() * 100
        
        plt.plot(normal_hours_norm.index, normal_hours_norm.values, 
                 'g-o', label='Normal Transactions', linewidth=2, markersize=8)
        plt.plot(fraud_hours_norm.index, fraud_hours_norm.values, 
                 'r-s', label='Fraud Transactions', linewidth=2, markersize=8)
        
        plt.xlabel('Hour of Day (0-23)', fontsize=12)
        plt.ylabel('Percentage of Transactions (%)', fontsize=12)
        plt.title('Transaction Patterns by Hour of Day', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(range(0, 24))
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/hourly_patterns.png", dpi=150, bbox_inches='tight')
        plt.show()
        
    def plot_correlation_matrix(self, df, features):
        """Plot correlation matrix of features"""
        # Select numerical features
        num_features = [col for col in features if col in df.columns and df[col].dtype in ['int64', 'float64']]
        corr_matrix = df[num_features].corr()
        
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', 
                    center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
        plt.title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/correlation_matrix.png", dpi=150, bbox_inches='tight')
        plt.show()
        
    def plot_merchant_analysis(self, df):
        """Analyze fraud by merchant category"""
        plt.figure(figsize=(12, 6))
        
        # Fraud rate by merchant
        fraud_by_merchant = df.groupby('merchant_category')['is_fraud'].agg(['mean', 'count'])
        fraud_by_merchant = fraud_by_merchant.sort_values('mean', ascending=False)
        
        ax = fraud_by_merchant['mean'].plot(kind='bar', color='coral', alpha=0.7)
        plt.xlabel('Merchant Category')
        plt.ylabel('Fraud Rate')
        plt.title('Fraud Rate by Merchant Category', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        
        # Add value labels
        for i, v in enumerate(fraud_by_merchant['mean']):
            ax.text(i, v + 0.001, f'{v:.3%}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/merchant_analysis.png", dpi=150, bbox_inches='tight')
        plt.show()
        
    def generate_complete_eda(self, df):
        """Generate complete EDA report"""
        print("\n" + "="*60)
        print("EXPLORATORY DATA ANALYSIS REPORT")
        print("="*60)
        
        # Basic statistics
        print(f"\nDataset Shape: {df.shape}")
        print(f"\nFeature Types:")
        print(df.dtypes.value_counts().to_string())
        
        print(f"\nMissing Values:")
        print(df.isnull().sum().to_string())
        
        print(f"\nNumerical Features Statistics:")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        print(df[numeric_cols].describe().to_string())
        
        # Generate plots
        print("\nGenerating EDA Plots...")
        self.plot_class_distribution(df)
        self.plot_amount_distribution(df)
        self.plot_hourly_patterns(df)
        
        # Select key features for correlation
        key_features = ['amount', 'log_amount', 'transaction_hour', 'day_of_week', 
                        'days_since_last_tx', 'avg_transaction_amount', 
                        'transaction_count_24h', 'is_night', 'is_weekend', 'is_fraud']
        available_features = [f for f in key_features if f in df.columns]
        self.plot_correlation_matrix(df, available_features)
        
        try:
            self.plot_merchant_analysis(df)
        except:
            print("Merchant analysis skipped (merchant_category not available)")
        
        print("\nEDA Complete! All plots saved to outputs/ directory")
        
    def plot_model_comparison(self, results_df):
        """Plot model comparison bar chart"""
        plt.figure(figsize=(10, 6))
        
        x = np.arange(len(results_df['Model']))
        width = 0.35
        
        plt.bar(x - width/2, results_df['ROC-AUC'], width, label='ROC-AUC', color='steelblue')
        plt.bar(x + width/2, results_df['PR-AUC'], width, label='PR-AUC', color='coral')
        
        plt.xlabel('Model')
        plt.ylabel('Score')
        plt.title('Model Performance Comparison', fontsize=14, fontweight='bold')
        plt.xticks(x, results_df['Model'], rotation=45, ha='right')
        plt.legend()
        plt.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for i, row in results_df.iterrows():
            plt.text(i - width/2, row['ROC-AUC'] + 0.01, f'{row["ROC-AUC"]:.3f}', 
                     ha='center', va='bottom', fontsize=9)
            plt.text(i + width/2, row['PR-AUC'] + 0.01, f'{row["PR-AUC"]:.3f}', 
                     ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/model_comparison.png", dpi=150, bbox_inches='tight')
        plt.show()

if __name__ == "__main__":
    # Test visualization
    from data_generator import FraudDataGenerator
    
    generator = FraudDataGenerator()
    df = generator.generate_transactions(n_transactions=10000)
    
    viz = FraudVisualizer()
    viz.generate_complete_eda(df)