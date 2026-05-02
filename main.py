"""
Credit Card Fraud Detection System - Main Pipeline
Run this file to execute the complete project
"""

import sys
import os
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_generator import FraudDataGenerator
from src.preprocess import DataPreprocessor
from src.train import FraudModelTrainer
from src.evaluate import FraudEvaluator
from src.visualize import FraudVisualizer

def main():
    print("="*80)
    print(" CREDIT CARD FRAUD DETECTION SYSTEM ".center(80, "="))
    print("="*80)
    
    # Step 1: Generate synthetic data
    print("\n📊 STEP 1: Generating Synthetic Transaction Data")
    print("-" * 50)
    generator = FraudDataGenerator(random_state=42)
    df = generator.generate_transactions(n_transactions=50000, fraud_ratio=0.005)
    generator.save_data(df, 'data/transactions.csv')
    
    # Step 2: Exploratory Data Analysis
    print("\n📈 STEP 2: Exploratory Data Analysis")
    print("-" * 50)
    viz = FraudVisualizer()
    viz.generate_complete_eda(df)
    
    # Step 3: Data Preprocessing
    print("\n🔄 STEP 3: Data Preprocessing")
    print("-" * 50)
    preprocessor = DataPreprocessor(random_state=42)
    processed_data = preprocessor.run_full_pipeline('data/transactions.csv', apply_smote=True)
    
    # Step 4: Model Training
    print("\n🤖 STEP 4: Model Training")
    print("-" * 50)
    trainer = FraudModelTrainer()
    best_model = trainer.train_all_models(
        processed_data['X_train'], 
        processed_data['y_train'],
        processed_data['X_test'], 
        processed_data['y_test']
    )
    
    # Step 5: Model Comparison
    print("\n📊 STEP 5: Model Comparison")
    print("-" * 50)
    comparison = trainer.compare_models()
    
    # Plot model comparison
    viz.plot_model_comparison(comparison)
    
    # Step 6: Evaluation on best model
    print("\n🎯 STEP 6: Model Evaluation")
    print("-" * 50)
    evaluator = FraudEvaluator()
    
    # Load best model and evaluate
    best_model_obj = trainer.results[best_model]['model']
    y_proba = trainer.results[best_model]['probabilities']
    
    # Get feature importance
    importance = trainer.get_feature_importance(best_model, processed_data['feature_names'])
    if importance is not None:
        print("\nTop 10 Most Important Features for Fraud Detection:")
        print(importance.head(10).to_string(index=False))
    
    # Generate evaluation plots
    evaluator.generate_all_plots(
        processed_data['y_test'], 
        y_proba, 
        best_model,
        processed_data['feature_names'],
        importance
    )
    
    # Generate comprehensive report
    optimal_metrics, optimal_threshold = evaluator.generate_report(
        processed_data['y_test'], 
        y_proba, 
        best_model
    )
    
    # Step 7: Final Summary
    print("\n" + "="*80)
    print(" PROJECT COMPLETED SUCCESSFULLY ".center(80, "="))
    print("="*80)
    
    print("\n📁 OUTPUT FILES GENERATED:")
    print("  - data/transactions.csv (Synthetic transaction data)")
    print("  - models/*.joblib (Trained models)")
    print("  - outputs/*.png (Visualizations)")
    print("  - outputs/*.txt (Evaluation reports)")
    
    print("\n🎯 KEY RESULTS:")
    print(f"  - Best Model: {best_model}")
    print(f"  - ROC-AUC: {trainer.results[best_model]['roc_auc']:.4f}")
    print(f"  - PR-AUC: {trainer.results[best_model]['pr_auc']:.4f}")
    print(f"  - Optimal Threshold: {optimal_threshold:.4f}")
    print(f"  - Precision at optimal threshold: {optimal_metrics['precision']:.4f}")
    print(f"  - Recall at optimal threshold: {optimal_metrics['recall']:.4f}")
    print(f"  - F1-Score at optimal threshold: {optimal_metrics['f1_score']:.4f}")
    
    print("\n💡 INTERPRETATION:")
    if optimal_metrics['recall'] > 0.7:
        print("  ✓ High recall - Model catches most fraud transactions")
    if optimal_metrics['precision'] > 0.5:
        print("  ✓ Good precision - Low false alarm rate")
    
    print("\n📚 NEXT STEPS FOR PRODUCTION:")
    print("  1. Set optimal threshold in API: threshold = {:.4f}".format(optimal_threshold))
    print("  2. Monitor precision/recall weekly for drift")
    print("  3. Retrain model monthly with new data")
    print("  4. Add real-time streaming with Kafka")
    
    print("\n✅ Project ready for GitHub upload!")

if __name__ == "__main__":
    main()