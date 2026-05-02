"""
Model Evaluation Module
Comprehensive evaluation with threshold optimization
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    precision_recall_curve,
    roc_curve,
    confusion_matrix,
    classification_report,
    average_precision_score,
    roc_auc_score
)
import joblib

class FraudEvaluator:
    def __init__(self, models_dir='models/', output_dir='outputs/'):
        self.models_dir = models_dir
        self.output_dir = output_dir
        import os
        os.makedirs(output_dir, exist_ok=True)
        
    def load_model(self, model_name):
        """Load a trained model"""
        model_path = f"{self.models_dir}/{model_name.lower().replace(' ', '_')}.joblib"
        try:
            model = joblib.load(model_path)
            print(f"Loaded model: {model_name}")
            return model
        except FileNotFoundError:
            print(f"Model not found: {model_path}")
            return None
    
    def find_optimal_threshold(self, y_true, y_proba, cost_fn=100, cost_fp=10):
        """
        Find optimal threshold to minimize cost
        
        Parameters:
        - y_true: True labels
        - y_proba: Predicted probabilities
        - cost_fn: Cost of false negative (missed fraud)
        - cost_fp: Cost of false positive (wrong alert)
        """
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
        
        costs = []
        optimal_idx = 0
        min_cost = float('inf')
        
        for i, threshold in enumerate(thresholds):
            y_pred = (y_proba >= threshold).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            
            cost = (fn * cost_fn) + (fp * cost_fp)
            costs.append(cost)
            
            if cost < min_cost:
                min_cost = cost
                optimal_idx = i
        
        if len(thresholds) > 0:
            optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
        else:
            optimal_threshold = 0.5
        
        print(f"\nOptimal threshold: {optimal_threshold:.4f}")
        print(f"Minimum cost: {min_cost:.2f}")
        
        return optimal_threshold, thresholds, costs
    
    def evaluate_at_threshold(self, y_true, y_proba, threshold=0.5):
        """Evaluate model performance at specific threshold"""
        y_pred = (y_proba >= threshold).astype(int)
        
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        metrics = {
            'threshold': threshold,
            'true_negatives': tn,
            'false_positives': fp,
            'false_negatives': fn,
            'true_positives': tp,
            'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
            'recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
            'accuracy': (tp + tn) / (tp + tn + fp + fn),
            'f1_score': 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
        }
        
        return metrics
    
    def generate_all_plots(self, y_true, y_proba, model_name, feature_names=None, importance=None):
        """Generate all evaluation plots"""
        
        # 1. ROC Curve
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 2, 1)
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = roc_auc_score(y_true, y_proba)
        plt.plot(fpr, tpr, 'b-', label=f'ROC Curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], 'r--', label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 2. Precision-Recall Curve
        plt.subplot(2, 2, 2)
        precision, recall, thresholds_pr = precision_recall_curve(y_true, y_proba)
        pr_auc = average_precision_score(y_true, y_proba)
        plt.plot(recall, precision, 'g-', label=f'PR Curve (AUC = {pr_auc:.4f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve - {model_name}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 3. Confusion Matrix
        plt.subplot(2, 2, 3)
        optimal_thresh, _, _ = self.find_optimal_threshold(y_true, y_proba)
        y_pred_opt = (y_proba >= optimal_thresh).astype(int)
        cm = confusion_matrix(y_true, y_pred_opt)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Normal', 'Fraud'],
                    yticklabels=['Normal', 'Fraud'])
        plt.title(f'Confusion Matrix (Threshold={optimal_thresh:.3f})')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        
        # 4. Threshold Analysis
        plt.subplot(2, 2, 4)
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
        if len(thresholds) > 0:
            plt.plot(thresholds, precisions[:-1], 'b-', label='Precision')
            plt.plot(thresholds, recalls[:-1], 'r-', label='Recall')
            plt.axvline(x=optimal_thresh, color='g', linestyle='--', label=f'Optimal ({optimal_thresh:.3f})')
            plt.xlabel('Threshold')
            plt.ylabel('Score')
            plt.title('Precision/Recall vs Threshold')
            plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = f"{self.output_dir}/{model_name.lower().replace(' ', '_')}_evaluation.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"Plots saved to {plot_path}")
        
        # 5. Feature Importance (if available)
        if importance is not None and feature_names is not None:
            plt.figure(figsize=(10, 8))
            top_features = importance.head(15)
            plt.barh(top_features['feature'], top_features['importance'])
            plt.xlabel('Importance')
            plt.title(f'Top 15 Features - {model_name}')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            importance_path = f"{self.output_dir}/{model_name.lower().replace(' ', '_')}_feature_importance.png"
            plt.savefig(importance_path, dpi=150, bbox_inches='tight')
            plt.show()
            print(f"Feature importance plot saved to {importance_path}")
    
    def generate_report(self, y_true, y_proba, model_name, threshold=0.5):
        """Generate comprehensive evaluation report"""
        
        print("\n" + "="*70)
        print(f"FRAUD DETECTION MODEL EVALUATION REPORT - {model_name}")
        print("="*70)
        
        # Find optimal threshold
        optimal_threshold, thresholds, costs = self.find_optimal_threshold(y_true, y_proba)
        
        # Evaluate at default and optimal thresholds
        default_metrics = self.evaluate_at_threshold(y_true, y_proba, threshold)
        optimal_metrics = self.evaluate_at_threshold(y_true, y_proba, optimal_threshold)
        
        print(f"\n{'Metric':<20} {'Default (0.5)':<20} {'Optimal':<20}")
        print("-" * 60)
        print(f"{'Precision':<20} {default_metrics['precision']:<20.4f} {optimal_metrics['precision']:<20.4f}")
        print(f"{'Recall':<20} {default_metrics['recall']:<20.4f} {optimal_metrics['recall']:<20.4f}")
        print(f"{'F1-Score':<20} {default_metrics['f1_score']:<20.4f} {optimal_metrics['f1_score']:<20.4f}")
        print(f"{'Accuracy':<20} {default_metrics['accuracy']:<20.4f} {optimal_metrics['accuracy']:<20.4f}")
        print(f"{'Specificity':<20} {default_metrics['specificity']:<20.4f} {optimal_metrics['specificity']:<20.4f}")
        
        print(f"\n{'Confusion Matrix (Optimal Threshold)':^50}")
        print("-" * 50)
        print(f"True Negatives:  {optimal_metrics['true_negatives']:,}")
        print(f"False Positives: {optimal_metrics['false_positives']:,}")
        print(f"False Negatives: {optimal_metrics['false_negatives']:,}")
        print(f"True Positives:  {optimal_metrics['true_positives']:,}")
        
        print(f"\n{'Classification Report (Optimal Threshold)':^50}")
        print("-" * 50)
        y_pred_opt = (y_proba >= optimal_threshold).astype(int)
        print(classification_report(y_true, y_pred_opt, target_names=['Normal', 'Fraud']))
        
        # Save report
        report_path = f"{self.output_dir}/{model_name.lower().replace(' ', '_')}_report.txt"
        with open(report_path, 'w') as f:
            f.write(f"Fraud Detection Model Evaluation Report - {model_name}\n")
            f.write("="*60 + "\n\n")
            f.write(f"Optimal Threshold: {optimal_threshold:.4f}\n\n")
            f.write("Metrics at Optimal Threshold:\n")
            for key, value in optimal_metrics.items():
                f.write(f"  {key}: {value}\n")
        
        print(f"\nReport saved to {report_path}")
        
        return optimal_metrics, optimal_threshold

if __name__ == "__main__":
    # Test evaluation
    from preprocess import DataPreprocessor
    from train import FraudModelTrainer
    
    # Prepare data
    preprocessor = DataPreprocessor()
    data = preprocessor.run_full_pipeline()
    
    # Train model
    trainer = FraudModelTrainer()
    trainer.train_all_models(data['X_train'], data['y_train'], data['X_test'], data['y_test'])
    
    # Evaluate best model
    evaluator = FraudEvaluator()
    model = evaluator.load_model('Random Forest')
    
    if model:
        y_proba = model.predict_proba(data['X_test'])[:, 1]
        evaluator.generate_all_plots(data['y_test'], y_proba, 'Random Forest')
        evaluator.generate_report(data['y_test'], y_proba, 'Random Forest')