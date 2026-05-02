"""
Model Training Module
Trains multiple models for fraud detection
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, average_precision_score
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

class FraudModelTrainer:
    def __init__(self, models_dir='models/'):
        self.models_dir = models_dir
        os.makedirs(models_dir, exist_ok=True)
        self.models = {}
        self.results = {}
        
    def get_models(self):
        """Define models to train"""
        models = {
            'Logistic Regression': LogisticRegression(
                max_iter=1000, 
                random_state=42,
                class_weight='balanced'
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                class_weight='balanced',
                n_jobs=-1
            ),
            'XGBoost': XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                scale_pos_weight=100,  # Adjust for imbalance
                use_label_encoder=False,
                eval_metric='logloss'
            )
        }
        return models
    
    def train_model(self, model, X_train, y_train, X_test, y_test, model_name):
        """Train and evaluate a single model"""
        print(f"\n{'='*40}")
        print(f"Training {model_name}...")
        print(f"{'='*40}")
        
        # Train
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        results = {
            'model': model,
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'pr_auc': average_precision_score(y_test, y_pred_proba),
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
        
        # Print results
        tn, fp, fn, tp = results['confusion_matrix'].ravel()
        
        print(f"\nResults for {model_name}:")
        print(f"  ROC-AUC: {results['roc_auc']:.4f}")
        print(f"  PR-AUC: {results['pr_auc']:.4f}")
        print(f"  Confusion Matrix:")
        print(f"    True Negatives: {tn:,}")
        print(f"    False Positives: {fp:,}")
        print(f"    False Negatives: {fn:,}")
        print(f"    True Positives: {tp:,}")
        print(f"\n  Precision: {tp/(tp+fp):.4f}" if (tp+fp) > 0 else "  Precision: N/A")
        print(f"  Recall: {tp/(tp+fn):.4f}" if (tp+fn) > 0 else "  Recall: N/A")
        print(f"  F1-Score: {2*tp/(2*tp+fp+fn):.4f}" if (2*tp+fp+fn) > 0 else "  F1-Score: N/A")
        
        return results
    
    def train_all_models(self, X_train, y_train, X_test, y_test):
        """Train all models and compare"""
        models = self.get_models()
        best_model = None
        best_score = 0
        
        for model_name, model in models.items():
            results = self.train_model(model, X_train, y_train, X_test, y_test, model_name)
            self.results[model_name] = results
            
            # Save model
            model_path = os.path.join(self.models_dir, f"{model_name.lower().replace(' ', '_')}.joblib")
            joblib.dump(model, model_path)
            print(f"  Model saved to {model_path}")
            
            # Track best based on PR-AUC (important for imbalanced data)
            if results['pr_auc'] > best_score:
                best_score = results['pr_auc']
                best_model = model_name
        
        print(f"\n{'='*50}")
        print(f"BEST MODEL: {best_model} (PR-AUC: {best_score:.4f})")
        print(f"{'='*50}")
        
        return best_model
    
    def compare_models(self):
        """Compare all trained models"""
        print("\n" + "="*60)
        print("MODEL COMPARISON")
        print("="*60)
        
        comparison = pd.DataFrame({
            'Model': list(self.results.keys()),
            'ROC-AUC': [self.results[m]['roc_auc'] for m in self.results],
            'PR-AUC': [self.results[m]['pr_auc'] for m in self.results]
        })
        
        comparison = comparison.sort_values('PR-AUC', ascending=False)
        print(comparison.to_string(index=False))
        
        return comparison
    
    def get_feature_importance(self, model_name='Random Forest', feature_names=None):
        """Get feature importance from a model"""
        if model_name not in self.results:
            print(f"Model {model_name} not found")
            return None
        
        model = self.results[model_name]['model']
        
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif model_name == 'Logistic Regression' and hasattr(model, 'coef_'):
            importances = np.abs(model.coef_[0])
        else:
            print(f"Cannot extract feature importance from {model_name}")
            return None
        
        if feature_names is not None:
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            return importance_df
        
        return importances

if __name__ == "__main__":
    # Test the trainer with sample data
    from preprocess import DataPreprocessor
    
    preprocessor = DataPreprocessor()
    data = preprocessor.run_full_pipeline()
    
    trainer = FraudModelTrainer()
    best = trainer.train_all_models(
        data['X_train'], data['y_train'],
        data['X_test'], data['y_test']
    )
    
    # Get feature importance
    importance = trainer.get_feature_importance('Random Forest', data['feature_names'])
    if importance is not None:
        print("\nTop 10 Most Important Features:")
        print(importance.head(10))