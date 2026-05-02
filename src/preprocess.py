"""
Data Preprocessing Module
Handles cleaning, encoding, scaling, and imbalance correction
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    def __init__(self, test_size=0.2, random_state=42):
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = None
        
    def load_data(self, filepath='data/transactions.csv'):
        """Load transaction data"""
        df = pd.read_csv(filepath)
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        print(f"Loaded {len(df)} transactions")
        print(f"Fraud rate: {df['is_fraud'].mean():.4f}")
        return df
    
    def clean_data(self, df):
        """Clean and prepare data for modeling"""
        df_clean = df.copy()
        
        # Remove unnecessary columns
        cols_to_drop = ['transaction_id', 'timestamp', 'customer_id']
        existing_drop = [c for c in cols_to_drop if c in df_clean.columns]
        if existing_drop:
            df_clean = df_clean.drop(columns=existing_drop)
        
        # Separate numeric and non-numeric columns for filling NaN
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        non_numeric_cols = df_clean.select_dtypes(exclude=[np.number]).columns
        
        # Fill NaN only in numeric columns with median
        if len(numeric_cols) > 0:
            for col in numeric_cols:
                if df_clean[col].isnull().any():
                    df_clean[col] = df_clean[col].fillna(df_clean[col].median())
        
        # Fill NaN in non-numeric columns with mode (most frequent value)
        for col in non_numeric_cols:
            if df_clean[col].isnull().any() and col != 'is_fraud':
                mode_val = df_clean[col].mode()
                if len(mode_val) > 0:
                    df_clean[col] = df_clean[col].fillna(mode_val[0])
        
        # Encode categorical variables
        categorical_cols = ['merchant_category', 'location', 'device_type']
        for col in categorical_cols:
            if col in df_clean.columns:
                le = LabelEncoder()
                df_clean[col] = le.fit_transform(df_clean[col].astype(str))
                self.label_encoders[col] = le
        
        print(f"Data cleaned. Shape: {df_clean.shape}")
        return df_clean
    
    def prepare_features(self, df):
        """Prepare features for modeling"""
        # Define feature columns that should exist
        possible_feature_cols = [
            'amount', 'log_amount', 'amount_normalized',
            'transaction_hour', 'day_of_week', 'days_since_last_tx',
            'avg_transaction_amount', 'transaction_count_24h',
            'merchant_category', 'location', 'device_type',
            'is_international', 'is_night', 'is_weekend'
        ]
        
        # Select only columns that actually exist in the dataframe
        available_cols = [col for col in possible_feature_cols if col in df.columns]
        
        # Also include any other numeric columns that might be useful
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col not in available_cols and col != 'is_fraud':
                available_cols.append(col)
        
        X = df[available_cols].copy()
        y = df['is_fraud'].copy()
        
        self.feature_names = X.columns.tolist()
        print(f"Prepared {len(available_cols)} features: {self.feature_names}")
        
        return X, y
    
    def split_data(self, X, y):
        """Split data into train and test sets"""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )
        
        print(f"Train set: {len(X_train)} samples (fraud rate: {y_train.mean():.6f})")
        print(f"Test set: {len(X_test)} samples (fraud rate: {y_test.mean():.6f})")
        
        return X_train, X_test, y_train, y_test
    
    def scale_features(self, X_train, X_test):
        """Scale numerical features"""
        # Identify numerical columns
        numerical_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
        
        # Fit scaler on training data
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()
        
        if len(numerical_cols) > 0:
            X_train_scaled[numerical_cols] = self.scaler.fit_transform(X_train[numerical_cols])
            X_test_scaled[numerical_cols] = self.scaler.transform(X_test[numerical_cols])
            print(f"Scaled {len(numerical_cols)} numerical features")
        else:
            print("No numerical features to scale")
        
        return X_train_scaled, X_test_scaled
    
    def apply_smote(self, X_train, y_train, sampling_strategy=0.1):
        """
        Apply SMOTE for handling class imbalance
        """
        try:
            from imblearn.over_sampling import SMOTE
            smote = SMOTE(sampling_strategy=sampling_strategy, random_state=self.random_state)
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
            
            print(f"SMOTE applied:")
            print(f"  Original: {len(X_train)} samples, fraud rate: {y_train.mean():.6f}")
            print(f"  Resampled: {len(X_train_resampled)} samples, fraud rate: {y_train_resampled.mean():.6f}")
            
            return X_train_resampled, y_train_resampled
        except ImportError:
            print("imbalanced-learn not installed. Skipping SMOTE.")
            return X_train, y_train
    
    def run_full_pipeline(self, filepath='data/transactions.csv', apply_smote=True):
        """Run complete preprocessing pipeline"""
        print("=" * 50)
        print("STARTING PREPROCESSING PIPELINE")
        print("=" * 50)
        
        # Load data
        df = self.load_data(filepath)
        
        # Clean data
        df_clean = self.clean_data(df)
        
        # Prepare features
        X, y = self.prepare_features(df_clean)
        
        # Split data
        X_train, X_test, y_train, y_test = self.split_data(X, y)
        
        # Scale features
        X_train_scaled, X_test_scaled = self.scale_features(X_train, X_test)
        
        # Apply SMOTE if requested
        if apply_smote:
            X_train_final, y_train_final = self.apply_smote(X_train_scaled, y_train)
        else:
            X_train_final, y_train_final = X_train_scaled, y_train
        
        print("\n" + "=" * 50)
        print("PREPROCESSING COMPLETE")
        print("=" * 50)
        
        return {
            'X_train': X_train_final,
            'X_test': X_test_scaled,
            'y_train': y_train_final,
            'y_test': y_test,
            'feature_names': self.feature_names
        }

if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    data = preprocessor.run_full_pipeline()
    print(f"\nFinal shapes - X_train: {data['X_train'].shape}, X_test: {data['X_test'].shape}")