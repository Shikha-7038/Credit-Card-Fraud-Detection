"""
Synthetic Credit Card Transaction Data Generator
Generates realistic transaction data with fraud patterns
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

class FraudDataGenerator:
    def __init__(self, random_state=42):
        self.random_state = random_state
        np.random.seed(random_state)
        random.seed(random_state)
    
    def generate_transactions(self, n_transactions=50000, fraud_ratio=0.005):
        """
        Generate synthetic transaction data
        
        Parameters:
        - n_transactions: Total number of transactions
        - fraud_ratio: Proportion of fraudulent transactions (default 0.5%)
        
        Returns:
        - DataFrame with transaction data
        """
        
        n_fraud = int(n_transactions * fraud_ratio)
        n_normal = n_transactions - n_fraud
        
        # Generate timestamps (last 30 days)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        timestamps = [start_date + timedelta(seconds=np.random.randint(0, 30*24*3600)) 
                      for _ in range(n_transactions)]
        timestamps.sort()
        
        # Initialize lists for data (using lists for efficiency)
        transaction_ids = [f'TX_{i:08d}' for i in range(n_transactions)]
        customer_ids = []
        merchant_categories = []
        locations = []
        device_types = []
        amounts = []
        hours = []
        day_of_weeks = []
        days_since_last = []
        avg_amounts = []
        tx_counts = []
        is_internationals = []
        is_nights = []
        is_frauds = [0] * n_transactions
        
        # Categories
        merchant_cats = ['retail', 'grocery', 'restaurant', 'travel', 'entertainment', 
                         'healthcare', 'utilities', 'online', 'gas_station', 'education']
        device_types_list = ['mobile', 'desktop', 'tablet']
        locations_list = ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix', 
                          'Philadelphia', 'San Antonio', 'San Diego', 'Dallas', 'Austin']
        customers = [f'CUST_{i:06d}' for i in range(1000)]  # 1000 unique customers
        
        # Pre-assign fraud indices
        all_indices = list(range(n_transactions))
        fraud_indices = set(np.random.choice(all_indices, n_fraud, replace=False))
        
        # Generate features for each transaction
        for idx in range(n_transactions):
            # Basic features
            customer = np.random.choice(customers)
            customer_ids.append(customer)
            
            merchant = np.random.choice(merchant_cats)
            merchant_categories.append(merchant)
            
            location = np.random.choice(locations_list)
            locations.append(location)
            
            device = np.random.choice(device_types_list)
            device_types.append(device)
            
            # Time features
            hour = timestamps[idx].hour
            hours.append(hour)
            day_of_weeks.append(timestamps[idx].weekday())
            is_nights.append(1 if (hour < 6 or hour > 22) else 0)
            
            # International (10% of transactions)
            is_internationals.append(1 if np.random.random() < 0.1 else 0)
            
            # Amount - depends on fraud status
            if idx not in fraud_indices:
                # Normal transaction: typical amounts
                if merchant == 'travel':
                    amount = np.random.gamma(2, 200)
                elif merchant == 'restaurant':
                    amount = np.random.gamma(2, 50)
                elif merchant == 'grocery':
                    amount = np.random.gamma(2, 60)
                elif merchant == 'online':
                    amount = np.random.gamma(2, 100)
                else:
                    amount = np.random.gamma(2, 80)
                amount = max(5, min(amount, 5000))
                amounts.append(amount)
                is_frauds[idx] = 0
            else:
                # Fraudulent: unusual amounts
                if np.random.random() < 0.7:
                    # Large transactions
                    amount = np.random.uniform(1000, 10000)
                else:
                    # Many small transactions
                    amount = np.random.uniform(5, 100)
                amounts.append(amount)
                is_frauds[idx] = 1
                
                # Modify some features for fraud patterns
                if np.random.random() < 0.5:
                    # Unusual hour (night)
                    hours[idx] = np.random.choice([2, 3, 4, 23])
                    is_nights[idx] = 1
                
                if np.random.random() < 0.4:
                    # International for fraud
                    is_internationals[idx] = 1
            
            # Customer history features (simulate)
            if idx > 0 and customer == customer_ids[idx-1]:
                time_diff = (timestamps[idx] - timestamps[idx-1]).total_seconds() / 3600
                days_since_last.append(time_diff / 24)
            else:
                days_since_last.append(np.random.exponential(2))
            
            avg_amounts.append(np.random.gamma(2, 80))
            tx_counts.append(np.random.poisson(2))
        
        # Create DataFrame
        df = pd.DataFrame({
            'transaction_id': transaction_ids,
            'timestamp': timestamps,
            'customer_id': customer_ids,
            'amount': amounts,
            'merchant_category': merchant_categories,
            'location': locations,
            'device_type': device_types,
            'transaction_hour': hours,
            'day_of_week': day_of_weeks,
            'days_since_last_tx': days_since_last,
            'avg_transaction_amount': avg_amounts,
            'transaction_count_24h': tx_counts,
            'is_international': is_internationals,
            'is_night': is_nights,
            'is_fraud': is_frauds
        })
        
        # Add engineered features
        df['log_amount'] = np.log1p(df['amount'])
        df['amount_normalized'] = df['amount'] / df['avg_transaction_amount'].replace(0, 1)
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        print(f"Generated {n_transactions} transactions")
        print(f"Fraud ratio: {df['is_fraud'].mean():.4f} ({df['is_fraud'].sum():,} fraud cases)")
        
        return df
    
    def save_data(self, df, filepath='data/transactions.csv'):
        """Save data to CSV"""
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        df.to_csv(filepath, index=False)
        print(f"Data saved to {filepath}")
        return filepath

# Test the generator
if __name__ == "__main__":
    generator = FraudDataGenerator(random_state=42)
    df = generator.generate_transactions(n_transactions=50000, fraud_ratio=0.005)
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nData types:")
    print(df.dtypes)
    print("\nColumn names:")
    print(df.columns.tolist())