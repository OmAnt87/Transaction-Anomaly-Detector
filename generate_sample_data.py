"""
Generate sample transaction data for testing the AML detector
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random


def generate_sample_transactions(num_transactions=1000, num_customers=5, 
                                 anomaly_rate=0.15, seed=42):
    """
    Generate sample transaction data with normal and anomalous transactions
    
    Args:
        num_transactions: Total number of transactions to generate
        num_customers: Number of unique customers
        anomaly_rate: Proportion of anomalous transactions
        seed: Random seed for reproducibility
    """
    np.random.seed(seed)
    random.seed(seed)
    
    # Define normal patterns for each customer
    customer_patterns = {}
    for i in range(num_customers):
        customer_id = f"CUST_{i+1:03d}"
        customer_patterns[customer_id] = {
            'avg_amount': np.random.uniform(50, 500),
            'std_amount': np.random.uniform(20, 100),
            'common_countries': random.sample(['USA', 'UK', 'Canada', 'Germany', 'France'], 2),
            'common_merchants': random.sample([
                'Amazon', 'Walmart', 'Target', 'Starbucks', 'Shell Gas',
                'McDonalds', 'Apple Store', 'Best Buy', 'Home Depot', 'CVS'
            ], 3),
            'common_channels': random.sample(['online', 'pos', 'atm', 'mobile'], 2),
            'common_hours': list(range(8, 20))  # 8 AM to 8 PM
        }
    
    transactions = []
    start_date = datetime.now() - timedelta(days=180)
    
    num_normal = int(num_transactions * (1 - anomaly_rate))
    num_anomalous = num_transactions - num_normal
    
    # Generate normal transactions
    for i in range(num_normal):
        customer_id = random.choice(list(customer_patterns.keys()))
        pattern = customer_patterns[customer_id]
        
        # Normal transaction following customer pattern
        amount = max(1, np.random.normal(pattern['avg_amount'], pattern['std_amount']))
        country = random.choice(pattern['common_countries'])
        merchant = random.choice(pattern['common_merchants'])
        channel = random.choice(pattern['common_channels'])
        hour = random.choice(pattern['common_hours'])
        
        # Random date within last 6 months
        days_ago = random.randint(0, 180)
        date = start_date + timedelta(days=days_ago, hours=hour, minutes=random.randint(0, 59))
        
        transactions.append({
            'transaction_id': f"TXN_{i+1:06d}",
            'customer_id': customer_id,
            'date': date.strftime('%Y-%m-%d %H:%M:%S'),
            'amount': round(amount, 2),
            'merchant': merchant,
            'country': country,
            'channel': channel,
            'is_anomaly': False
        })
    
    # Generate anomalous transactions
    anomaly_types = [
        'high_risk_country',
        'large_amount',
        'unusual_time',
        'unusual_merchant',
        'structuring',
        'rapid_movement'
    ]
    
    for i in range(num_anomalous):
        customer_id = random.choice(list(customer_patterns.keys()))
        pattern = customer_patterns[customer_id]
        
        anomaly_type = random.choice(anomaly_types)
        
        # Base transaction
        amount = max(1, np.random.normal(pattern['avg_amount'], pattern['std_amount']))
        country = random.choice(pattern['common_countries'])
        merchant = random.choice(pattern['common_merchants'])
        channel = random.choice(pattern['common_channels'])
        hour = random.choice(pattern['common_hours'])
        days_ago = random.randint(0, 180)
        date = start_date + timedelta(days=days_ago, hours=hour, minutes=random.randint(0, 59))
        
        # Modify based on anomaly type
        if anomaly_type == 'high_risk_country':
            country = random.choice(['Iran', 'North Korea', 'Syria', 'Venezuela', 'Somalia'])
        
        elif anomaly_type == 'large_amount':
            amount = pattern['avg_amount'] * random.uniform(5, 15)
        
        elif anomaly_type == 'unusual_time':
            hour = random.choice([2, 3, 4, 5])  # Late night
            date = start_date + timedelta(days=days_ago, hours=hour, minutes=random.randint(0, 59))
        
        elif anomaly_type == 'unusual_merchant':
            merchant = random.choice(['Unknown Merchant', 'Cash Advance', 'Wire Transfer', 'Crypto Exchange'])
        
        elif anomaly_type == 'structuring':
            # Multiple transactions just below $10,000
            amount = random.uniform(9000, 9900)
        
        elif anomaly_type == 'rapid_movement':
            # Transaction in unusual country with different channel
            country = random.choice(['Brazil', 'Russia', 'China', 'India', 'Mexico'])
            channel = 'atm'
        
        transactions.append({
            'transaction_id': f"TXN_{num_normal + i + 1:06d}",
            'customer_id': customer_id,
            'date': date.strftime('%Y-%m-%d %H:%M:%S'),
            'amount': round(amount, 2),
            'merchant': merchant,
            'country': country,
            'channel': channel,
            'is_anomaly': True
        })
    
    # Create DataFrame and shuffle
    df = pd.DataFrame(transactions)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    # Sort by date
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    df['date'] = df['date'].astype(str)
    
    return df


if __name__ == '__main__':
    # Generate sample data
    print("Generating sample transaction data...")
    
    # Training data (historical, mostly normal)
    training_data = generate_sample_transactions(
        num_transactions=800,
        num_customers=5,
        anomaly_rate=0.05,  # Low anomaly rate for training
        seed=42
    )
    
    # Test data (recent transactions with more anomalies)
    test_data = generate_sample_transactions(
        num_transactions=200,
        num_customers=5,
        anomaly_rate=0.20,  # Higher anomaly rate for testing
        seed=123
    )
    
    # Save to CSV
    training_data.to_csv('transactions_training.csv', index=False)
    test_data.to_csv('transactions_test.csv', index=False)
    
    print(f"✓ Generated training data: {len(training_data)} transactions")
    print(f"  - Normal: {len(training_data[training_data['is_anomaly']==False])}")
    print(f"  - Anomalous: {len(training_data[training_data['is_anomaly']==True])}")
    
    print(f"\n✓ Generated test data: {len(test_data)} transactions")
    print(f"  - Normal: {len(test_data[test_data['is_anomaly']==False])}")
    print(f"  - Anomalous: {len(test_data[test_data['is_anomaly']==True])}")
    
    print("\nFiles saved:")
    print("  - transactions_training.csv")
    print("  - transactions_test.csv")

