"""
Customer Baseline Computation Engine
Computes normal behavior patterns for each customer
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any


class BaselineEngine:
    """Computes and stores customer baseline behavior patterns"""
    
    def __init__(self):
        self.baselines = {}
    
    def compute_baseline(self, customer_id: str, transactions: pd.DataFrame) -> Dict[str, Any]:
        """
        Compute baseline behavior for a customer
        
        Args:
            customer_id: Customer identifier
            transactions: DataFrame with customer's historical transactions
            
        Returns:
            Dictionary containing baseline metrics
        """
        if len(transactions) == 0:
            return self._empty_baseline()
        
        # Convert date to datetime if needed
        if 'date' in transactions.columns:
            transactions['date'] = pd.to_datetime(transactions['date'])
            transactions['hour'] = transactions['date'].dt.hour
            transactions['day_of_week'] = transactions['date'].dt.dayofweek
        
        baseline = {
            'customer_id': customer_id,
            'num_transactions': len(transactions),
            
            # Amount statistics
            'avg_amount': transactions['amount'].mean(),
            'median_amount': transactions['amount'].median(),
            'std_amount': transactions['amount'].std(),
            'min_amount': transactions['amount'].min(),
            'max_amount': transactions['amount'].max(),
            'q25_amount': transactions['amount'].quantile(0.25),
            'q75_amount': transactions['amount'].quantile(0.75),
            
            # Location patterns
            'common_countries': transactions['country'].value_counts().head(5).to_dict(),
            'num_unique_countries': transactions['country'].nunique(),
            
            # Merchant patterns
            'common_merchants': transactions['merchant'].value_counts().head(5).to_dict(),
            'num_unique_merchants': transactions['merchant'].nunique(),
            
            # Channel patterns
            'common_channels': transactions['channel'].value_counts().to_dict(),
            
            # Temporal patterns
            'common_hours': transactions['hour'].value_counts().head(5).to_dict() if 'hour' in transactions.columns else {},
            'common_days': transactions['day_of_week'].value_counts().to_dict() if 'day_of_week' in transactions.columns else {},
            
            # Frequency
            'avg_transactions_per_day': self._compute_transaction_frequency(transactions),
        }
        
        self.baselines[customer_id] = baseline
        return baseline
    
    def _empty_baseline(self) -> Dict[str, Any]:
        """Return empty baseline for customers with no history"""
        return {
            'customer_id': None,
            'num_transactions': 0,
            'avg_amount': 0,
            'median_amount': 0,
            'std_amount': 0,
            'min_amount': 0,
            'max_amount': 0,
            'q25_amount': 0,
            'q75_amount': 0,
            'common_countries': {},
            'num_unique_countries': 0,
            'common_merchants': {},
            'num_unique_merchants': 0,
            'common_channels': {},
            'common_hours': {},
            'common_days': {},
            'avg_transactions_per_day': 0,
        }
    
    def _compute_transaction_frequency(self, transactions: pd.DataFrame) -> float:
        """Compute average transactions per day"""
        if 'date' not in transactions.columns or len(transactions) < 2:
            return 0
        
        date_range = (transactions['date'].max() - transactions['date'].min()).days
        if date_range == 0:
            return len(transactions)
        
        return len(transactions) / date_range
    
    def get_baseline(self, customer_id: str) -> Dict[str, Any]:
        """Retrieve baseline for a customer"""
        return self.baselines.get(customer_id, self._empty_baseline())
    
    def compare_to_baseline(self, transaction: Dict[str, Any], baseline: Dict[str, Any]) -> Dict[str, float]:
        """
        Compare a transaction to baseline and return deviation scores
        
        Args:
            transaction: Single transaction dictionary
            baseline: Customer baseline dictionary
            
        Returns:
            Dictionary of deviation scores (0-1, higher = more anomalous)
        """
        if baseline['num_transactions'] == 0:
            return {'insufficient_history': 1.0}
        
        deviations = {}
        
        # Amount deviation (using z-score)
        if baseline['std_amount'] > 0:
            z_score = abs((transaction['amount'] - baseline['avg_amount']) / baseline['std_amount'])
            deviations['amount_deviation'] = min(z_score / 3, 1.0)  # Normalize to 0-1
        else:
            deviations['amount_deviation'] = 0.0
        
        # Country deviation
        if transaction['country'] not in baseline['common_countries']:
            deviations['country_deviation'] = 1.0
        else:
            # Score based on frequency
            total_txns = baseline['num_transactions']
            country_freq = baseline['common_countries'][transaction['country']] / total_txns
            deviations['country_deviation'] = 1.0 - country_freq
        
        # Merchant deviation
        if transaction['merchant'] not in baseline['common_merchants']:
            deviations['merchant_deviation'] = 0.7  # New merchant is somewhat unusual
        else:
            total_txns = baseline['num_transactions']
            merchant_freq = baseline['common_merchants'][transaction['merchant']] / total_txns
            deviations['merchant_deviation'] = 1.0 - merchant_freq
        
        # Channel deviation
        if transaction['channel'] not in baseline['common_channels']:
            deviations['channel_deviation'] = 0.8
        else:
            total_txns = baseline['num_transactions']
            channel_freq = baseline['common_channels'][transaction['channel']] / total_txns
            deviations['channel_deviation'] = 1.0 - channel_freq
        
        # Time deviation (if hour available)
        if 'hour' in transaction and baseline['common_hours']:
            hour = transaction['hour']
            if hour not in baseline['common_hours']:
                deviations['time_deviation'] = 0.8
            else:
                total_txns = baseline['num_transactions']
                hour_freq = baseline['common_hours'][hour] / total_txns
                deviations['time_deviation'] = 1.0 - hour_freq
        
        return deviations

