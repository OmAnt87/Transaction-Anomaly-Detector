"""
Rule-Based Detection Engine
Implements business rules for AML detection
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple
import config


class RuleEngine:
    """Implements rule-based AML detection"""
    
    def __init__(self):
        self.high_risk_countries = config.HIGH_RISK_COUNTRIES
        self.structuring_threshold = config.STRUCTURING_THRESHOLD
        self.structuring_window_days = config.STRUCTURING_WINDOW_DAYS
        self.structuring_count = config.STRUCTURING_COUNT
        self.rapid_movement_hours = config.RAPID_MOVEMENT_HOURS
        self.rapid_movement_count = config.RAPID_MOVEMENT_COUNT
    
    def check_high_risk_country(self, transaction: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Check if transaction is from a high-risk country
        
        Returns:
            (is_flagged, explanation)
        """
        country = transaction.get('country', '')
        if country in self.high_risk_countries:
            return True, f"Transaction from high-risk country: {country}"
        return False, ""
    
    def check_structuring(self, customer_id: str, current_transaction: Dict[str, Any], 
                         all_transactions: pd.DataFrame) -> Tuple[bool, str]:
        """
        Check for structuring (multiple transactions just below reporting threshold)
        
        Structuring is when someone makes multiple smaller transactions to avoid
        reporting requirements (e.g., multiple $9,500 deposits to avoid $10,000 threshold)
        
        Returns:
            (is_flagged, explanation)
        """
        current_date = pd.to_datetime(current_transaction['date'])
        current_amount = current_transaction['amount']
        
        # Check if current transaction is just below threshold
        threshold_margin = 0.9  # Within 90% of threshold
        if current_amount < self.structuring_threshold * threshold_margin:
            return False, ""
        
        # Look for similar transactions in the time window
        window_start = current_date - timedelta(days=self.structuring_window_days)
        
        customer_txns = all_transactions[all_transactions['customer_id'] == customer_id].copy()
        customer_txns['date'] = pd.to_datetime(customer_txns['date'])
        
        recent_txns = customer_txns[
            (customer_txns['date'] >= window_start) & 
            (customer_txns['date'] <= current_date) &
            (customer_txns['amount'] >= self.structuring_threshold * threshold_margin) &
            (customer_txns['amount'] < self.structuring_threshold)
        ]
        
        if len(recent_txns) >= self.structuring_count:
            total_amount = recent_txns['amount'].sum()
            return True, (f"Potential structuring: {len(recent_txns)} transactions "
                         f"just below ${self.structuring_threshold:,.0f} threshold "
                         f"in {self.structuring_window_days} days (total: ${total_amount:,.2f})")
        
        return False, ""
    
    def check_rapid_movement(self, customer_id: str, current_transaction: Dict[str, Any],
                            all_transactions: pd.DataFrame) -> Tuple[bool, str]:
        """
        Check for rapid movement between accounts/locations
        
        Returns:
            (is_flagged, explanation)
        """
        current_date = pd.to_datetime(current_transaction['date'])
        window_start = current_date - timedelta(hours=self.rapid_movement_hours)
        
        customer_txns = all_transactions[all_transactions['customer_id'] == customer_id].copy()
        customer_txns['date'] = pd.to_datetime(customer_txns['date'])
        
        recent_txns = customer_txns[
            (customer_txns['date'] >= window_start) & 
            (customer_txns['date'] <= current_date)
        ]
        
        if len(recent_txns) >= self.rapid_movement_count:
            unique_countries = recent_txns['country'].nunique()
            unique_channels = recent_txns['channel'].nunique()
            
            if unique_countries >= 3 or unique_channels >= 3:
                return True, (f"Rapid movement: {len(recent_txns)} transactions "
                             f"across {unique_countries} countries and {unique_channels} channels "
                             f"in {self.rapid_movement_hours} hours")
        
        return False, ""
    
    def check_large_transaction(self, transaction: Dict[str, Any], baseline: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Check if transaction is significantly larger than customer's typical amount
        
        Returns:
            (is_flagged, explanation)
        """
        if baseline['num_transactions'] < 5:
            return False, ""
        
        amount = transaction['amount']
        avg_amount = baseline['avg_amount']
        
        # Flag if transaction is 5x or more than average
        if amount >= avg_amount * 5:
            multiplier = amount / avg_amount
            return True, f"Large transaction: {multiplier:.1f}× customer's average (${avg_amount:,.2f})"
        
        return False, ""
    
    def check_round_amount(self, transaction: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Check for suspiciously round amounts (common in money laundering)
        
        Returns:
            (is_flagged, explanation)
        """
        amount = transaction['amount']
        
        # Check if amount is a round number (e.g., 10000, 5000, 50000)
        if amount >= 1000:
            # Check if divisible by 1000 or 5000
            if amount % 5000 == 0 or amount % 1000 == 0:
                if amount >= 10000:
                    return True, f"Suspiciously round large amount: ${amount:,.2f}"
        
        return False, ""
    
    def check_unusual_channel(self, transaction: Dict[str, Any], baseline: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Check if transaction uses an unusual channel for this customer
        
        Returns:
            (is_flagged, explanation)
        """
        if baseline['num_transactions'] < 5:
            return False, ""
        
        channel = transaction['channel']
        common_channels = baseline.get('common_channels', {})
        
        if channel not in common_channels:
            return True, f"Unusual channel: {channel} (customer typically uses {', '.join(common_channels.keys())})"
        
        # Check if this channel is rarely used (< 10% of transactions)
        total_txns = baseline['num_transactions']
        channel_freq = common_channels[channel] / total_txns
        
        if channel_freq < 0.1:
            return True, f"Rarely used channel: {channel} (only {channel_freq*100:.1f}% of transactions)"
        
        return False, ""
    
    def apply_all_rules(self, customer_id: str, transaction: Dict[str, Any], 
                       baseline: Dict[str, Any], all_transactions: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Apply all rules to a transaction
        
        Returns:
            List of rule violations with explanations
        """
        violations = []
        
        # Check each rule
        rules_to_check = [
            ('high_risk_country', self.check_high_risk_country(transaction)),
            ('structuring', self.check_structuring(customer_id, transaction, all_transactions)),
            ('rapid_movement', self.check_rapid_movement(customer_id, transaction, all_transactions)),
            ('large_transaction', self.check_large_transaction(transaction, baseline)),
            ('round_amount', self.check_round_amount(transaction)),
            ('unusual_channel', self.check_unusual_channel(transaction, baseline)),
        ]
        
        for rule_name, (is_flagged, explanation) in rules_to_check:
            if is_flagged:
                violations.append({
                    'rule': rule_name,
                    'explanation': explanation,
                    'severity': self._get_rule_severity(rule_name)
                })
        
        return violations
    
    def _get_rule_severity(self, rule_name: str) -> str:
        """Get severity level for a rule"""
        severity_map = {
            'high_risk_country': 'high',
            'structuring': 'critical',
            'rapid_movement': 'high',
            'large_transaction': 'medium',
            'round_amount': 'low',
            'unusual_channel': 'low',
        }
        return severity_map.get(rule_name, 'medium')
