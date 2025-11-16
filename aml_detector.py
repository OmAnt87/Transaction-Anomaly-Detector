"""
Main AML Transaction Anomaly Detector
Orchestrates all detection engines
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from baseline_engine import BaselineEngine
from rule_engine import RuleEngine
from ml_engine import MLAnomalyDetector
from explainability_engine import ExplainabilityEngine
import config


class AMLDetector:
    """Main AML detection system that orchestrates all engines"""
    
    def __init__(self):
        self.baseline_engine = BaselineEngine()
        self.rule_engine = RuleEngine()
        self.ml_engine = MLAnomalyDetector()
        self.explainability_engine = ExplainabilityEngine()
        self.is_trained = False
        self.transactions_df = None
    
    def load_data(self, filepath: str) -> pd.DataFrame:
        """
        Load transaction data from CSV
        
        Args:
            filepath: Path to CSV file
            
        Returns:
            DataFrame with transaction data
        """
        df = pd.read_csv(filepath)
        
        # Validate required columns
        required_cols = ['date', 'amount', 'merchant', 'country', 'channel']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Add transaction_id if not present
        if 'transaction_id' not in df.columns:
            df['transaction_id'] = [f"TXN_{i:06d}" for i in range(len(df))]
        
        # Add customer_id if not present (assume single customer)
        if 'customer_id' not in df.columns:
            df['customer_id'] = 'CUST_001'
        
        self.transactions_df = df
        return df
    
    def train(self, training_data: Optional[pd.DataFrame] = None):
        """
        Train the AML detection system
        
        Args:
            training_data: Optional training data. If None, uses loaded data.
        """
        if training_data is None:
            if self.transactions_df is None:
                raise ValueError("No data loaded. Call load_data() first or provide training_data.")
            training_data = self.transactions_df
        
        print("Training AML Detection System...")
        print(f"Training data: {len(training_data)} transactions")
        
        # Compute baselines for each customer
        print("\n1. Computing customer baselines...")
        for customer_id in training_data['customer_id'].unique():
            customer_txns = training_data[training_data['customer_id'] == customer_id]
            baseline = self.baseline_engine.compute_baseline(customer_id, customer_txns)
            print(f"   Customer {customer_id}: {baseline['num_transactions']} transactions, "
                  f"avg amount: ${baseline['avg_amount']:,.2f}")
        
        # Train ML models
        print("\n2. Training ML anomaly detection models...")
        self.ml_engine.train(training_data)
        
        self.is_trained = True
        print("\n✓ Training complete!")
    
    def analyze_transaction(self, transaction: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a single transaction
        
        Args:
            transaction: Transaction dictionary
            
        Returns:
            Analysis results with explanation
        """
        if not self.is_trained:
            raise ValueError("System must be trained first. Call train().")
        
        customer_id = transaction.get('customer_id', 'CUST_001')
        
        # Get customer baseline
        baseline = self.baseline_engine.get_baseline(customer_id)
        
        # Compare to baseline
        baseline_deviations = self.baseline_engine.compare_to_baseline(transaction, baseline)
        
        # Apply rules
        rule_violations = self.rule_engine.apply_all_rules(
            customer_id, transaction, baseline, self.transactions_df
        )
        
        # ML prediction (need to create DataFrame for single transaction)
        txn_df = pd.DataFrame([transaction])
        ml_results = self.ml_engine.predict(txn_df)
        ml_explanation = self.ml_engine.explain_anomaly(0, txn_df, ml_results)
        
        # Generate explanation
        explanation = self.explainability_engine.generate_explanation(
            transaction, baseline, baseline_deviations, rule_violations, ml_explanation
        )
        
        return explanation
    
    def analyze_batch(self, transactions: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze a batch of transactions
        
        Args:
            transactions: DataFrame of transactions to analyze
            
        Returns:
            DataFrame with analysis results
        """
        if not self.is_trained:
            raise ValueError("System must be trained first. Call train().")
        
        print(f"Analyzing {len(transactions)} transactions...")
        
        results = []
        
        # Get ML predictions for all transactions at once
        ml_results = self.ml_engine.predict(transactions)
        
        # Analyze each transaction
        for idx, row in transactions.iterrows():
            transaction = row.to_dict()
            customer_id = transaction.get('customer_id', 'CUST_001')
            
            # Get customer baseline
            baseline = self.baseline_engine.get_baseline(customer_id)
            
            # Compare to baseline
            baseline_deviations = self.baseline_engine.compare_to_baseline(transaction, baseline)
            
            # Apply rules
            rule_violations = self.rule_engine.apply_all_rules(
                customer_id, transaction, baseline, self.transactions_df
            )
            
            # Get ML explanation
            ml_explanation = self.ml_engine.explain_anomaly(idx, transactions, ml_results)
            
            # Generate explanation
            explanation = self.explainability_engine.generate_explanation(
                transaction, baseline, baseline_deviations, rule_violations, ml_explanation
            )
            
            results.append(explanation)
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        
        # Add original transaction data
        results_df = pd.concat([transactions.reset_index(drop=True), results_df], axis=1)
        
        return results_df
    
    def get_flagged_transactions(self, results: pd.DataFrame, 
                                min_severity: str = 'medium') -> pd.DataFrame:
        """
        Filter for flagged transactions above a severity threshold
        
        Args:
            results: Results from analyze_batch()
            min_severity: Minimum severity level ('low', 'medium', 'high', 'critical')
            
        Returns:
            Filtered DataFrame
        """
        severity_order = {'low': 0, 'medium': 1, 'high': 2, 'critical': 3}
        min_level = severity_order[min_severity]
        
        flagged = results[results['severity'].map(severity_order) >= min_level]
        
        return flagged.sort_values('risk_score', ascending=False)
    
    def generate_report(self, results: pd.DataFrame) -> str:
        """
        Generate a summary report
        
        Args:
            results: Results from analyze_batch()
            
        Returns:
            Formatted report string
        """
        total = len(results)
        flagged = results[results['is_flagged'] == True]
        num_flagged = len(flagged)
        
        severity_counts = results['severity'].value_counts().to_dict()
        
        report = []
        report.append("=" * 80)
        report.append("AML TRANSACTION ANALYSIS REPORT")
        report.append("=" * 80)
        report.append(f"Total Transactions Analyzed: {total}")
        report.append(f"Flagged Transactions: {num_flagged} ({num_flagged/total*100:.1f}%)")
        report.append("")
        
        report.append("SEVERITY BREAKDOWN:")
        report.append("-" * 80)
        for severity in ['critical', 'high', 'medium', 'low']:
            count = severity_counts.get(severity, 0)
            pct = count / total * 100 if total > 0 else 0
            report.append(f"  {severity.upper():10s}: {count:4d} ({pct:5.1f}%)")
        report.append("")
        
        if num_flagged > 0:
            report.append("TOP 5 HIGHEST RISK TRANSACTIONS:")
            report.append("-" * 80)
            top_flagged = flagged.nlargest(5, 'risk_score')
            for idx, row in top_flagged.iterrows():
                report.append(f"Transaction {row['transaction_id']}:")
                report.append(f"  Risk Score: {row['risk_score']:.2f}")
                report.append(f"  Severity: {row['severity'].upper()}")
                if 'primary_reasons' in row and isinstance(row['primary_reasons'], list):
                    report.append(f"  Primary Reason: {row['primary_reasons'][0]['explanation']}")
                report.append("")
        
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def export_results(self, results: pd.DataFrame, filepath: str):
        """
        Export results to CSV
        
        Args:
            results: Results DataFrame
            filepath: Output file path
        """
        # Flatten nested columns for CSV export
        export_df = results.copy()
        
        # Convert list/dict columns to strings
        for col in export_df.columns:
            if export_df[col].dtype == 'object':
                export_df[col] = export_df[col].astype(str)
        
        export_df.to_csv(filepath, index=False)
        print(f"Results exported to {filepath}")

