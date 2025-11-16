"""
Machine Learning Anomaly Detection Engine
Implements ML-based anomaly detection using multiple algorithms
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from typing import Dict, Any, List, Tuple
import config


class MLAnomalyDetector:
    """ML-based anomaly detection using ensemble of algorithms"""
    
    def __init__(self):
        self.contamination = config.CONTAMINATION
        self.random_state = config.RANDOM_STATE
        
        # Initialize models
        self.isolation_forest = IsolationForest(
            contamination=self.contamination,
            random_state=self.random_state,
            n_estimators=100
        )
        
        self.one_class_svm = OneClassSVM(
            nu=self.contamination,
            kernel='rbf',
            gamma='auto'
        )
        
        self.lof = LocalOutlierFactor(
            contamination=self.contamination,
            novelty=True,
            n_neighbors=20
        )
        
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.feature_names = []
    
    def prepare_features(self, transactions: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare feature matrix for ML models
        
        Args:
            transactions: DataFrame with transaction data
            
        Returns:
            DataFrame with engineered features
        """
        df = transactions.copy()
        
        # Convert date to datetime
        df['date'] = pd.to_datetime(df['date'])
        
        # Temporal features
        df['hour'] = df['date'].dt.hour
        df['day_of_week'] = df['date'].dt.dayofweek
        df['day_of_month'] = df['date'].dt.day
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['is_night'] = df['hour'].isin(range(0, 6)).astype(int)
        
        # Amount features (log transform to handle skewness)
        df['log_amount'] = np.log1p(df['amount'])
        
        # Encode categorical variables
        df['country_encoded'] = pd.Categorical(df['country']).codes
        df['merchant_encoded'] = pd.Categorical(df['merchant']).codes
        df['channel_encoded'] = pd.Categorical(df['channel']).codes
        
        # Customer-level aggregations (if customer_id available)
        if 'customer_id' in df.columns:
            customer_stats = df.groupby('customer_id')['amount'].agg([
                ('customer_mean', 'mean'),
                ('customer_std', 'std'),
                ('customer_max', 'max'),
                ('customer_count', 'count')
            ]).reset_index()
            df = df.merge(customer_stats, on='customer_id', how='left')
            
            # Deviation from customer mean
            df['amount_deviation'] = (df['amount'] - df['customer_mean']) / (df['customer_std'] + 1)
        
        # Select features for modeling
        feature_cols = [
            'log_amount', 'hour', 'day_of_week', 'day_of_month',
            'is_weekend', 'is_night', 'country_encoded', 
            'merchant_encoded', 'channel_encoded'
        ]
        
        # Add customer features if available
        if 'customer_id' in df.columns:
            feature_cols.extend(['amount_deviation', 'customer_count'])
        
        self.feature_names = feature_cols
        
        return df[feature_cols].fillna(0)
    
    def train(self, transactions: pd.DataFrame):
        """
        Train all anomaly detection models
        
        Args:
            transactions: Historical transaction data
        """
        # Prepare features
        X = self.prepare_features(transactions)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train models
        print("Training Isolation Forest...")
        self.isolation_forest.fit(X_scaled)
        
        print("Training One-Class SVM...")
        self.one_class_svm.fit(X_scaled)
        
        print("Training Local Outlier Factor...")
        self.lof.fit(X_scaled)
        
        self.is_fitted = True
        print("All models trained successfully!")
    
    def predict(self, transactions: pd.DataFrame) -> pd.DataFrame:
        """
        Predict anomalies using ensemble of models
        
        Args:
            transactions: Transactions to score
            
        Returns:
            DataFrame with anomaly scores and predictions
        """
        if not self.is_fitted:
            raise ValueError("Models must be trained before prediction. Call train() first.")
        
        # Prepare features
        X = self.prepare_features(transactions)
        X_scaled = self.scaler.transform(X)
        
        # Get predictions from each model
        if_pred = self.isolation_forest.predict(X_scaled)
        if_score = self.isolation_forest.score_samples(X_scaled)
        
        svm_pred = self.one_class_svm.predict(X_scaled)
        svm_score = self.one_class_svm.score_samples(X_scaled)
        
        lof_pred = self.lof.predict(X_scaled)
        lof_score = self.lof.score_samples(X_scaled)
        
        # Create results DataFrame
        results = pd.DataFrame({
            'isolation_forest_pred': if_pred,
            'isolation_forest_score': if_score,
            'one_class_svm_pred': svm_pred,
            'one_class_svm_score': svm_score,
            'lof_pred': lof_pred,
            'lof_score': lof_score,
        })
        
        # Normalize scores to 0-1 range (higher = more anomalous)
        results['if_anomaly_score'] = self._normalize_score(-if_score)
        results['svm_anomaly_score'] = self._normalize_score(-svm_score)
        results['lof_anomaly_score'] = self._normalize_score(-lof_score)
        
        # Ensemble score (average of all models)
        results['ensemble_score'] = (
            results['if_anomaly_score'] + 
            results['svm_anomaly_score'] + 
            results['lof_anomaly_score']
        ) / 3
        
        # Ensemble prediction (majority vote: -1 = anomaly, 1 = normal)
        results['ensemble_pred'] = (
            (if_pred + svm_pred + lof_pred) / 3
        ).apply(lambda x: -1 if x < 0 else 1)
        
        # Count how many models flagged as anomaly
        results['num_models_flagged'] = (
            (if_pred == -1).astype(int) + 
            (svm_pred == -1).astype(int) + 
            (lof_pred == -1).astype(int)
        )
        
        return results
    
    def _normalize_score(self, scores: np.ndarray) -> np.ndarray:
        """Normalize scores to 0-1 range using min-max scaling"""
        min_score = scores.min()
        max_score = scores.max()
        if max_score - min_score == 0:
            return np.zeros_like(scores)
        return (scores - min_score) / (max_score - min_score)
    
    def get_feature_importance(self, transaction_features: pd.DataFrame) -> Dict[str, float]:
        """
        Get feature importance for anomaly detection
        Uses feature deviation from mean as proxy for importance
        
        Args:
            transaction_features: Prepared features for a transaction
            
        Returns:
            Dictionary of feature importance scores
        """
        if not self.is_fitted:
            return {}
        
        importance = {}
        
        # Calculate how much each feature deviates from training mean
        for feature in self.feature_names:
            if feature in transaction_features.columns:
                value = transaction_features[feature].iloc[0]
                # Simple importance: absolute deviation from 0 (scaled data)
                importance[feature] = abs(value)
        
        # Normalize to sum to 1
        total = sum(importance.values())
        if total > 0:
            importance = {k: v/total for k, v in importance.items()}
        
        return importance
    
    def explain_anomaly(self, transaction_idx: int, transactions: pd.DataFrame, 
                       ml_results: pd.DataFrame) -> Dict[str, Any]:
        """
        Explain why a transaction was flagged as anomalous
        
        Args:
            transaction_idx: Index of transaction to explain
            transactions: Original transaction data
            ml_results: Results from predict()
            
        Returns:
            Dictionary with explanation details
        """
        # Get ML scores
        ensemble_score = ml_results.loc[transaction_idx, 'ensemble_score']
        num_models = ml_results.loc[transaction_idx, 'num_models_flagged']
        
        # Get individual model scores
        if_score = ml_results.loc[transaction_idx, 'if_anomaly_score']
        svm_score = ml_results.loc[transaction_idx, 'svm_anomaly_score']
        lof_score = ml_results.loc[transaction_idx, 'lof_anomaly_score']
        
        explanation = {
            'ml_ensemble_score': ensemble_score,
            'num_models_flagged': num_models,
            'model_scores': {
                'isolation_forest': if_score,
                'one_class_svm': svm_score,
                'local_outlier_factor': lof_score
            },
            'is_anomaly': num_models >= 2,  # Flagged if 2+ models agree
            'confidence': ensemble_score
        }
        
        return explanation

