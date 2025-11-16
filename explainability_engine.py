"""
Explainability Engine
Provides human-readable explanations for why transactions were flagged
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List
from baseline_engine import BaselineEngine
from rule_engine import RuleEngine
from ml_engine import MLAnomalyDetector


class ExplainabilityEngine:
    """Generates comprehensive explanations for flagged transactions"""
    
    def __init__(self):
        self.severity_levels = {
            'low': 0.3,
            'medium': 0.5,
            'high': 0.7,
            'critical': 0.9
        }
    
    def generate_explanation(self, 
                           transaction: Dict[str, Any],
                           baseline: Dict[str, Any],
                           baseline_deviations: Dict[str, float],
                           rule_violations: List[Dict[str, Any]],
                           ml_explanation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate comprehensive explanation for a flagged transaction
        
        Args:
            transaction: The transaction being analyzed
            baseline: Customer baseline
            baseline_deviations: Deviations from baseline
            rule_violations: List of rule violations
            ml_explanation: ML model explanation
            
        Returns:
            Comprehensive explanation dictionary
        """
        # Calculate overall risk score
        risk_score = self._calculate_risk_score(
            baseline_deviations, rule_violations, ml_explanation
        )
        
        # Determine severity
        severity = self._get_severity(risk_score)
        
        # Generate human-readable reasons
        reasons = self._generate_reasons(
            transaction, baseline, baseline_deviations, rule_violations, ml_explanation
        )
        
        # Create detailed explanation
        explanation = {
            'transaction_id': transaction.get('transaction_id', 'N/A'),
            'customer_id': transaction.get('customer_id', 'N/A'),
            'risk_score': risk_score,
            'severity': severity,
            'is_flagged': risk_score >= self.severity_levels['medium'],
            'primary_reasons': reasons[:3],  # Top 3 reasons
            'all_reasons': reasons,
            'num_rule_violations': len(rule_violations),
            'ml_confidence': ml_explanation.get('confidence', 0),
            'num_ml_models_flagged': ml_explanation.get('num_models_flagged', 0),
            'breakdown': {
                'baseline_deviations': baseline_deviations,
                'rule_violations': rule_violations,
                'ml_scores': ml_explanation.get('model_scores', {})
            },
            'recommendation': self._get_recommendation(severity, reasons)
        }
        
        return explanation
    
    def _calculate_risk_score(self, 
                             baseline_deviations: Dict[str, float],
                             rule_violations: List[Dict[str, Any]],
                             ml_explanation: Dict[str, Any]) -> float:
        """
        Calculate overall risk score (0-1)
        
        Combines:
        - Baseline deviations (30%)
        - Rule violations (40%)
        - ML anomaly score (30%)
        """
        # Baseline deviation score (average of all deviations)
        if baseline_deviations:
            baseline_score = np.mean(list(baseline_deviations.values()))
        else:
            baseline_score = 0
        
        # Rule violation score (weighted by severity)
        if rule_violations:
            severity_weights = {'low': 0.25, 'medium': 0.5, 'high': 0.75, 'critical': 1.0}
            rule_scores = [severity_weights.get(v['severity'], 0.5) for v in rule_violations]
            rule_score = min(np.mean(rule_scores) * len(rule_violations) / 3, 1.0)
        else:
            rule_score = 0
        
        # ML score
        ml_score = ml_explanation.get('ml_ensemble_score', 0)
        
        # Weighted combination
        risk_score = (
            baseline_score * 0.3 +
            rule_score * 0.4 +
            ml_score * 0.3
        )
        
        return min(risk_score, 1.0)
    
    def _get_severity(self, risk_score: float) -> str:
        """Determine severity level from risk score"""
        if risk_score >= self.severity_levels['critical']:
            return 'critical'
        elif risk_score >= self.severity_levels['high']:
            return 'high'
        elif risk_score >= self.severity_levels['medium']:
            return 'medium'
        else:
            return 'low'
    
    def _generate_reasons(self,
                         transaction: Dict[str, Any],
                         baseline: Dict[str, Any],
                         baseline_deviations: Dict[str, float],
                         rule_violations: List[Dict[str, Any]],
                         ml_explanation: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate list of human-readable reasons
        
        Returns:
            List of reason dictionaries with score and explanation
        """
        reasons = []
        
        # Add rule violations (highest priority)
        for violation in rule_violations:
            reasons.append({
                'type': 'rule_violation',
                'score': self._severity_to_score(violation['severity']),
                'explanation': violation['explanation'],
                'category': violation['rule']
            })
        
        # Add significant baseline deviations
        for deviation_type, score in baseline_deviations.items():
            if score >= 0.5:  # Only include significant deviations
                explanation = self._explain_deviation(
                    deviation_type, score, transaction, baseline
                )
                if explanation:
                    reasons.append({
                        'type': 'baseline_deviation',
                        'score': score,
                        'explanation': explanation,
                        'category': deviation_type
                    })
        
        # Add ML-based reasons if confidence is high
        if ml_explanation.get('confidence', 0) >= 0.6:
            num_models = ml_explanation.get('num_models_flagged', 0)
            if num_models >= 2:
                reasons.append({
                    'type': 'ml_detection',
                    'score': ml_explanation['confidence'],
                    'explanation': f"Flagged by {num_models} out of 3 ML models as anomalous behavior",
                    'category': 'machine_learning'
                })
        
        # Sort by score (highest first)
        reasons.sort(key=lambda x: x['score'], reverse=True)
        
        return reasons
    
    def _explain_deviation(self, deviation_type: str, score: float, 
                          transaction: Dict[str, Any], baseline: Dict[str, Any]) -> str:
        """Generate human-readable explanation for a deviation"""
        
        if deviation_type == 'amount_deviation':
            amount = transaction['amount']
            avg = baseline['avg_amount']
            multiplier = amount / avg if avg > 0 else 0
            if multiplier > 1:
                return f"Transaction amount ${amount:,.2f} is {multiplier:.1f}× customer's average (${avg:,.2f})"
            else:
                return f"Transaction amount ${amount:,.2f} is unusually low compared to average (${avg:,.2f})"
        
        elif deviation_type == 'country_deviation':
            country = transaction['country']
            common = list(baseline.get('common_countries', {}).keys())[:3]
            if common:
                return f"Unusual location: {country} (customer typically uses {', '.join(common)})"
            else:
                return f"Transaction from new country: {country}"
        
        elif deviation_type == 'merchant_deviation':
            merchant = transaction['merchant']
            return f"New or rarely used merchant: {merchant}"
        
        elif deviation_type == 'channel_deviation':
            channel = transaction['channel']
            common = list(baseline.get('common_channels', {}).keys())
            if common:
                return f"Unusual channel: {channel} (customer typically uses {', '.join(common)})"
            else:
                return f"New channel: {channel}"
        
        elif deviation_type == 'time_deviation':
            hour = transaction.get('hour', 0)
            return f"Unusual transaction time: {hour:02d}:00 (outside customer's normal hours)"
        
        return ""
    
    def _severity_to_score(self, severity: str) -> float:
        """Convert severity level to numeric score"""
        return self.severity_levels.get(severity, 0.5)
    
    def _get_recommendation(self, severity: str, reasons: List[Dict[str, Any]]) -> str:
        """Generate action recommendation based on severity and reasons"""
        
        if severity == 'critical':
            return ("IMMEDIATE ACTION REQUIRED: Block transaction and escalate to compliance team. "
                   "High risk of money laundering or fraud.")
        
        elif severity == 'high':
            return ("URGENT REVIEW: Hold transaction for manual review by AML analyst. "
                   "Multiple risk indicators detected.")
        
        elif severity == 'medium':
            return ("REVIEW RECOMMENDED: Flag for review within 24 hours. "
                   "Monitor customer for additional suspicious activity.")
        
        else:
            return ("LOW PRIORITY: Log for record-keeping. "
                   "May be legitimate but slightly unusual behavior.")
    
    def format_explanation_text(self, explanation: Dict[str, Any]) -> str:
        """
        Format explanation as readable text
        
        Args:
            explanation: Explanation dictionary
            
        Returns:
            Formatted text string
        """
        lines = []
        lines.append("=" * 80)
        lines.append(f"TRANSACTION ALERT - {explanation['severity'].upper()} RISK")
        lines.append("=" * 80)
        lines.append(f"Transaction ID: {explanation['transaction_id']}")
        lines.append(f"Customer ID: {explanation['customer_id']}")
        lines.append(f"Risk Score: {explanation['risk_score']:.2f} / 1.00")
        lines.append(f"Severity: {explanation['severity'].upper()}")
        lines.append("")
        
        lines.append("PRIMARY REASONS FOR FLAGGING:")
        lines.append("-" * 80)
        for i, reason in enumerate(explanation['primary_reasons'], 1):
            lines.append(f"{i}. [{reason['type'].upper()}] {reason['explanation']}")
            lines.append(f"   Risk Score: {reason['score']:.2f}")
        lines.append("")
        
        if explanation['num_rule_violations'] > 0:
            lines.append(f"Rule Violations: {explanation['num_rule_violations']}")
        
        if explanation['num_ml_models_flagged'] > 0:
            lines.append(f"ML Models Flagged: {explanation['num_ml_models_flagged']} / 3")
            lines.append(f"ML Confidence: {explanation['ml_confidence']:.2f}")
        
        lines.append("")
        lines.append("RECOMMENDATION:")
        lines.append("-" * 80)
        lines.append(explanation['recommendation'])
        lines.append("=" * 80)
        
        return "\n".join(lines)

