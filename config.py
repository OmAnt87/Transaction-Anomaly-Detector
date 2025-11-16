"""
Configuration file for AML Transaction Anomaly Detector
"""

# High-risk countries (simplified list for demonstration)
HIGH_RISK_COUNTRIES = [
    'North Korea', 'Iran', 'Syria', 'Afghanistan', 'Yemen',
    'Somalia', 'Sudan', 'Libya', 'Iraq', 'Venezuela'
]

# Structuring detection thresholds
STRUCTURING_THRESHOLD = 10000  # Amount just below reporting threshold
STRUCTURING_WINDOW_DAYS = 7    # Time window to check for multiple transactions
STRUCTURING_COUNT = 3          # Number of transactions to trigger alert

# Rapid movement detection
RAPID_MOVEMENT_HOURS = 24      # Time window for rapid movement
RAPID_MOVEMENT_COUNT = 5       # Number of transactions to trigger alert

# Anomaly detection parameters
CONTAMINATION = 0.1            # Expected proportion of outliers
RANDOM_STATE = 42

# Baseline computation parameters
MIN_TRANSACTIONS_FOR_BASELINE = 10  # Minimum transactions needed to compute baseline

# Alert severity thresholds
SEVERITY_THRESHOLDS = {
    'low': 0.3,
    'medium': 0.5,
    'high': 0.7,
    'critical': 0.9
}
