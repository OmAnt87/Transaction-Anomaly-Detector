# 🔍 AML Transaction Anomaly Detector

A comprehensive **Anti-Money Laundering (AML) Transaction Anomaly Detection System** that combines machine learning, rule-based detection, and customer baseline analysis with full explainability.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 🎯 Overview

This system mirrors what real banks use for AML compliance, featuring:

- **🤖 Machine Learning Detection**: Ensemble of Isolation Forest, One-Class SVM, and Local Outlier Factor
- **📋 Rule-Based Detection**: High-risk countries, structuring, rapid movement, and more
- **📊 Customer Baseline Analysis**: Personalized behavior patterns for each customer
- **💡 Explainability Engine**: Clear, human-readable explanations for every alert
- **🌐 Web Interface**: Beautiful Streamlit dashboard for visualization and analysis

## ✨ Key Features

### Detection Methods

1. **Machine Learning Models**
   - Isolation Forest: Detects anomalies by isolating outliers
   - One-Class SVM: Learns the boundary of normal behavior
   - Local Outlier Factor: Identifies local density deviations

2. **Rule-Based Detection**
   - High-risk country transactions
   - Structuring (multiple transactions just below reporting thresholds)
   - Rapid movement between accounts/locations
   - Large transactions (5x+ customer average)
   - Suspiciously round amounts
   - Unusual transaction channels

3. **Baseline Analysis**
   - Customer-specific spending patterns
   - Typical locations and merchants
   - Normal transaction times and frequencies
   - Deviation scoring from baseline behavior

4. **Explainability**
   - Detailed reasons for each alert
   - Risk score breakdown (0-1 scale)
   - Severity classification (Low/Medium/High/Critical)
   - Actionable recommendations

## 🚀 Quick Start

### Installation

```bash
# Clone or download the project
cd "AML Project 1"

# Install dependencies
pip install -r requirements.txt
```

### Generate Sample Data

```bash
python generate_sample_data.py
```

This creates:
- `transactions_training.csv`: 800 historical transactions (5% anomalies)
- `transactions_test.csv`: 200 recent transactions (20% anomalies)

### Launch Web Interface

```bash
streamlit run app.py
```

The web interface will open at `http://localhost:8501`

### Using the Web Interface

1. **Load Data**: Click "Use Sample Data" or upload your own CSV
2. **Train System**: Click "Train System" to build models and baselines
3. **Analyze**: Upload transactions to analyze or use test data
4. **Review**: Explore dashboard, detailed analysis, and visualizations

## 📊 CSV Format

Your transaction CSV should include these columns:

```csv
date,amount,merchant,country,channel,customer_id
2024-01-15 14:30:00,125.50,Amazon,USA,online,CUST_001
2024-01-15 18:45:00,45.00,Starbucks,USA,pos,CUST_001
2024-01-16 02:30:00,9500.00,Wire Transfer,Iran,online,CUST_001
```

**Required Columns:**
- `date`: Transaction timestamp (YYYY-MM-DD HH:MM:SS)
- `amount`: Transaction amount (numeric)
- `merchant`: Merchant name
- `country`: Country name or code
- `channel`: Transaction channel (online, pos, atm, mobile)

**Optional Columns:**
- `customer_id`: Customer identifier (auto-generated if missing)
- `transaction_id`: Transaction ID (auto-generated if missing)

## 🔬 Command-Line Usage

For programmatic use or batch processing:

```python
from aml_detector import AMLDetector

# Initialize detector
detector = AMLDetector()

# Load and train
training_data = detector.load_data('transactions_training.csv')
detector.train()

# Analyze transactions
test_data = detector.load_data('transactions_test.csv')
results = detector.analyze_batch(test_data)

# Get flagged transactions
flagged = detector.get_flagged_transactions(results, min_severity='medium')

# Generate report
report = detector.generate_report(results)
print(report)

# Export results
detector.export_results(results, 'aml_results.csv')
```

### Analyze Single Transaction

```python
transaction = {
    'transaction_id': 'TXN_001',
    'customer_id': 'CUST_001',
    'date': '2024-01-15 14:30:00',
    'amount': 15000.00,
    'merchant': 'Wire Transfer',
    'country': 'Iran',
    'channel': 'online'
}

explanation = detector.analyze_transaction(transaction)

print(f"Risk Score: {explanation['risk_score']:.2f}")
print(f"Severity: {explanation['severity']}")
print(f"Flagged: {explanation['is_flagged']}")

for reason in explanation['primary_reasons']:
    print(f"- {reason['explanation']}")
```

## 📈 Risk Scoring

Each transaction receives a **risk score (0-1)** calculated as:

- **30%** Baseline deviations (how much it differs from customer's normal behavior)
- **40%** Rule violations (number and severity of rule triggers)
- **30%** ML anomaly scores (ensemble of 3 models)

### Severity Levels

| Severity | Score Range | Action |
|----------|-------------|--------|
| 🟢 Low | 0.0 - 0.3 | Log for record-keeping |
| 🟡 Medium | 0.3 - 0.5 | Review within 24 hours |
| 🟠 High | 0.5 - 0.7 | Urgent review required |
| 🔴 Critical | 0.7 - 1.0 | Immediate action, block transaction |

## 🏗️ Architecture

```
AML Transaction Anomaly Detector
│
├── config.py                    # Configuration and thresholds
├── baseline_engine.py           # Customer baseline computation
├── rule_engine.py              # Rule-based detection
├── ml_engine.py                # ML anomaly detection
├── explainability_engine.py    # Explanation generation
├── aml_detector.py             # Main orchestration
├── app.py                      # Streamlit web interface
└── generate_sample_data.py     # Sample data generator
```

### Component Details

**BaselineEngine**: Computes customer-specific patterns
- Average/median/std of transaction amounts
- Common countries, merchants, channels
- Typical transaction times
- Transaction frequency

**RuleEngine**: Implements AML business rules
- High-risk country detection
- Structuring detection (Smurfing)
- Rapid movement detection
- Large transaction alerts
- Round amount detection
- Unusual channel detection

**MLAnomalyDetector**: Ensemble ML detection
- Isolation Forest (tree-based isolation)
- One-Class SVM (boundary learning)
- Local Outlier Factor (density-based)
- Feature engineering (temporal, categorical, amount)
- Ensemble voting and scoring

**ExplainabilityEngine**: Human-readable explanations
- Risk score calculation
- Reason generation and ranking
- Severity classification
- Action recommendations

## 🎨 Web Interface Features

### Dashboard
- Key metrics (total, flagged, critical alerts)
- Severity distribution pie chart
- Risk score histogram
- Top 10 highest risk transactions

### Analysis
- Filterable transaction list
- Detailed explanations for each alert
- Risk breakdown by category
- ML model scores and rule violations

### Visualizations
- Time series of transactions and alerts
- Geographic distribution
- Channel analysis
- Amount vs risk scatter plots
- Box plots by severity

## ⚙️ Configuration

Edit `config.py` to customize:

```python
# High-risk countries
HIGH_RISK_COUNTRIES = ['North Korea', 'Iran', 'Syria', ...]

# Structuring detection
STRUCTURING_THRESHOLD = 10000      # Reporting threshold
STRUCTURING_WINDOW_DAYS = 7        # Time window
STRUCTURING_COUNT = 3              # Min transactions to flag

# Rapid movement
RAPID_MOVEMENT_HOURS = 24          # Time window
RAPID_MOVEMENT_COUNT = 5           # Min transactions

# ML parameters
CONTAMINATION = 0.1                # Expected anomaly rate
```

## 📊 Example Output

```
================================================================================
TRANSACTION ALERT - HIGH RISK
================================================================================
Transaction ID: TXN_000523
Customer ID: CUST_002
Risk Score: 0.78 / 1.00
Severity: HIGH

PRIMARY REASONS FOR FLAGGING:
--------------------------------------------------------------------------------
1. [RULE_VIOLATION] Transaction from high-risk country: Iran
   Risk Score: 0.75

2. [BASELINE_DEVIATION] Transaction amount $15,000.00 is 12.5× customer's average ($1,200.00)
   Risk Score: 0.85

3. [ML_DETECTION] Flagged by 3 out of 3 ML models as anomalous behavior
   Risk Score: 0.82

Rule Violations: 2
ML Models Flagged: 3 / 3
ML Confidence: 0.82

RECOMMENDATION:
--------------------------------------------------------------------------------
URGENT REVIEW: Hold transaction for manual review by AML analyst.
Multiple risk indicators detected.
================================================================================
```

## 🧪 Testing

The system includes comprehensive test data with known anomalies:

```bash
# Generate test data
python generate_sample_data.py

# Run analysis
python -c "
from aml_detector import AMLDetector
detector = AMLDetector()
detector.load_data('transactions_training.csv')
detector.train()
test_data = detector.load_data('transactions_test.csv')
results = detector.analyze_batch(test_data)
print(detector.generate_report(results))
"
```

## 🔒 Security & Compliance

**Important Notes:**
- This is a demonstration system for educational purposes
- Production use requires additional features:
  - Audit trails and logging
  - User authentication and authorization
  - Regulatory reporting capabilities
  - Data encryption and privacy controls
  - Integration with case management systems
  - Compliance with local AML regulations (BSA, AMLD, etc.)

## 🛠️ Technology Stack

- **Python 3.8+**
- **pandas**: Data manipulation
- **numpy**: Numerical operations
- **scikit-learn**: Machine learning models
- **streamlit**: Web interface
- **plotly**: Interactive visualizations
- **matplotlib/seaborn**: Static plots

## 📚 How It Works

### Training Phase
1. Load historical transaction data
2. Compute customer baselines (spending patterns, locations, etc.)
3. Train ML models on feature-engineered data
4. Store models and baselines for inference

### Detection Phase
1. Receive new transaction
2. Compare to customer baseline → deviation scores
3. Apply rule-based checks → violations
4. Run through ML models → anomaly scores
5. Combine all signals → risk score
6. Generate human-readable explanation
7. Classify severity and recommend action

### Explainability
- Each alert includes 3-5 primary reasons
- Reasons ranked by contribution to risk score
- Breakdown shows: baseline deviations, rule violations, ML scores
- Recommendations based on severity level

## 🎓 Educational Value

This project demonstrates:
- **Ensemble ML**: Combining multiple algorithms for robust detection
- **Feature Engineering**: Creating meaningful features from raw data
- **Rule-Based Systems**: Implementing domain knowledge as rules
- **Explainable AI**: Making ML decisions interpretable
- **Full-Stack Development**: Backend ML + Frontend visualization
- **Real-World Application**: Solving actual financial crime problems

## 🤝 Contributing

Suggestions for enhancements:
- Additional ML models (Autoencoders, LSTM for sequences)
- Network analysis (transaction graphs)
- Real-time streaming detection
- Integration with external APIs (sanctions lists, PEP databases)
- Advanced NLP for merchant categorization
- Federated learning for privacy-preserving training

## 📝 License

MIT License - Feel free to use for educational and commercial purposes.

## 🙏 Acknowledgments

Inspired by real-world AML systems used by major financial institutions. Built to demonstrate the intersection of machine learning, rule-based systems, and explainable AI in financial crime detection.

---

**Built with ❤️ for the fight against financial crime**

For questions or support, please open an issue on the repository.

