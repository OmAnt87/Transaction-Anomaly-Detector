# Quick Start Guide

Get up and running with the AML Transaction Anomaly Detector in 5 minutes!

## 🚀 Fast Track

```bash
# 1. Navigate to project directory
cd "AML Project 1"

# 2. Install dependencies
pip3 install pandas numpy scikit-learn streamlit plotly matplotlib seaborn

# 3. Generate sample data
python3 generate_sample_data.py

# 4. Launch web interface
streamlit run app.py
```

Open your browser to `http://localhost:8501` and you're ready to go!

## 📋 Step-by-Step

### 1. Install Dependencies (2 minutes)

```bash
pip3 install -r requirements.txt
```

### 2. Generate Sample Data (30 seconds)

```bash
python3 generate_sample_data.py
```

This creates realistic transaction data with:
- 800 training transactions (normal behavior)
- 200 test transactions (with anomalies to detect)

### 3. Choose Your Interface

#### Option A: Web Interface (Recommended)

```bash
streamlit run app.py
```

Features:
- 📊 Interactive dashboard
- 🔬 Detailed transaction analysis
- 📈 Beautiful visualizations
- 💡 Full explainability

#### Option B: Command Line

```bash
python3 cli_demo.py
```

Features:
- 🖥️ Terminal-based analysis
- 📄 Text reports
- 🚀 Fast batch processing
- 📁 CSV export

#### Option C: Python API

```python
from aml_detector import AMLDetector

# Initialize and train
detector = AMLDetector()
detector.load_data('transactions_training.csv')
detector.train()

# Analyze transactions
test_data = detector.load_data('transactions_test.csv')
results = detector.analyze_batch(test_data)

# Get flagged transactions
flagged = detector.get_flagged_transactions(results, min_severity='high')
print(f"Found {len(flagged)} high-risk transactions")
```

## 🎯 Using the Web Interface

### Step 1: Load Data
- Click **"Use Sample Data"** in the sidebar
- Or upload your own CSV file

### Step 2: Train System
- Click **"Train System"**
- Wait ~10 seconds for training to complete

### Step 3: Analyze Transactions
- Click **"Analyze Test Data"**
- Or upload new transactions to analyze

### Step 4: Explore Results
- **Dashboard**: Overview metrics and charts
- **Analysis**: Detailed transaction explanations
- **Visualizations**: Interactive plots and graphs

## 📊 Understanding Results

### Risk Scores
- **0.0 - 0.3** (🟢 Low): Minor deviations
- **0.3 - 0.5** (🟡 Medium): Review recommended
- **0.5 - 0.7** (🟠 High): Urgent review
- **0.7 - 1.0** (🔴 Critical): Immediate action

### Why Transactions Are Flagged

Each alert shows:
1. **Primary Reasons**: Top 3 factors (e.g., "Unusual country + Large amount")
2. **Rule Violations**: Which AML rules were triggered
3. **ML Detection**: How many models flagged it (out of 3)
4. **Recommendation**: What action to take

### Example Alert

```
Transaction TXN_000523 - HIGH RISK (0.78)

Primary Reasons:
1. Transaction from high-risk country: Iran
2. Amount $15,000 is 12.5× customer's average
3. Flagged by 3/3 ML models

Recommendation: URGENT REVIEW - Hold for manual review
```

## 🔧 Customization

### Adjust Detection Sensitivity

Edit `config.py`:

```python
# More sensitive (flag more transactions)
CONTAMINATION = 0.05  # Expect 5% anomalies

# Less sensitive (flag fewer transactions)
CONTAMINATION = 0.15  # Expect 15% anomalies
```

### Add High-Risk Countries

```python
HIGH_RISK_COUNTRIES = [
    'North Korea', 'Iran', 'Syria',
    'YourCountry',  # Add your own
]
```

### Change Thresholds

```python
STRUCTURING_THRESHOLD = 10000  # Reporting threshold
RAPID_MOVEMENT_HOURS = 24      # Time window
```

## 📁 Using Your Own Data

### CSV Format Required

```csv
date,amount,merchant,country,channel,customer_id
2024-01-15 14:30:00,125.50,Amazon,USA,online,CUST_001
2024-01-15 18:45:00,45.00,Starbucks,USA,pos,CUST_001
```

**Required columns:**
- `date`: YYYY-MM-DD HH:MM:SS
- `amount`: Numeric
- `merchant`: Text
- `country`: Text
- `channel`: online/pos/atm/mobile

**Optional:**
- `customer_id`: Auto-generated if missing
- `transaction_id`: Auto-generated if missing

### Upload Process

1. Prepare CSV with required columns
2. Upload as training data (historical, mostly normal)
3. Train the system
4. Upload test data (recent transactions to analyze)
5. Review results

## 🎓 Example Workflow

### Scenario: Small Bank AML Compliance

```bash
# 1. Generate or prepare your transaction data
python3 generate_sample_data.py

# 2. Launch web interface
streamlit run app.py

# 3. In the web interface:
#    - Load training data (last 6 months of transactions)
#    - Train system (builds customer profiles)
#    - Analyze today's transactions
#    - Review flagged transactions
#    - Export results for compliance team

# 4. Daily monitoring:
#    - Upload new day's transactions
#    - Review alerts
#    - Investigate high-risk cases
#    - Document findings
```

## 🆘 Common Issues

### "Module not found"
```bash
pip3 install -r requirements.txt
```

### "File not found"
```bash
# Make sure you're in the right directory
cd "AML Project 1"

# Generate sample data
python3 generate_sample_data.py
```

### Web interface won't start
```bash
# Check if streamlit is installed
pip3 install streamlit

# Try with full path
python3 -m streamlit run app.py
```

### Slow performance
- Reduce training data size
- Use fewer transactions for analysis
- Adjust `CONTAMINATION` in config.py

## 📚 Learn More

- **Full Documentation**: See `README.md`
- **Installation Help**: See `INSTALLATION.md`
- **Configuration**: Edit `config.py`

## 💡 Tips

1. **Start with sample data** to understand the system
2. **Train on clean data** (mostly normal transactions)
3. **Review high-risk alerts first** (critical/high severity)
4. **Customize rules** for your specific use case
5. **Export results** for compliance documentation

## 🎉 You're Ready!

The system is now detecting:
- ✅ High-risk country transactions
- ✅ Structuring patterns
- ✅ Unusual amounts
- ✅ Rapid movement
- ✅ ML-detected anomalies
- ✅ Baseline deviations

Start exploring and detecting financial crime! 🔍

