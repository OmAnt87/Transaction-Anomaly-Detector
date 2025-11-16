# Installation Guide

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

## Quick Installation

### Option 1: Automatic Setup (Recommended)

```bash
cd "AML Project 1"
chmod +x setup.sh
./setup.sh
```

This will:
1. Install all dependencies
2. Generate sample data
3. Run a demo analysis

### Option 2: Manual Setup

#### Step 1: Install Dependencies

```bash
cd "AML Project 1"
pip3 install -r requirements.txt
```

Or install packages individually:

```bash
pip3 install pandas==2.1.4
pip3 install numpy==1.26.2
pip3 install scikit-learn==1.3.2
pip3 install streamlit==1.29.0
pip3 install plotly==5.18.0
pip3 install matplotlib==3.8.2
pip3 install seaborn==0.13.0
```

#### Step 2: Generate Sample Data

```bash
python3 generate_sample_data.py
```

This creates:
- `transactions_training.csv` (800 transactions)
- `transactions_test.csv` (200 transactions)

#### Step 3: Run Demo

```bash
python3 cli_demo.py
```

#### Step 4: Launch Web Interface

```bash
streamlit run app.py
```

The web interface will open at `http://localhost:8501`

## Troubleshooting

### "Module not found" errors

Make sure all dependencies are installed:

```bash
pip3 install -r requirements.txt --upgrade
```

### Permission errors on macOS

If you encounter permission errors, try:

```bash
pip3 install --user -r requirements.txt
```

### Python version issues

Check your Python version:

```bash
python3 --version
```

Should be 3.8 or higher. If not, install a newer version from [python.org](https://www.python.org/downloads/)

### Virtual Environment (Recommended)

For a clean installation, use a virtual environment:

```bash
# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate  # On macOS/Linux
# or
venv\Scripts\activate  # On Windows

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

## Verification

To verify installation, run:

```bash
python3 -c "import pandas, numpy, sklearn, streamlit, plotly; print('✓ All dependencies installed')"
```

## Next Steps

After installation:

1. **Explore the Web Interface**: `streamlit run app.py`
2. **Run CLI Demo**: `python3 cli_demo.py`
3. **Read Documentation**: See `README.md`
4. **Customize Configuration**: Edit `config.py`

## System Requirements

- **RAM**: 2GB minimum, 4GB recommended
- **Storage**: 100MB for application + data
- **OS**: macOS, Linux, or Windows
- **Browser**: Modern browser for web interface (Chrome, Firefox, Safari, Edge)

## Support

If you encounter issues:

1. Check that Python 3.8+ is installed
2. Verify all dependencies are installed
3. Ensure you're in the correct directory
4. Try using a virtual environment

For additional help, refer to the README.md file.

