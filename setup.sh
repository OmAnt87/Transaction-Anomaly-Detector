#!/bin/bash

# AML Transaction Anomaly Detector - Setup Script

echo "=================================================="
echo "AML Transaction Anomaly Detector - Setup"
echo "=================================================="
echo ""

# Check Python version
echo "Checking Python installation..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version)
    echo "✓ Found: $PYTHON_VERSION"
else
    echo "✗ Python 3 not found. Please install Python 3.8 or higher."
    exit 1
fi

# Install dependencies
echo ""
echo "Installing dependencies..."
pip3 install -r requirements.txt

if [ $? -eq 0 ]; then
    echo "✓ Dependencies installed successfully"
else
    echo "✗ Failed to install dependencies"
    exit 1
fi

# Generate sample data
echo ""
echo "Generating sample transaction data..."
python3 generate_sample_data.py

if [ $? -eq 0 ]; then
    echo "✓ Sample data generated"
else
    echo "✗ Failed to generate sample data"
    exit 1
fi

# Run CLI demo
echo ""
echo "Running CLI demo..."
python3 cli_demo.py

echo ""
echo "=================================================="
echo "Setup Complete!"
echo "=================================================="
echo ""
echo "To launch the web interface, run:"
echo "  streamlit run app.py"
echo ""

