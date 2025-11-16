"""
Command-Line Demo for AML Transaction Anomaly Detector
"""

import sys
from aml_detector import AMLDetector
from explainability_engine import ExplainabilityEngine


def main():
    """Run a complete demo of the AML detection system"""
    
    print("=" * 80)
    print("AML TRANSACTION ANOMALY DETECTOR - CLI DEMO")
    print("=" * 80)
    print()
    
    # Initialize detector
    print("Initializing AML Detection System...")
    detector = AMLDetector()
    explainer = ExplainabilityEngine()
    
    # Load training data
    print("\n1. Loading training data...")
    try:
        training_data = detector.load_data('transactions_training.csv')
        print(f"   ✓ Loaded {len(training_data)} training transactions")
        print(f"   ✓ Customers: {training_data['customer_id'].nunique()}")
    except FileNotFoundError:
        print("   ✗ Error: transactions_training.csv not found")
        print("   Please run: python generate_sample_data.py")
        sys.exit(1)
    
    # Train system
    print("\n2. Training detection system...")
    detector.train()
    print("   ✓ Training complete!")
    
    # Load test data
    print("\n3. Loading test transactions...")
    try:
        test_data = detector.load_data('transactions_test.csv')
        print(f"   ✓ Loaded {len(test_data)} test transactions")
    except FileNotFoundError:
        print("   ✗ Error: transactions_test.csv not found")
        print("   Please run: python generate_sample_data.py")
        sys.exit(1)
    
    # Analyze transactions
    print("\n4. Analyzing transactions...")
    results = detector.analyze_batch(test_data)
    print(f"   ✓ Analysis complete!")
    
    # Generate report
    print("\n5. Generating summary report...")
    print()
    report = detector.generate_report(results)
    print(report)
    
    # Show detailed examples
    print("\n6. Detailed Alert Examples")
    print("=" * 80)
    
    flagged = detector.get_flagged_transactions(results, min_severity='high')
    
    if len(flagged) > 0:
        print(f"\nShowing top 3 highest risk transactions:\n")
        
        for idx, (_, row) in enumerate(flagged.head(3).iterrows(), 1):
            print(f"\n{'=' * 80}")
            print(f"ALERT #{idx}")
            print('=' * 80)
            
            # Format explanation
            explanation_text = explainer.format_explanation_text({
                'transaction_id': row['transaction_id'],
                'customer_id': row['customer_id'],
                'risk_score': row['risk_score'],
                'severity': row['severity'],
                'primary_reasons': row['primary_reasons'] if isinstance(row['primary_reasons'], list) else [],
                'num_rule_violations': row.get('num_rule_violations', 0),
                'num_ml_models_flagged': row.get('num_ml_models_flagged', 0),
                'ml_confidence': row.get('ml_confidence', 0),
                'recommendation': row.get('recommendation', 'Review recommended')
            })
            
            print(explanation_text)
    else:
        print("\n✓ No high-risk transactions detected!")
    
    # Export results
    print("\n\n7. Exporting results...")
    output_file = 'aml_analysis_results.csv'
    detector.export_results(results, output_file)
    print(f"   ✓ Results exported to {output_file}")
    
    # Summary statistics
    print("\n8. Summary Statistics")
    print("=" * 80)
    
    total = len(results)
    flagged_count = len(results[results['is_flagged'] == True])
    
    print(f"Total Transactions: {total}")
    print(f"Flagged: {flagged_count} ({flagged_count/total*100:.1f}%)")
    print(f"Clean: {total - flagged_count} ({(total-flagged_count)/total*100:.1f}%)")
    print()
    
    severity_counts = results['severity'].value_counts()
    print("Severity Breakdown:")
    for severity in ['critical', 'high', 'medium', 'low']:
        count = severity_counts.get(severity, 0)
        pct = count / total * 100 if total > 0 else 0
        print(f"  {severity.upper():10s}: {count:4d} ({pct:5.1f}%)")
    
    print("\n" + "=" * 80)
    print("DEMO COMPLETE!")
    print("=" * 80)
    print("\nNext steps:")
    print("  • Review aml_analysis_results.csv for full details")
    print("  • Launch web interface: streamlit run app.py")
    print("  • Customize detection rules in config.py")
    print()


if __name__ == '__main__':
    main()

