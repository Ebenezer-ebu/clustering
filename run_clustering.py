"""
Main entry point for K-Means Clustering Application

This script orchestrates the entire clustering pipeline:
1. Loads data from AWS Open Dataset (or synthetic fallback)
2. Runs K-Means clustering
3. Performs unit tests
4. Generates visualizations
5. Outputs results summary

Usage: python run_clustering.py
"""

import os
import sys
import subprocess
import argparse

# Create necessary directories
os.makedirs('data', exist_ok=True)
os.makedirs('output', exist_ok=True)
os.makedirs('tests', exist_ok=True)


def run_unit_tests():
    """Execute unit tests and return results."""
    print("\n" + "="*60)
    print("RUNNING UNIT TESTS")
    print("="*60)
    
    result = subprocess.run([sys.executable, '-m', 'pytest', 'tests/', '-v', 
                            '--tb=short', '--disable-warnings'], 
                           capture_output=True, text=True)
    
    print(result.stdout)
    if result.stderr:
        print("Errors:", result.stderr)
    
    return result.returncode == 0


def run_clustering():
    """Execute main clustering script."""
    print("\n" + "="*60)
    print("RUNNING K-MEANS CLUSTERING")
    print("="*60)
    
    # Add src to path and import
    sys.path.insert(0, os.path.abspath('src'))
    from kmeans_clustering import main
    
    return main()


def main():
    parser = argparse.ArgumentParser(description='K-Means Clustering Application')
    parser.add_argument('--skip-tests', action='store_true', 
                       help='Skip unit tests and run clustering only')
    parser.add_argument('--tests-only', action='store_true',
                       help='Run only unit tests, no clustering')
    
    args = parser.parse_args()
    
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║     K-MEANS CLUSTERING APPLICATION - AWS Open Dataset       ║
    ║                                                              ║
    ║  This application implements K-Means clustering from        ║
    ║  scratch and compares with scikit-learn implementation.     ║
    ║                                                              ║
    ║  Dataset: NYC Taxi Trip Records (AWS Open Data)             ║
    ║  Features: trip_distance, fare_amount, tip_amount           ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    if args.tests_only:
        tests_passed = run_unit_tests()
        sys.exit(0 if tests_passed else 1)
    
    if not args.skip_tests:
        tests_passed = run_unit_tests()
        if not tests_passed:
            print("\n⚠️  Unit tests failed! Continuing with clustering anyway...\n")
    
    # Run clustering
    result = run_clustering()
    
    print("\n" + "="*60)
    print("APPLICATION COMPLETE")
    print("="*60)
    print("Output files generated:")
    print("  - output/elbow_curve.png (Elbow method plot)")
    print("  - output/cluster_plot.png (Cluster visualization)")
    print("  - output/feature_distributions.png (EDA plot)")
    print("="*60)


if __name__ == "__main__":
    main()