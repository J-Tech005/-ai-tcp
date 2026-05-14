"""
Synthetic Test Execution Data Generator
Generates realistic test execution logs for AI-TCP system
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# Set random seed for reproducibility
np.random.seed(42)

def generate_synthetic_data(n_tests=750, n_cycles=25, output_path="data/raw"):
    """
    Generate synthetic test execution history data.
    
    Args:
        n_tests (int): Number of unique test cases
        n_cycles (int): Number of regression cycles
        output_path (str): Path to save the CSV
    
    Returns:
        pd.DataFrame: Generated dataset
    """
    
    # Create output directory if it doesn't exist
    Path(output_path).mkdir(parents=True, exist_ok=True)
    
    print(f"[INFO] Generating synthetic data: {n_tests} tests × {n_cycles} cycles = {n_tests * n_cycles} records")
    
    # Test suites
    suites = ["auth", "payment", "search", "checkout", "api"]
    
    # Pre-generate test_id and test_name mapping
    test_ids = [f"TC_{i:04d}" for i in range(1, n_tests + 1)]
    test_names = [f"test_{suite}_{i:03d}" for i, suite in enumerate([suites[i % 5] for i in range(n_tests)])]
    test_suite_map = [suites[i % 5] for i in range(n_tests)]
    
    # Initialize storage
    data = []
    
    # Track last_result for each test (persistence across cycles)
    last_results = {test_id: 0 for test_id in test_ids}
    days_since_failure = {test_id: 60 for test_id in test_ids}
    
    # Generate data per cycle
    for cycle in range(1, n_cycles + 1):
        for test_idx, test_id in enumerate(test_ids):
            test_name = test_names[test_idx]
            suite = test_suite_map[test_idx]
            
            # Change frequency: typically 0-10, with some correlation to test coverage
            change_freq = np.random.poisson(lam=3)
            change_freq = min(change_freq, 10)
            
            # Base failure probability
            base_failure_prob = 0.15  # 15% baseline
            
            # Adjust by suite (payment is riskier)
            if suite == "payment":
                base_failure_prob += 0.15  # 30% for payment suite
            
            # Adjust by change frequency (more changes = more risk)
            if change_freq > 5:
                base_failure_prob += 0.25  # 40%+ for high-change tests
            
            # Adjust by last result (flaky tests tend to fail again)
            if last_results[test_id] == 1:
                base_failure_prob += 0.35  # ~50% if previously failed
            
            # Clamp to [0, 1]
            failure_prob = min(base_failure_prob, 0.95)
            
            # Add ~3% random noise
            failure_prob += np.random.normal(0, 0.03)
            failure_prob = np.clip(failure_prob, 0.01, 0.99)
            
            # Determine outcome (0 = pass, 1 = fail)
            outcome = 1 if np.random.random() < failure_prob else 0
            
            # Execution time: log-normal distribution (0.5 - 30 sec)
            execution_time = np.random.lognormal(mean=1.5, sigma=0.8)
            execution_time = np.clip(execution_time, 0.5, 30.0)
            
            # Update tracking
            days_since_failure[test_id] = 0 if outcome == 1 else days_since_failure[test_id] + 1
            last_results[test_id] = outcome
            
            # Build record
            record = {
                "test_id": test_id,
                "test_name": test_name,
                "test_suite": suite,
                "cycle_id": cycle,
                "outcome": outcome,
                "execution_time": round(execution_time, 2),
                "change_frequency": change_freq,
                "last_result": last_results[test_id],
                "days_since_last_failure": days_since_failure[test_id]
            }
            data.append(record)
    
    df = pd.DataFrame(data)
    
    # Save to CSV
    output_file = Path(output_path) / "test_execution_history.csv"
    df.to_csv(output_file, index=False)
    
    print(f"[SUCCESS] Data saved to: {output_file}")
    print(f"\n=== Dataset Summary ===")
    print(f"Total records: {len(df)}")
    print(f"Unique tests: {df['test_id'].nunique()}")
    print(f"Cycles: {df['cycle_id'].nunique()}")
    print(f"Failure rate: {df['outcome'].mean():.2%}")
    print(f"\nFailure rate by suite:")
    print(df.groupby("test_suite")["outcome"].agg(["count", "sum", lambda x: f"{x.mean():.2%}"]).rename(columns={"<lambda_0>": "fail_rate"}))
    print(f"\nFailure rate by change frequency:")
    change_freq_stats = df.groupby("change_frequency")["outcome"].agg(["count", "sum", lambda x: f"{x.mean():.2%}"]).rename(columns={"<lambda_0>": "fail_rate"})
    print(change_freq_stats)
    print(f"\nExecution time stats (seconds):")
    print(df["execution_time"].describe())
    print(f"\n=== First 10 records ===")
    print(df.head(10).to_string())
    
    return df

if __name__ == "__main__":
    generate_synthetic_data()