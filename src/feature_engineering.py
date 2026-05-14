"""
Feature Engineering Module
Engineers predictive features from test execution history
"""

import logging
import pandas as pd
import numpy as np
from preprocessing import encode_labels, scale_features

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """
    Engineers features from raw test execution data.
    
    Attributes:
        failure_rate (float): Historical failure rate per test
        recency_weight (float): Exponential decay weight for recent failures
        change_frequency (int): Code change frequency
    """
    
    def __init__(self):
        """Initialize the FeatureEngineer."""
        self.failure_rate = None
        self.recency_weight = None
        self.change_frequency = None
        logger.info("[FeatureEngineer] Initialized")
    
    def compute_failure_rate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute historical failure rate per test_id.
        
        Args:
            df (pd.DataFrame): Raw data with test_id and outcome columns
        
        Returns:
            pd.DataFrame: Data with added 'failure_rate' column
        """
        logger.info("[FeatureEngineer] Computing failure rate...")
        
        # Group by test_id and compute failure rate
        failure_rates = df.groupby('test_id')['outcome'].mean().reset_index()
        failure_rates.columns = ['test_id', 'failure_rate']
        
        # Merge back to original dataframe
        df = df.merge(failure_rates, on='test_id', how='left')
        
        logger.info(f"[FeatureEngineer] Failure rate range: [{df['failure_rate'].min():.3f}, {df['failure_rate'].max():.3f}]")
        return df
    
    def compute_recency_weighted(self, df: pd.DataFrame, decay: float = 0.9) -> pd.DataFrame:
        """
        Compute exponentially weighted moving average of failures (recent failures weighted higher).
        
        Args:
            df (pd.DataFrame): Data with test_id, cycle_id, and outcome columns
            decay (float): Decay factor (0.9 = 10% decay per cycle)
        
        Returns:
            pd.DataFrame: Data with added 'recency_weighted_failures' column
        """
        logger.info(f"[FeatureEngineer] Computing recency-weighted failures (decay={decay})...")
        
        recency_scores = []
        
        for test_id in df['test_id'].unique():
            test_data = df[df['test_id'] == test_id].sort_values('cycle_id')
            max_cycle = test_data['cycle_id'].max()
            
            # Compute exponentially weighted average
            weights = decay ** (max_cycle - test_data['cycle_id'])
            weighted_failures = (test_data['outcome'] * weights).sum() / weights.sum()
            
            # Map back to all rows for this test
            for idx in test_data.index:
                recency_scores.append(weighted_failures)
        
        df['recency_weighted_failures'] = recency_scores
        logger.info(f"[FeatureEngineer] Recency-weighted range: [{df['recency_weighted_failures'].min():.3f}, {df['recency_weighted_failures'].max():.3f}]")
        return df
    
    def build_feature_set(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build complete feature set by applying all feature engineering steps.
        
        Args:
            df (pd.DataFrame): Raw data
        
        Returns:
            pd.DataFrame: Feature matrix with engineered features
        """
        logger.info("[FeatureEngineer] Building complete feature set...")
        
        # Apply feature engineering
        df = self.compute_failure_rate(df)
        df = self.compute_recency_weighted(df, decay=0.9)
        
        # Encode categorical features (one-hot encode test_suite)
        df = encode_labels(df)
        
        # Select feature columns (exclude test metadata and target)
        exclude_cols = ['test_id', 'test_name', 'cycle_id', 'outcome', 'test_suite']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        logger.info(f"[FeatureEngineer] Feature columns: {feature_cols}")
        
        # Scale numeric features
        numeric_feature_cols = [
            col for col in feature_cols
            if df[col].dtype in [np.float64, np.int64]
        ]
        df = scale_features(df, numeric_feature_cols)
        
        # Return feature matrix with outcome
        result = df[feature_cols + ['outcome']].copy()
        logger.info(f"[FeatureEngineer] Feature set complete. Shape: {result.shape}")
        return result
