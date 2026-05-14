"""
Data Preprocessing Module
Cleaning, encoding, scaling, and splitting functions
"""

import logging
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean data by handling missing values.
    
    Args:
        df (pd.DataFrame): Raw data
    
    Returns:
        pd.DataFrame: Cleaned data
    """
    logger.info("[Preprocessing] Starting data cleaning...")
    
    # Drop rows with >50% missing values
    threshold = len(df.columns) * 0.5
    df = df.dropna(thresh=threshold)
    logger.info(f"[Preprocessing] Dropped rows with >50% missing values")
    
    # Fill remaining missing numerics with median
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().any():
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            logger.info(f"[Preprocessing] Filled {col} with median: {median_val}")
    
    # Fill missing categoricals with mode
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df[col].isnull().any():
            mode_val = df[col].mode()[0]
            df[col].fillna(mode_val, inplace=True)
            logger.info(f"[Preprocessing] Filled {col} with mode: {mode_val}")
    
    logger.info(f"[Preprocessing] Data cleaning complete. Shape: {df.shape}")
    return df

def encode_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode labels and categorical features.
    
    Args:
        df (pd.DataFrame): Data with categorical columns
    
    Returns:
        pd.DataFrame: Data with encoded labels
    """
    logger.info("[Preprocessing] Starting label encoding...")
    
    # Ensure outcome is int (0/1)
    if 'outcome' in df.columns:
        df['outcome'] = df['outcome'].astype(int)
        logger.info(f"[Preprocessing] Outcome column ensured as int")
    
    # One-hot encode test_suite if it exists
    if 'test_suite' in df.columns:
        suite_dummies = pd.get_dummies(df['test_suite'], prefix='suite', drop_first=False)
        df = pd.concat([df, suite_dummies], axis=1)
        logger.info(f"[Preprocessing] One-hot encoded test_suite: {suite_dummies.columns.tolist()}")
    
    logger.info(f"[Preprocessing] Label encoding complete. Shape: {df.shape}")
    return df

def scale_features(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """
    Scale numeric features using StandardScaler.
    
    Args:
        df (pd.DataFrame): Data to scale
        feature_cols (list): Columns to scale (numeric only)
    
    Returns:
        pd.DataFrame: Data with scaled features
    """
    logger.info(f"[Preprocessing] Scaling features: {feature_cols}")
    
    # Filter to only numeric columns that exist
    numeric_feature_cols = [
        col for col in feature_cols 
        if col in df.columns and df[col].dtype in [np.float64, np.int64]
    ]
    
    if numeric_feature_cols:
        scaler = StandardScaler()
        df[numeric_feature_cols] = scaler.fit_transform(df[numeric_feature_cols])
        logger.info(f"[Preprocessing] Scaled {len(numeric_feature_cols)} numeric features")
    
    return df

def split_data(
    df: pd.DataFrame,
    target_col: str = 'outcome',
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42
):
    """
    Split data into train, validation, and test sets with stratification.
    
    Args:
        df (pd.DataFrame): Full dataset
        target_col (str): Target column name
        test_size (float): Proportion for test set
        val_size (float): Proportion for validation set
        random_state (int): Random seed
    
    Returns:
        tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    logger.info(f"[Preprocessing] Splitting data: test_size={test_size}, val_size={val_size}")
    
    # Separate features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # First split: train+val vs test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )
    
    # Second split: train vs val
    val_ratio = val_size / (1 - test_size)  # Adjust ratio for remaining data
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_ratio,
        stratify=y_temp,
        random_state=random_state
    )
    
    logger.info(f"[Preprocessing] Split complete:")
    logger.info(f"  Train: {X_train.shape[0]} ({X_train.shape[0]/len(df):.1%})")
    logger.info(f"  Val:   {X_val.shape[0]} ({X_val.shape[0]/len(df):.1%})")
    logger.info(f"  Test:  {X_test.shape[0]} ({X_test.shape[0]/len(df):.1%})")
    
    return X_train, X_val, X_test, y_train, y_val, y_test
