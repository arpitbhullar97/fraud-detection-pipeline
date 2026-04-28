"""
preprocess.py
-------------
Reusable preprocessing functions for the Financial Fraud Detection Pipeline.
Handles scaling, train/test splitting, and SMOTE oversampling.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE


def load_data(filepath: str) -> pd.DataFrame:
    """
    Load raw credit card transaction data from CSV.

    Args:
        filepath: Path to creditcard.csv

    Returns:
        Raw DataFrame
    """
    df = pd.read_csv(filepath)
    print(f"Data loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")
    print(f"Fraud cases: {df['Class'].sum():,} ({df['Class'].mean()*100:.4f}%)")
    return df


def scale_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize 'Amount' and 'Time' columns.
    V1-V28 are already PCA-scaled so we leave them as-is.

    Args:
        df: Raw DataFrame

    Returns:
        DataFrame with Amount_scaled and Time_scaled replacing originals
    """
    df = df.copy()
    scaler = StandardScaler()
    df['Amount_scaled'] = scaler.fit_transform(df[['Amount']])
    df['Time_scaled']   = scaler.fit_transform(df[['Time']])
    df.drop(columns=['Amount', 'Time'], inplace=True)
    print("Features scaled ✅")
    return df


def split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    Stratified train/test split preserving fraud ratio.

    Args:
        df:           Scaled DataFrame
        test_size:    Proportion for test set (default 0.2)
        random_state: Reproducibility seed

    Returns:
        X_train, X_test, y_train, y_test
    """
    X = df.drop(columns=['Class'])
    y = df['Class']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    print(f"Train: {X_train.shape[0]:,} rows | Fraud: {y_train.sum():,} ({y_train.mean()*100:.4f}%)")
    print(f"Test:  {X_test.shape[0]:,} rows  | Fraud: {y_test.sum():,} ({y_test.mean()*100:.4f}%)")
    return X_train, X_test, y_train, y_test


def apply_smote(X_train, y_train, random_state: int = 42, k_neighbors: int = 5):
    """
    Apply SMOTE to training data only to balance fraud/legitimate classes.
    Never apply to test data — prevents data leakage.

    Args:
        X_train:      Training features
        y_train:      Training labels
        random_state: Reproducibility seed
        k_neighbors:  SMOTE k-nearest neighbors parameter

    Returns:
        X_resampled, y_resampled
    """
    print(f"Before SMOTE → Legitimate: {(y_train==0).sum():,} | Fraud: {(y_train==1).sum():,}")
    smote = SMOTE(random_state=random_state, k_neighbors=k_neighbors)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    print(f"After SMOTE  → Legitimate: {(y_res==0).sum():,} | Fraud: {(y_res==1).sum():,}")
    print("SMOTE applied ✅")
    return X_res, y_res


def save_splits(X_train_res, y_train_res, X_test, y_test, output_dir: str = '../data'):
    """
    Save resampled train and test sets to CSV.

    Args:
        X_train_res:  SMOTE-resampled training features
        y_train_res:  SMOTE-resampled training labels
        X_test:       Test features
        y_test:       Test labels
        output_dir:   Directory to save CSV files
    """
    train_df = pd.DataFrame(X_train_res, columns=X_train_res.columns if hasattr(X_train_res, 'columns') else None)
    train_df['Class'] = y_train_res.values if hasattr(y_train_res, 'values') else y_train_res

    test_df = pd.DataFrame(X_test, columns=X_test.columns if hasattr(X_test, 'columns') else None)
    test_df['Class'] = y_test.values if hasattr(y_test, 'values') else y_test

    train_df.to_csv(f'{output_dir}/train_resampled.csv', index=False)
    test_df.to_csv(f'{output_dir}/test.csv', index=False)

    print(f"Saved train_resampled.csv ({train_df.shape[0]:,} rows) ✅")
    print(f"Saved test.csv ({test_df.shape[0]:,} rows) ✅")


def run_preprocessing(input_path: str, output_dir: str = '../data'):
    """
    Full preprocessing pipeline — load, scale, split, SMOTE, save.

    Args:
        input_path: Path to raw creditcard.csv
        output_dir: Directory to save processed files
    """
    print("=" * 50)
    print("  Running Preprocessing Pipeline")
    print("=" * 50)
    df = load_data(input_path)
    df = scale_features(df)
    X_train, X_test, y_train, y_test = split_data(df)
    X_res, y_res = apply_smote(X_train, y_train)
    save_splits(X_res, y_res, X_test, y_test, output_dir)
    print("\nPreprocessing complete ✅")


if __name__ == '__main__':
    run_preprocessing('../data/creditcard.csv')
