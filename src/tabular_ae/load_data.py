"""
load_data.py

Tabular data loading for the vanilla autoencoder baseline.

Reads SAML-D.csv directly (no graph construction), engineers transaction-level
features, applies MinMax scaling, and returns stratified splits with the
training set restricted to genuine transactions only.

Feature engineering follows the approach in:
    https://e-jurnal.rokania.ac.id/index.php/jictas/article/view/431
"""

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from typing import Tuple

from src.utils.config import PROJECT_ROOT


def load_tabular_data(
    sample_ratio: float = 0.1,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, MinMaxScaler]:
    """
    Load SAML-D as a flat tabular dataset with engineered features.

    Returns:
        X:             FloatTensor [N, D] — scaled features
        y:             LongTensor  [N]    — Is_laundering labels
        idx_train:     LongTensor — genuine-only indices for training
        idx_val:       LongTensor — mixed indices for validation
        idx_test:      LongTensor — mixed indices for test
        scaler:        fitted MinMaxScaler (for inverse transforms if needed)
    """
    csv_path = PROJECT_ROOT / "datasets" / "SAML-D.csv"
    print(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df):,} transactions")

    if sample_ratio < 1.0:
        df = df.sample(frac=sample_ratio, random_state=seed).reset_index(drop=True)
        print(f"Sampled {len(df):,} transactions")

    # ------------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------------
    features = _engineer_features(df)
    labels = df["Is_laundering"].values.astype(int)

    print(f"Feature matrix: {features.shape}  |  Fraud rate: {labels.mean():.4%}")

    # ------------------------------------------------------------------
    # MinMax scaling (as in the reference paper)
    # ------------------------------------------------------------------
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(features)

    X = torch.tensor(X_scaled, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.long)

    # ------------------------------------------------------------------
    # Stratified split
    # ------------------------------------------------------------------
    idx_train, idx_val, idx_test = _stratified_split(y, val_ratio, test_ratio, seed)

    # Restrict training to genuine only
    genuine_mask = y == 0
    idx_train = idx_train[genuine_mask[idx_train]]

    _print_stats(y, idx_train, idx_val, idx_test)

    return X, y, idx_train, idx_val, idx_test, scaler


def _engineer_features(df: pd.DataFrame) -> np.ndarray:
    """
    Create transaction-level features from raw SAML-D columns.

    Features:
        0: Amount (numeric)
        1: Hour of day (from Time column)
        2: Day of week (from Date column)
        3: Is cross-border (sender_bank_location != receiver_bank_location)
        4: Is same currency (payment_currency == received_currency)
        5-11: Payment_type one-hot encoded
        12+: Sender/Receiver bank location encoded
    """
    feat_cols = []

    # Numeric: Amount
    feat_cols.append(df["Amount"].values.reshape(-1, 1))

    # Temporal: hour of day
    hour = df["Time"].str.split(":").str[0].astype(float).fillna(0)
    feat_cols.append(hour.values.reshape(-1, 1))

    # Temporal: day of week (0=Mon, 6=Sun)
    day_of_week = pd.to_datetime(df["Date"]).dt.dayofweek.astype(float)
    feat_cols.append(day_of_week.values.reshape(-1, 1))

    # Binary: cross-border
    is_cross_border = (df["Sender_bank_location"] != df["Receiver_bank_location"]).astype(float)
    feat_cols.append(is_cross_border.values.reshape(-1, 1))

    # Binary: same currency
    is_same_currency = (df["Payment_currency"] == df["Received_currency"]).astype(float)
    feat_cols.append(is_same_currency.values.reshape(-1, 1))

    # Categorical: Payment_type (one-hot)
    payment_dummies = pd.get_dummies(df["Payment_type"], prefix="ptype").astype(float)
    feat_cols.append(payment_dummies.values)

    # Categorical: Sender + Receiver bank location (label encoded)
    le_sender = LabelEncoder()
    le_receiver = LabelEncoder()
    sender_loc = le_sender.fit_transform(df["Sender_bank_location"].fillna("UNK"))
    receiver_loc = le_receiver.fit_transform(df["Receiver_bank_location"].fillna("UNK"))
    feat_cols.append(sender_loc.reshape(-1, 1).astype(float))
    feat_cols.append(receiver_loc.reshape(-1, 1).astype(float))

    return np.hstack(feat_cols)


def _stratified_split(
    y: torch.Tensor,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split indices preserving fraud ratio in each split."""
    torch.manual_seed(seed)

    fraud_idx = torch.where(y == 1)[0]
    genuine_idx = torch.where(y == 0)[0]

    def split(idx):
        perm = idx[torch.randperm(len(idx))]
        n_val = int(val_ratio * len(idx))
        n_test = int(test_ratio * len(idx))
        return perm[n_val + n_test :], perm[:n_val], perm[n_val : n_val + n_test]

    g_train, g_val, g_test = split(genuine_idx)
    f_train, f_val, f_test = split(fraud_idx)

    return (
        torch.cat([g_train, f_train]),
        torch.cat([g_val, f_val]),
        torch.cat([g_test, f_test]),
    )


def _print_stats(y, idx_train, idx_val, idx_test):
    """Print split statistics."""
    total = len(y)
    total_fraud = int((y == 1).sum().item())
    print(f"\nSplit statistics (total: {total:,} txns, {total_fraud:,} fraud):")
    for name, idx in [
        ("Train (genuine only)", idx_train),
        ("Val", idx_val),
        ("Test", idx_test),
    ]:
        n = len(idx)
        fraud = int(y[idx].sum().item())
        print(f"  {name}: {n:,} txns — {fraud:,} fraud ({100*fraud/max(n,1):.2f}%)")
