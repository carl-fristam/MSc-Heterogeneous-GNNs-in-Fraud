"""
Data loading utilities for HI-Small AML dataset.
"""

import pandas as pd
from pathlib import Path
from typing import Tuple, Dict


def load_transactions(data_dir: str = 'data') -> pd.DataFrame:
    """
    Load transaction CSV file.
    
    Args:
        data_dir: Directory containing data files
        
    Returns:
        DataFrame with transaction data
    """
    trans_path = Path(data_dir) / 'HI-Small_Trans.csv'
    trans_df = pd.read_csv(trans_path)
    print(f"Loaded {len(trans_df):,} transactions")
    return trans_df


def load_accounts(data_dir: str = 'data') -> pd.DataFrame:
    """
    Load accounts CSV file.
    
    Args:
        data_dir: Directory containing data files
        
    Returns:
        DataFrame with account data
    """
    accounts_path = Path(data_dir) / 'HI-Small_accounts.csv'
    accounts_df = pd.read_csv(accounts_path)
    print(f"Loaded {len(accounts_df):,} accounts")
    return accounts_df


def load_data(data_dir: str = 'data') -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load both transaction and account CSV files.
    
    Args:
        data_dir: Directory containing data files
        
    Returns:
        Tuple of (transactions_df, accounts_df)
    """
    print("Loading data...")
    trans_df = load_transactions(data_dir)
    accounts_df = load_accounts(data_dir)
    return trans_df, accounts_df


def create_account_mapping(trans_df: pd.DataFrame, accounts_df: pd.DataFrame) -> Tuple[Dict[str, int], pd.DataFrame]:
    """
    Create mapping from account identifiers to unique integer IDs.
    
    Args:
        trans_df: Transaction DataFrame
        accounts_df: Accounts DataFrame
        
    Returns:
        Tuple of (account_to_id mapping, all_accounts DataFrame)
    """
    print("\nCreating account mappings...")
    
    # Get all unique accounts from transactions
    from_accounts = trans_df[['From Bank', 'Account']].copy()
    from_accounts.columns = ['Bank', 'Account']
    
    to_accounts = trans_df[['To Bank', 'Account.1']].copy()
    to_accounts.columns = ['Bank', 'Account']
    
    all_accounts = pd.concat([from_accounts, to_accounts]).drop_duplicates()
    
    # Create unique account identifier (Bank + Account)
    # Convert to string to handle mixed types
    all_accounts['account_id'] = all_accounts['Bank'].astype(str) + '_' + all_accounts['Account'].astype(str)
    
    # Map to integer IDs
    account_to_id = {acc: idx for idx, acc in enumerate(all_accounts['account_id'].unique())}
    
    print(f"Found {len(account_to_id):,} unique accounts")
    
    return account_to_id, all_accounts
