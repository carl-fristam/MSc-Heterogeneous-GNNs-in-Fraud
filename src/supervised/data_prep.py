import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek

from src.utils.config import PROJECT_ROOT



def load_data_preprocess(path=str(PROJECT_ROOT / 'datasets' / 'SAML-D.csv')):
    """
    Loads and preprocess the data. Made specifically for SAML-D.csv

    :param path: The path for the dataset
    """

    df = pd.read_csv(path)

    df["Hour"] = pd.to_datetime(df["Time"], format = '%H:%M:%S').dt.hour

    df["DoW"] = pd.to_datetime(df["Date"]).dt.dayofweek

    df["log_amount"] = np.log1p(df["Amount"])


    #Encoding
    cat_cols = ["Payment_currency", "Received_currency",
                "Sender_bank_location", "Receiver_bank_location",
                "Payment_type"]

    df_encoded = pd.get_dummies(df, columns = cat_cols)

    return df_encoded



def split(df, target = "Is_laundering", test_size = 0.2, random_state = 42):
    """
    Splits the data. User specified target variable.

    :param df: Dataframe
    :param target: Label you wanna predict
    :param test_size: Test size (keep default)
    :param random_state: Random state but keep default too
    """

    drop_cols = ["Is_laundering", "Laundering_type", "Time", "Date",
                 "Sender_account", "Receiver_account", "Amount"]

    feature_cols = [c for c in df.columns if c not in drop_cols]

    X = df[feature_cols]

    y = df[target]

    return train_test_split(
        X,
        y,
        test_size = test_size,
        stratify = y,
        random_state = random_state)



def scale_features(X_train, X_test):
    """
    Scale just for LR.
    """

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)

    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, scaler


def resample(X_train, y_train, strategy="smote", ratio=0.1, random_state=42):
    """
    Resample training data to handle class imbalance.

    :param X_train: Training features
    :param y_train: Training labels
    :param strategy: 'smote', 'undersample', 'smote_tomek', or None
    :param ratio: Target ratio of minority to majority class
    :param random_state: Random state
    """
    if strategy is None:
        return X_train, y_train

    if strategy == "smote":
        sampler = SMOTE(sampling_strategy=ratio, random_state=random_state)
    elif strategy == "undersample":
        sampler = RandomUnderSampler(sampling_strategy=ratio, random_state=random_state)
    elif strategy == "smote_tomek":
        sampler = SMOTETomek(sampling_strategy=ratio, random_state=random_state)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)

    print(f"Resampled with {strategy} (ratio={ratio})")
    print(f"  Before: {y_train.value_counts().to_dict()}")
    print(f"  After:  {pd.Series(y_resampled).value_counts().to_dict()}")

    return X_resampled, y_resampled
