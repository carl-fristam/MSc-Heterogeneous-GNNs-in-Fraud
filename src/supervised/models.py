from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


def get_models(pos_weight=None, auto_balance=True):
    """
    Return dict of models to train.

    :param pos_weight: Manual weight for positive class. If None and auto_balance=True, uses 'balanced'.
    :param auto_balance: Use sklearn's automatic balancing when pos_weight is None.
    """

    if pos_weight is not None:
        # Manual weights
        class_weight = {0: 1, 1: pos_weight}
        xgb_weight = pos_weight
    elif auto_balance:
        # Auto balanced
        class_weight = "balanced"
        xgb_weight = None  # Will be computed in train.py from data
    else:
        # No balancing
        class_weight = None
        xgb_weight = 1.0

    return {
        "LogisticRegression": LogisticRegression(
            class_weight=class_weight, max_iter=5000, n_jobs=-1
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=100, class_weight=class_weight,
            n_jobs=-1, random_state=42
        ),
        "XGBoost": XGBClassifier(
            scale_pos_weight=xgb_weight if xgb_weight else 1.0,
            n_estimators=100, learning_rate=0.1,
            max_depth=6, n_jobs=-1, random_state=42
        ),
    }
