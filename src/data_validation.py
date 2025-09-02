import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def validate_and_fix_features(train_features: pd.DataFrame, test_features: pd.DataFrame):
    errors = []
    fixes = []

    train_cols = set(train_features.columns)
    test_cols = set(test_features.columns)

    missing_in_train = test_cols - train_cols
    missing_in_test = train_cols - test_cols

    if missing_in_train:
        msg = f"Columns present in test but missing in train: {missing_in_train}"
        errors.append(msg)
        logging.error(msg)
        # Auto-fix: Add missing columns to train with zeros
        for col in missing_in_train:
            train_features[col] = 0
            fixes.append(f"Added missing column '{col}' to train features with zeros")

    if missing_in_test:
        msg = f"Columns present in train but missing in test: {missing_in_test}"
        errors.append(msg)
        logging.error(msg)
        # Auto-fix: Add missing columns to test with zeros
        for col in missing_in_test:
            test_features[col] = 0
            fixes.append(f"Added missing column '{col}' to test features with zeros")

    # Reorder columns to be the same
    common_cols = sorted(list(train_features.columns))
    train_features = train_features[common_cols]
    test_features = test_features[common_cols]

    # Check for null values
    if train_features.isnull().any().any():
        msg = "Null values present in train features"
        errors.append(msg)
        logging.error(msg)
        train_features.fillna(0, inplace=True)
        fixes.append("Filled null values in train features with zeros")

    if test_features.isnull().any().any():
        msg = "Null values present in test features"
        errors.append(msg)
        logging.error(msg)
        test_features.fillna(0, inplace=True)
        fixes.append("Filled null values in test features with zeros")

    # Check data types consistency and attempt conversions
    for col in common_cols:
        train_dtype = train_features[col].dtype
        test_dtype = test_features[col].dtype
        if train_dtype != test_dtype:
            msg = f"Data type mismatch in column '{col}': train is {train_dtype}, test is {test_dtype}"
            errors.append(msg)
            logging.error(msg)
            try:
                test_features[col] = test_features[col].astype(train_dtype)
                fixes.append(f"Casted test feature column '{col}' to dtype {train_dtype}")
            except Exception as e:
                logging.error(f"Failed to cast '{col}' in test features: {e}")

    # Check for infinite values
    if np.isinf(train_features.select_dtypes(include=[np.number])).any().any():
        msg = "Infinite values detected in train features"
        errors.append(msg)
        logging.error(msg)
        train_features.replace([np.inf, -np.inf], 0, inplace=True)
        fixes.append("Replaced infinite values with zeros in train features")

    if np.isinf(test_features.select_dtypes(include=[np.number])).any().any():
        msg = "Infinite values detected in test features"
        errors.append(msg)
        logging.error(msg)
        test_features.replace([np.inf, -np.inf], 0, inplace=True)
        fixes.append("Replaced infinite values with zeros in test features")

    if errors:
        logging.info("Data validation found issues but applied fixes where possible.")
        for fix in fixes:
            logging.info(f"Fix applied: {fix}")
        return False, train_features, test_features, errors, fixes
    else:
        logging.info("Data validation passed. Train and Test features are aligned and clean.")
        return True, train_features, test_features, [], []
