import pandas as pd
import os
from sklearn.model_selection import train_test_split

PROCESSED_DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')

def load_processed_data(return_validation=False, val_size=0.2, random_state=42):
    """
    Load processed train, test (and optionally split validation from train).
    
    Args:
        return_validation (bool): If True, split validation set from train and return it.
        val_size (float): Fraction of train data to use as validation.
        random_state (int): Random seed for reproducibility.
    
    Returns:
        If return_validation=False: train_df, test_df, store_df
        If return_validation=True: train_df, val_df, test_df
    """
    train_path = os.path.join(PROCESSED_DATA_PATH, 'train_store_merged.csv')
    test_path = os.path.join(PROCESSED_DATA_PATH, 'test_store_merged.csv')
    store_path = os.path.join(PROCESSED_DATA_PATH, 'store_feature_engineered.csv')

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    store_df = pd.read_csv(store_path)

    if return_validation:
        train_df, val_df = train_test_split(train_df, test_size=val_size, random_state=random_state)
        return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df
    else:
        return train_df, test_df, store_df

def prepare_training_data(train_df):
    target_cols = ['Sales', 'Customers']
    y_train = train_df['Sales']
    X_train = train_df.drop(columns=target_cols, errors='ignore')
    return X_train, y_train

def prepare_validation_data(val_df):
    target_cols = ['Sales', 'Customers']
    y_val = val_df['Sales']
    X_val = val_df.drop(columns=target_cols, errors='ignore')
    return X_val, y_val

def prepare_test_data(test_df):
    target_cols = ['Sales', 'Customers']
    X_test = test_df.drop(columns=target_cols, errors='ignore')
    return X_test
