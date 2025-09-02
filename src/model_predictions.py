import os
import logging
import joblib
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed')
MODELS_PATH = os.path.join(PROJECT_ROOT, 'models')

def load_feature_list():
    feature_list_path = os.path.join(MODELS_PATH, 'feature_list.txt')
    with open(feature_list_path, 'r') as f:
        features = [line.strip() for line in f]
    logging.info(f"Loaded feature list with {len(features)} features")
    return features

def load_model(model_path=None):
    if model_path is None:
        model_path = os.path.join(MODELS_PATH, 'xgb_random_search_model.pkl')
    logging.info(f"Loading model from {model_path}")
    return joblib.load(model_path)

def prepare_test_features(test_df, model_features):
    # Extract IDs if available
    if 'Id' in test_df.columns:
        ids = test_df['Id']
        test_df = test_df.drop(columns=['Id'])
    else:
        ids = pd.Series(range(len(test_df)))

    # Add missing model features filled with zeros
    for feat in model_features:
        if feat not in test_df.columns:
            test_df[feat] = 0

    # Reorder columns to match model training feature order exactly
    test_df = test_df[model_features]

    return test_df, ids

def predict_on_processed_test():
    logging.info("Loading processed test dataset for prediction")
    test_df = pd.read_csv(os.path.join(DATA_PATH, 'test_store_merged.csv'))

    model_features = load_feature_list()
    test_features, ids = prepare_test_features(test_df, model_features)

    model = load_model()
    logging.info("Model loaded successfully")

    logging.info(f"Making predictions on processed test features of shape: {test_features.shape}")
    preds_log = model.predict(test_features)
    preds = np.expm1(preds_log)  # reverse log1p transform

    pred_df = pd.DataFrame({'Id': ids, 'Sales_Prediction': preds})

    # Include actual Sales values if present
    if 'Sales' in test_df.columns:
        pred_df['Sales_Actual'] = test_df['Sales']

    output_path = os.path.join(MODELS_PATH, 'test_predictions.csv')
    pred_df.to_csv(output_path, index=False)
    logging.info(f"Predictions saved to {output_path}")

    return pred_df


def predict(model, X):
    logging.info(f"Making predictions on input data of shape {X.shape}")
    preds_log = model.predict(X)
    clip_upper = 20  # Adjust as needed
    clip_lower = -10  # Adjust as needed
    preds_log = np.clip(preds_log, clip_lower, clip_upper)
    preds = np.expm1(preds_log)  # reverse log1p transform
    return preds

if __name__ == "__main__":
    predict_on_processed_test()
