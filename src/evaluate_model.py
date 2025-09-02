import os
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def load_feature_list(features_path):
    with open(features_path, 'r') as f:
        features = [line.strip() for line in f]
    return features


def evaluate_and_save_metrics(y_true, y_pred, csv_path):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    metrics = {
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2
    }

    df_new = pd.DataFrame([metrics])

    if os.path.exists(csv_path):
        df_existing = pd.read_csv(csv_path)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        df_combined.to_csv(csv_path, index=False)
    else:
        df_new.to_csv(csv_path, index=False)

    return metrics


def main():
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'xgb_random_search_model.pkl')
    FEATURES_PATH = os.path.join(PROJECT_ROOT, 'models', 'feature_list.txt')
    TRAIN_DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed', 'train_store_merged.csv')
    METRICS_CSV_PATH = os.path.join(PROJECT_ROOT, 'models', 'model_performance.csv')

    # Load feature list
    model_features = load_feature_list(FEATURES_PATH)

    # Load full train data
    train_df = pd.read_csv(TRAIN_DATA_PATH)

    # Prepare features and target
    X = train_df[model_features]
    y = train_df['Sales'].values

    # Split into train & validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Load pretrained model
    model = joblib.load(MODEL_PATH)

    # Predict on validation set
    y_pred_log = model.predict(X_val)
    y_pred = np.expm1(y_pred_log)  # inverse log1p transform

    # Calculate and save metrics on validation set
    metrics = evaluate_and_save_metrics(y_val, y_pred, METRICS_CSV_PATH)

    print("Model evaluation metrics on validation data saved to CSV:")
    print(metrics)


if __name__ == "__main__":
    main()
