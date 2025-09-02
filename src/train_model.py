import os
import joblib
import logging
import numpy as np
import warnings
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODELS_PATH = os.path.join(PROJECT_ROOT, 'models')
os.makedirs(MODELS_PATH, exist_ok=True)


def train_model(train_X, y_train, n_iter=50, cv=3, random_state=42):
    try:
        logging.info("Starting hyperparameter tuning and training with RandomizedSearchCV")

        # Log transform target
        y_log = np.log1p(y_train)

        model = XGBRegressor(
            objective='reg:squarederror',
            tree_method='gpu_hist',
            predictor='gpu_predictor',
            random_state=random_state,
            n_jobs=-1,
        )

        param_dist = {
            'n_estimators': [100, 200, 500, 1000],
            'max_depth': [3, 4, 6, 8, 10],
            'learning_rate': [0.01, 0.03, 0.05, 0.1],
            'subsample': [0.6, 0.7, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.7, 0.8, 1.0],
            'reg_alpha': [0, 0.01, 0.1, 1],
            'reg_lambda': [1, 1.5, 2, 3]
        }

        # Suppress XGBoost GPU warnings
        warnings.filterwarnings(action='ignore', category=UserWarning)

        random_search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_dist,
            n_iter=n_iter,
            scoring='neg_mean_squared_error',
            cv=cv,
            verbose=2,
            random_state=random_state,
            n_jobs=-1,
            refit=True
        )

        random_search.fit(train_X, y_log)

        best_model = random_search.best_estimator_

        logging.info(f"Best params found: {random_search.best_params_}")
        logging.info(f"Best RMSE from CV: {np.sqrt(-random_search.best_score_):.4f}")

        # Save the best model
        model_pickle_path = os.path.join(MODELS_PATH, "xgb_random_search_model.pkl")
        joblib.dump(best_model, model_pickle_path)
        logging.info(f"Best model saved as Pickle to {model_pickle_path}")

        # Save feature list
        feature_list_path = os.path.join(MODELS_PATH, "feature_list.txt")
        with open(feature_list_path, "w") as f:
            for feat in train_X.columns:
                f.write(f"{feat}\n")
        logging.info(f"Feature list saved to {feature_list_path}")

        return best_model

    except Exception as e:
        logging.error(f"Training failed: {e}")
        raise
