import os
import sys
import logging
import traceback
import papermill as pm
import pandas as pd
import numpy as np

from src.data_ingestion import load_processed_data, prepare_training_data, prepare_validation_data, prepare_test_data
from src.data_validation import validate_and_fix_features
from src.train_model import train_model
from src.model_predictions import load_model, predict_on_processed_test, predict
from src.evaluate_model import evaluate_and_save_metrics

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
NOTEBOOKS_PATH = os.path.join(PROJECT_ROOT, 'notebooks')
LOGS_PATH = os.path.join(PROJECT_ROOT, 'logs')
EXECUTION_OUTPUT_PATH = os.path.join(LOGS_PATH, 'notebook_executions')
MODELS_PATH = os.path.join(PROJECT_ROOT, 'models')

os.makedirs(LOGS_PATH, exist_ok=True)
os.makedirs(EXECUTION_OUTPUT_PATH, exist_ok=True)
os.makedirs(MODELS_PATH, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOGS_PATH, 'pipeline.log')),
        logging.StreamHandler()
    ]
)

def run_notebook_with_papermill(input_path, output_path, parameters=None):
    if not os.path.isfile(input_path):
        logging.error(f"Notebook file not found: {input_path}")
        raise FileNotFoundError(f"Notebook file not found: {input_path}")

    logging.info(f"Running notebook with papermill: {input_path}")
    try:
        pm.execute_notebook(
            input_path,
            output_path,
            parameters=parameters or {},
            kernel_name='python3',
            log_output=True
        )
        logging.info(f"Finished running notebook: {input_path}")
    except Exception as e:
        logging.error(f"Error running notebook {input_path}: {e}")
        raise

def run_full_pipeline():
    try:
        # 1) EDA
        eda_nb = os.path.join(NOTEBOOKS_PATH, '01_eda.ipynb')
        eda_nb_out = os.path.join(EXECUTION_OUTPUT_PATH, '01_eda_executed.ipynb')
        run_notebook_with_papermill(eda_nb, eda_nb_out)

        # 2) Feature Engineering
        fe_nb = os.path.join(NOTEBOOKS_PATH, '02_feature_engineering.ipynb')
        fe_nb_out = os.path.join(EXECUTION_OUTPUT_PATH, '02_feature_engineering_executed.ipynb')
        run_notebook_with_papermill(fe_nb, fe_nb_out)

        logging.info("Notebooks executed successfully, proceeding with data loading and validation")

        # 3) Data Ingestion - load processed train, validation, and test data
        train_df, val_df, test_df = load_processed_data(return_validation=True)

        # Prepare features and targets separately
        train_X, y_train = prepare_training_data(train_df)
        val_X, y_val = prepare_validation_data(val_df)
        test_X = prepare_test_data(test_df)

        # 4) Data Validation - validate feature consistency between train, val, and test
        valid, train_X, val_X, errors, fixes = validate_and_fix_features(train_X, val_X)
        if not valid:
            logging.error("Data validation failed between train and validation sets, aborting pipeline.")
            for err in errors:
                logging.error(err)
            for fix in fixes:
                logging.info(f"Applied fix: {fix}")
            return False

        valid, train_X, test_X, errors, fixes = validate_and_fix_features(train_X, test_X)
        if not valid:
            logging.error("Data validation failed between train and test sets, aborting pipeline.")
            for err in errors:
                logging.error(err)
            for fix in fixes:
                logging.info(f"Applied fix: {fix}")
            return False

        logging.info("Data validation passed. Proceeding with model training.")

        # 5) Model Training
        model = train_model(train_X, y_train)

        logging.info("Model training completed successfully!")

        # Skip test prediction and validation evaluation steps here

        return True

    except Exception:
        logging.error("Pipeline failed due to exception:\n" + traceback.format_exc())
        return False


if __name__ == "__main__":
    run_full_pipeline()
