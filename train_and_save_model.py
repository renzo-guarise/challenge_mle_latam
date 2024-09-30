import pandas as pd
from typing import Tuple
from sklearn.model_selection import train_test_split
from prefect import flow, task, get_run_logger
from challenge.model import DelayModel

@task
def load_data(file_path: str) -> pd.DataFrame:
    logger = get_run_logger()
    logger.info(f"Loading data from {file_path}")
    data = pd.read_csv(file_path)
    return data

@task
def preprocess_data(data: pd.DataFrame, model: DelayModel) -> Tuple[pd.DataFrame, pd.Series]:
    logger = get_run_logger()
    logger.info("Preprocessing data")
    features, target = model.preprocess(data, "delay")
    return features, target

@task
def train_model(features: pd.DataFrame, target: pd.Series, model: DelayModel) -> DelayModel:
    logger = get_run_logger()
    logger.info("Training model")
    model.fit(features, target)
    return model

@task
def evaluate_model(features: pd.DataFrame, target: pd.Series, model: DelayModel) -> None:
    logger = get_run_logger()
    logger.info("Evaluating model")
    model.evaluate(features, target)

@task
def save_model(model: DelayModel, path: str) -> None:
    logger = get_run_logger()
    logger.info("Saving model")
    model.save(path)

@flow
def model_pipeline(file_path: str, model_path: str):
    logger = get_run_logger()
    logger.info("Starting model pipeline")

    # Initialize model
    model = DelayModel()
    
    # Load data
    data = load_data(file_path)

    # Preprocess data
    features, target = preprocess_data(data, model)

    # Split data
    x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.33, random_state=42)

    # Train model
    trained_model = train_model(x_train, y_train, model)

    # Evaluate model
    evaluate_model(x_test, y_test, trained_model)

    save_model(trained_model, model_path)

if __name__ == "__main__":
    model_pipeline("data/data.csv", "model_v1.bin")