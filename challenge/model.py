import pandas as pd
import pathlib
import pickle
import numpy as np
from typing import Tuple, Union, List
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import  OneHotEncoder


TOP_10_FEATURES = [
    "OPERA_Latin American Wings", 
    "MES_7",
    "MES_10",
    "OPERA_Grupo LATAM",
    "MES_12",
    "TIPOVUELO_I",
    "MES_4",
    "MES_11",
    "OPERA_Sky Airline",
    "OPERA_Copa Air"
]

DELAY_THRESHOLD_IN_MINUTES: int = 15

THRESHOLD_PREDICT: float = 0.5

FEATURE_COLUMNS =['OPERA', 'TIPOVUELO', 'MES']

class DelayModel:

    def __init__(
        self
    ):
        self._model = None # Model should be saved in this attribute.
        self.preprocess_pipeline = OneHotEncoder(sparse=False)

    def _get_min_diff(self, row: pd.Series) -> float:
        """
        Calculate the minute difference between two date columns.

        Args:
            row (pd.Series): Row of the dataset.
        """
        fecha_o = datetime.strptime(row['Fecha-O'], '%Y-%m-%d %H:%M:%S')
        fecha_i = datetime.strptime(row['Fecha-I'], '%Y-%m-%d %H:%M:%S')
        min_diff = ((fecha_o - fecha_i).total_seconds()) / 60
        return min_diff

    def preprocess(
        self,
        data: pd.DataFrame,
        target_column: str = None
    ) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
        """
        Prepare raw data for training or predict.

        Args:
            data (pd.DataFrame): raw data.
            target_column (str, optional): if set, the target is returned.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: features and target.
            or
            pd.DataFrame: features.
        """
        
        
        # Generate 'delay' column based on a threshold of 15 minutes
        
        # Feature encoding
        
        
        if target_column:
            data['min_diff'] = data.apply(self._get_min_diff, axis=1)
            data['delay'] = np.where(data['min_diff'] > DELAY_THRESHOLD_IN_MINUTES, 1, 0)
            features_encoded = self.preprocess_pipeline.fit_transform(data[FEATURE_COLUMNS])
            column_names = self.preprocess_pipeline.get_feature_names_out(input_features=FEATURE_COLUMNS)
            features = pd.DataFrame(features_encoded, columns=column_names)[TOP_10_FEATURES]
            return features, data[[target_column]]
        else:
            features_encoded = self.preprocess_pipeline.transform(data[FEATURE_COLUMNS])
            features = pd.DataFrame(features_encoded, columns=self.preprocess_pipeline.get_feature_names_out())[TOP_10_FEATURES]
            return features

    def fit(
        self,
        features: pd.DataFrame,
        target: pd.DataFrame
    ) -> None:
        """
        Fit model with preprocessed data.

        Args:
            features (pd.DataFrame): preprocessed data.
            target (pd.DataFrame): target.
        """
        # Balance Class Weights
        target = target.to_numpy()
        n_y0 = len(target[target == 0])
        n_y1 = len(target[target == 1])
        
        # Train Logistic Regression with balanced class weights
        self._model = LogisticRegression(class_weight={1: n_y0 / len(target), 0: n_y1 / len(target)})
        self._model.fit(features, target)

    def predict(
        self,
        features: pd.DataFrame
    ) -> List[int]:
        """
        Predict delays for new flights.

        Args:
            features (pd.DataFrame): preprocessed data.
        
        Returns:
            (List[int]): predicted targets.
        """
        return [1 if y_pred > THRESHOLD_PREDICT else 0 for y_pred in self._model.predict(features)]

    def evaluate(self, features: pd.DataFrame, target: pd.Series) -> None:
        """Evaluate the trained model."""
        y_preds = self._model.predict(features)
        print(confusion_matrix(target, y_preds))
        print(classification_report(target, y_preds))
    
    def save(self, path: str) -> None:
        """
        Method to save train model

        Args:
            path (str): path where it is going to be save
        """
        pathlib.Path("models").mkdir(exist_ok=True)
        with open(f"models/{path}", "wb") as f_out:
            pickle.dump(self, f_out)