"""
ML Engine Module
Abstract and concrete ML model implementations
"""

import logging
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib
from pathlib import Path

logger = logging.getLogger(__name__)

class MLEngine(ABC):
    """
    Abstract base class for ML models.
    
    Attributes:
        model_type (str): Type of model (e.g., 'LogisticRegression')
        hyper_params (dict): Hyperparameters
        trained_model: The fitted model
        model_version (str): Model version identifier
        cross_val_score (float): Cross-validation score
    """
    
    def __init__(self, model_type: str, hyper_params: dict):
        """
        Initialize the MLEngine.
        
        Args:
            model_type (str): Name of the model type
            hyper_params (dict): Hyperparameters for the model
        """
        self.model_type = model_type
        self.hyper_params = hyper_params
        self.trained_model = None
        self.model_version = "1.0"
        self.cross_val_score = None
        logger.info(f"[MLEngine] Initialized {model_type} with params: {hyper_params}")
    
    @abstractmethod
    def train(self, X_train, y_train):
        """Train the model. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def predict(self, X) -> np.ndarray:
        """Make predictions. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def evaluate(self, X_test, y_test) -> dict:
        """Evaluate the model. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def save_model(self, path: str):
        """Save the model to disk. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def load_model(self, path: str):
        """Load the model from disk. Must be implemented by subclasses."""
        pass


class LogisticRegressionModel(MLEngine):
    """
    Logistic Regression model for test failure prediction.
    """
    
    def __init__(self):
        """Initialize Logistic Regression model."""
        hyper_params = {'C': 1.0, 'max_iter': 1000, 'class_weight': 'balanced'}
        super().__init__('LogisticRegression', hyper_params)
        self.trained_model = LogisticRegression(**hyper_params, random_state=42)
    
    def train(self, X_train, y_train):
        """
        Train the Logistic Regression model.
        
        Args:
            X_train: Training features
            y_train: Training target
        """
        logger.info("[LogisticRegression] Training...")
        self.trained_model.fit(X_train, y_train)
        
        # Compute cross-validation score
        cv_scores = cross_val_score(self.trained_model, X_train, y_train, cv=5, scoring='f1')
        self.cross_val_score = cv_scores.mean()
        logger.info(f"[LogisticRegression] Cross-validation F1: {self.cross_val_score:.4f}")
    
    def predict(self, X) -> np.ndarray:
        """
        Predict failure probability (not class labels).
        
        Args:
            X: Features
        
        Returns:
            np.ndarray: Probability of failure [0, 1]
        """
        return self.trained_model.predict_proba(X)[:, 1]
    
    def evaluate(self, X_test, y_test) -> dict:
        """
        Evaluate the model on test set.
        
        Args:
            X_test: Test features
            y_test: Test target
        
        Returns:
            dict: Evaluation metrics
        """
        logger.info("[LogisticRegression] Evaluating...")
        
        # Get predictions
        y_pred = self.trained_model.predict(X_test)
        y_pred_proba = self.predict(X_test)
        
        # Compute metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'auc_roc': roc_auc_score(y_test, y_pred_proba)
        }
        
        logger.info(f"[LogisticRegression] Metrics: {metrics}")
        return metrics
    
    def save_model(self, path: str):
        """
        Save model to disk using joblib.
        
        Args:
            path (str): File path to save model
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.trained_model, path)
        logger.info(f"[LogisticRegression] Model saved to {path}")
    
    def load_model(self, path: str):
        """
        Load model from disk using joblib.
        
        Args:
            path (str): File path to load model from
        """
        self.trained_model = joblib.load(path)
        logger.info(f"[LogisticRegression] Model loaded from {path}")


class RandomForestModel(MLEngine):
    """
    Random Forest model for test failure prediction.
    """
    
    def __init__(self):
        """Initialize Random Forest model."""
        hyper_params = {
            'n_estimators': 100,
            'max_depth': None,
            'class_weight': 'balanced',
            'random_state': 42
        }
        super().__init__('RandomForest', hyper_params)
        self.trained_model = RandomForestClassifier(**hyper_params)
    
    def tune_hyperparams(self, X_train, y_train, cv: int = 5):
        """
        Tune hyperparameters using GridSearchCV.
        
        Args:
            X_train: Training features
            y_train: Training target
            cv (int): Number of cross-validation folds
        """
        logger.info("[RandomForest] Starting hyperparameter tuning...")
        
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 5, 10],
            'min_samples_split': [2, 5]
        }
        
        grid_search = GridSearchCV(
            RandomForestClassifier(class_weight='balanced', random_state=42),
            param_grid,
            cv=cv,
            scoring='f1',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        # Update model and hyperparameters
        self.trained_model = grid_search.best_estimator_
        self.hyper_params = grid_search.best_params_
        self.cross_val_score = grid_search.best_score_
        
        logger.info(f"[RandomForest] Best params: {self.hyper_params}")
        logger.info(f"[RandomForest] Best CV F1: {self.cross_val_score:.4f}")
    
    def train(self, X_train, y_train):
        """
        Train the Random Forest model.
        
        Args:
            X_train: Training features
            y_train: Training target
        """
        logger.info("[RandomForest] Training...")
        self.trained_model.fit(X_train, y_train)
        
        # Compute cross-validation score
        cv_scores = cross_val_score(self.trained_model, X_train, y_train, cv=5, scoring='f1')
        self.cross_val_score = cv_scores.mean()
        logger.info(f"[RandomForest] Cross-validation F1: {self.cross_val_score:.4f}")
    
    def predict(self, X) -> np.ndarray:
        """
        Predict failure probability.
        
        Args:
            X: Features
        
        Returns:
            np.ndarray: Probability of failure [0, 1]
        """
        return self.trained_model.predict_proba(X)[:, 1]
    
    def evaluate(self, X_test, y_test) -> dict:
        """
        Evaluate the model on test set.
        
        Args:
            X_test: Test features
            y_test: Test target
        
        Returns:
            dict: Evaluation metrics
        """
        logger.info("[RandomForest] Evaluating...")
        
        # Get predictions
        y_pred = self.trained_model.predict(X_test)
        y_pred_proba = self.predict(X_test)
        
        # Compute metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'auc_roc': roc_auc_score(y_test, y_pred_proba)
        }
        
        logger.info(f"[RandomForest] Metrics: {metrics}")
        return metrics
    
    def save_model(self, path: str):
        """
        Save model to disk using joblib.
        
        Args:
            path (str): File path to save model
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.trained_model, path)
        logger.info(f"[RandomForest] Model saved to {path}")
    
    def load_model(self, path: str):
        """
        Load model from disk using joblib.
        
        Args:
            path (str): File path to load model from
        """
        self.trained_model = joblib.load(path)
        logger.info(f"[RandomForest] Model loaded from {path}")
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance from the trained Random Forest.
        
        Returns:
            pd.DataFrame: Feature importance dataframe
        """
        if self.trained_model is None:
            logger.warning("[RandomForest] Model not trained yet")
            return None
        
        importance = self.trained_model.feature_importances_
        logger.info(f"[RandomForest] Feature importance shape: {importance.shape}")
        return importance
