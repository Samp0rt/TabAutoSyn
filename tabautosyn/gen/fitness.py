import numpy as np
import pandas as pd
import xgboost as xgb
import warnings
from typing import Optional
from .individ import Individual
from abc import ABC, abstractmethod
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import LabelEncoder


class FitnessEvaluator(ABC):
    """Abstract class for fitness evaluation"""

    @abstractmethod
    def evaluate(self, individual: Individual, test_data: pd.DataFrame) -> float:
        """Evaluate individual fitness"""
        pass


class MLFitnessEvaluator(FitnessEvaluator):
    """Fitness evaluation through machine learning"""

    def __init__(self, target_col: str):
        self.target_col = target_col

    _ensure_1d_labels = staticmethod(
        lambda y: np.argmax(y, axis=1) if (y := np.asarray(y)).ndim == 2 and y.shape[1] > 1 else y.ravel())

    @staticmethod
    def _sum_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> Optional[float]:
        y_true, y_score = MLFitnessEvaluator._ensure_1d_labels(y_true), np.asarray(y_score)
        if np.unique(y_true).size < 2:
            return None
        try:
            if y_score.ndim == 1:
                return float(roc_auc_score(y_true, y_score))
            if y_score.shape[1] == 2:
                return float(roc_auc_score(y_true, y_score[:, 1]))
            return float(roc_auc_score(y_true, y_score, multi_class="ovr", average="macro"))
        except Exception:
            return None

    def _fit_and_score(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> float:
        """Train models and compute overall score"""
        if len(np.unique(y_train)) < 2:
            return -1e6

        # Logistic Regression
        try:
            
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=FutureWarning)
                warnings.filterwarnings("ignore", category=UserWarning)
                warnings.filterwarnings("ignore", category=ConvergenceWarning)

                lr = LogisticRegression(solver="lbfgs", max_iter=100, random_state=42)
                lr.fit(X_train, y_train)
                proba_lr = lr.predict_proba(X_test)
        except Exception:
            return -1e6

        clf = xgb.XGBClassifier(
            n_estimators=20,
            max_depth=3,
            learning_rate=0.2,
            use_label_encoder=False,
            eval_metric="logloss",
            verbosity=0,
        )

        clf.fit(X_train, y_train)
        proba_clf = clf.predict_proba(X_test)

        auc_lr = self._sum_roc_auc(y_test, proba_lr)
        auc_clf = self._sum_roc_auc(y_test, proba_clf)

        if auc_lr is None or auc_clf is None:
            return -1e6

        return float(auc_lr + auc_clf)

    def evaluate(self, individual: Individual, test_data: pd.DataFrame) -> float:
        """Evaluate individual fitness"""
        df_ind = individual.to_dataframe()
        feature_cols = individual.feature_cols

        X_train = df_ind[feature_cols].astype(float).to_numpy()
        y_train = df_ind[self.target_col].to_numpy()

        X_test = test_data[feature_cols].astype(float).to_numpy()
        y_test = test_data[self.target_col].to_numpy()

        # Common LabelEncoder for train and test
        le = LabelEncoder()
        le.fit(np.concatenate([y_train, y_test]))
        y_train = le.transform(y_train)
        y_test = le.transform(y_test)

        return self._fit_and_score(X_train, y_train, X_test, y_test)

