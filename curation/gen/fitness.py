import numpy as np
import pandas as pd
import xgboost as xgb
from typing import Optional
from gen.individ import Individual
from abc import ABC, abstractmethod
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
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

    @staticmethod
    def _ensure_1d_labels(y: np.ndarray) -> np.ndarray:
        """Convert labels to 1D format"""
        y = np.asarray(y)
        if y.ndim == 1:
            return y
        if y.ndim == 2:
            if y.shape[1] == 1:
                return y.ravel()
            return np.argmax(y, axis=1)
        return y.ravel()

    @staticmethod
    def _safe_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> Optional[float]:
        """Safe ROC AUC computation"""
        y_true = MLFitnessEvaluator._ensure_1d_labels(y_true)
        y_score = np.asarray(y_score)

        if np.unique(y_true).size < 2:
            return None

        try:
            if y_score.ndim == 2 and y_score.shape[1] == 2:
                return float(roc_auc_score(y_true, y_score[:, 1]))
            if y_score.ndim == 1:
                return float(roc_auc_score(y_true, y_score))
            if y_score.ndim == 2 and y_score.shape[1] > 2:
                return float(
                    roc_auc_score(y_true, y_score, multi_class="ovr", average="macro")
                )
        except Exception:
            return None
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
            import warnings

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=FutureWarning)
                warnings.filterwarnings("ignore", category=UserWarning)
                from sklearn.exceptions import ConvergenceWarning

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

        auc_lr = self._safe_roc_auc(y_test, proba_lr)
        auc_clf = self._safe_roc_auc(y_test, proba_clf)

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
