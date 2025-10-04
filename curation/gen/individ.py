import numpy as np
import pandas as pd
from typing import List, Tuple

class Individual:
    """Класс для представления индивида"""
    def __init__(self, data: List[Tuple], feature_cols: List[str], target_col: str):
        self.data = data
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.fitness_value = None
        
    def to_dataframe(self) -> pd.DataFrame:
        """Преобразование индивида в DataFrame"""
        arr = np.array(self.data)
        cols = self.feature_cols + [self.target_col]
        return pd.DataFrame(arr, columns=cols)
    
    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, feature_cols: List[str], target_col: str):
        """Создание индивида из DataFrame"""
        data = [tuple(row) for row in df[feature_cols + [target_col]].to_numpy()]
        return cls(data, feature_cols, target_col)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    def __setitem__(self, idx, value):
        self.data[idx] = value