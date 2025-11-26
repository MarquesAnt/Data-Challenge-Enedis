# features/temporal.py

from .base import Feature
import numpy as np
import pandas as pd

class TemporalFeatures(Feature):
    """Features temporelles : heure, jour, etc."""
    
    def __init__(self, include_hour=True, include_day=True, 
                 include_weekend=True, cyclic=True):
        self.include_hour = include_hour
        self.include_day = include_day
        self.include_weekend = include_weekend
        self.cyclic = cyclic  # sin/cos encoding
    
    def extract(self, X: pd.DataFrame, scaler=None) -> np.ndarray:
        features = []
        
        # Heure (0-23)
        if self.include_hour:
            hour = X.index.hour.values.astype(float)  # ← .values ici OK
            if self.cyclic:
                hour_sin = np.sin(2 * np.pi * hour / 24)
                hour_cos = np.cos(2 * np.pi * hour / 24)
                features.extend([hour_sin.reshape(-1, 1), 
                            hour_cos.reshape(-1, 1)])
            else:
                hour_norm = hour / 24.0
                features.append(hour_norm.reshape(-1, 1))
        
        # Jour de la semaine (0-6)
        if self.include_day:
            day = X.index.dayofweek.values.astype(float)  # ← .values ici OK
            if self.cyclic:
                day_sin = np.sin(2 * np.pi * day / 7)
                day_cos = np.cos(2 * np.pi * day / 7)
                features.extend([day_sin.reshape(-1, 1), 
                            day_cos.reshape(-1, 1)])
            else:
                day_norm = day / 7.0
                features.append(day_norm.reshape(-1, 1))
        
        # Week-end (0/1)
        if self.include_weekend:
            is_weekend = (X.index.dayofweek >= 5).astype(float)
            #  AVANT : is_weekend.values.reshape(-1, 1)
            #  APRÈS : is_weekend déjà numpy array
            features.append(is_weekend.reshape(-1, 1))
        
        # Concatener toutes les features
        return np.concatenate(features, axis=1)
    
    def get_dim(self) -> int:
        dim = 0
        if self.include_hour:
            dim += 2 if self.cyclic else 1
        if self.include_day:
            dim += 2 if self.cyclic else 1
        if self.include_weekend:
            dim += 1
        return dim
    
    def get_name(self) -> str:
        parts = []
        if self.include_hour:
            parts.append("hour")
        if self.include_day:
            parts.append("day")
        if self.include_weekend:
            parts.append("weekend")
        return f"temporal_{'_'.join(parts)}"