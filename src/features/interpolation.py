# features/interpolation.py

from .base import Feature
import numpy as np
import pandas as pd

class InterpolationFeature(Feature):
    """Feature : interpolation linéaire normalisée"""
    
    def extract(self, X: pd.DataFrame, scaler=None) -> np.ndarray:
        # Interpolation linéaire
        X_interp = X.interpolate(method='linear', limit_direction='both')
        X_interp = X_interp.bfill().ffill()
        
        values = X_interp.values
        
        # Normaliser
        if scaler is not None:
            values = scaler.transform(values.reshape(-1, 1)).flatten()
        
        return values.reshape(-1, 1)
    
    def get_dim(self) -> int:
        return 1
    
    def get_name(self) -> str:
        return "interpolation"