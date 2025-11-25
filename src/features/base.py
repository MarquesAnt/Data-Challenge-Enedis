# features/base.py

from abc import ABC, abstractmethod
import numpy as np
import pandas as pd

class Feature(ABC):
    """Interface de base pour toutes les features"""
    
    @abstractmethod
    def extract(self, X: pd.DataFrame, scaler=None) -> np.ndarray:
        """
        Extraire la feature depuis X
        
        Args:
            X: DataFrame (timesteps × 1 colonne)
            scaler: StandardScaler optionnel
        
        Returns:
            array (timesteps, feature_dim)
        """
        pass
    
    @abstractmethod
    def get_dim(self) -> int:
        """Retourner la dimension de la feature"""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Nom de la feature"""
        pass


class BaseValueFeature(Feature):
    """Feature de base : valeur observée normalisée"""
    
    def extract(self, X: pd.DataFrame, scaler=None) -> np.ndarray:
        values = X.values.copy()
        
        # Normaliser si scaler fourni
        if scaler is not None:
            mask = ~np.isnan(values)
            if mask.any():
                values[mask] = scaler.transform(values[mask].reshape(-1, 1)).flatten()
        
        # Remplacer NaN par 0
        values = np.nan_to_num(values, nan=0.0)
        return values.reshape(-1, 1)
    
    def get_dim(self) -> int:
        return 1
    
    def get_name(self) -> str:
        return "value"


class MaskFeature(Feature):
    """Feature mask : 1=observé, 0=manquant"""
    
    def extract(self, X: pd.DataFrame, scaler=None) -> np.ndarray:
        mask = (~X.isna()).astype(np.float32).values
        return mask.reshape(-1, 1)
    
    def get_dim(self) -> int:
        return 1
    
    def get_name(self) -> str:
        return "mask"