# features/profile.py

from .base import Feature
import numpy as np
import pandas as pd

class HourlyProfileFeature(Feature):
    """
    Profil moyen par heure de la journée
    """
    
    def __init__(self, fit_data=None):
        """
        Args:
            fit_data: DataFrame pour calculer le profil (courbes complètes)
        """
        self.hourly_profile = None
        if fit_data is not None:
            self._fit_profile(fit_data)
    
    def _fit_profile(self, X):
        """Calculer le profil moyen par heure"""
        # Grouper par heure et calculer la moyenne
        hourly_means = {}
        for hour in range(24):
            mask = X.index.hour == hour
            if mask.any():
                hourly_means[hour] = X[mask].mean().mean()
        
        self.hourly_profile = hourly_means
    
    def extract(self, X: pd.DataFrame, scaler=None) -> np.ndarray:
        if self.hourly_profile is None:
            # Fallback : utiliser la moyenne globale
            global_mean = X.mean().mean() if not X.isna().all().all() else 0
            profile_values = np.full(len(X), global_mean)
        else:
            # Mapper chaque heure à son profil
            hours = X.index.hour.values
            profile_values = np.array([self.hourly_profile.get(h, 0) for h in hours])
        
        # Normaliser si scaler fourni
        if scaler is not None:
            profile_values = scaler.transform(profile_values.reshape(-1, 1)).flatten()
        
        return profile_values.reshape(-1, 1)
    
    def get_dim(self) -> int:
        return 1
    
    def get_name(self) -> str:
        return "hourly_profile"