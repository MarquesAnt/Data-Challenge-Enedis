# features/rolling_stats.py

from .base import Feature
import numpy as np
import pandas as pd

class RollingStatsFeature(Feature):
    """
    Features statistiques glissantes et pentes
    """
    
    def __init__(self, windows=[6, 12], include_slope=True, include_stability=True):
        """
        Args:
            windows: liste de fenêtres en timesteps (ex: [6, 12] = 3h, 6h avec timestep=30min)
            include_slope: inclure les pentes
            include_stability: inclure stabilité (rolling std)
        """
        self.windows = windows
        self.include_slope = include_slope
        self.include_stability = include_stability
    
    def extract(self, X: pd.DataFrame, scaler=None) -> np.ndarray:
        features = []
        
        for window in self.windows:
            # Rolling mean
            rolling_mean = X.rolling(window=window, center=True, min_periods=1).mean()
            rolling_mean = rolling_mean.fillna(method='bfill').fillna(method='ffill')
            
            # Normaliser si scaler fourni
            if scaler is not None:
                values = rolling_mean.values.reshape(-1, 1)
                rolling_mean_norm = scaler.transform(values).flatten()
            else:
                rolling_mean_norm = rolling_mean.values.flatten()
            
            features.append(rolling_mean_norm.reshape(-1, 1))
            
            # Rolling std (stabilité)
            if self.include_stability:
                rolling_std = X.rolling(window=window, center=True, min_periods=1).std()
                rolling_std = rolling_std.fillna(0)
                
                # Normaliser la std (échelle différente)
                rolling_std_norm = rolling_std.values.flatten()
                if rolling_std_norm.max() > 0:
                    rolling_std_norm = rolling_std_norm / (rolling_std_norm.max() + 1e-8)
                
                features.append(rolling_std_norm.reshape(-1, 1))
            
            # Slope (pente)
            if self.include_slope:
                # Calculer la pente sur la fenêtre
                slope = self._compute_slope(X, window)
                features.append(slope.reshape(-1, 1))
        
        return np.concatenate(features, axis=1)
    
    def _compute_slope(self, X, window):
        """Calculer la pente locale"""
        values = X.values.flatten()
        slopes = np.zeros_like(values)
        
        for i in range(len(values)):
            start = max(0, i - window//2)
            end = min(len(values), i + window//2)
            
            if end - start > 1:
                # Régression linéaire simple
                x = np.arange(end - start)
                y = values[start:end]
                
                # Ignorer les NaN
                mask = ~np.isnan(y)
                if mask.sum() > 1:
                    x_valid = x[mask]
                    y_valid = y[mask]
                    
                    # Pente : coefficient de la régression
                    slope = np.polyfit(x_valid, y_valid, 1)[0]
                    slopes[i] = slope
        
        # Normaliser les pentes
        if slopes.max() > slopes.min():
            slopes = (slopes - slopes.mean()) / (slopes.std() + 1e-8)
        
        return slopes
    
    def get_dim(self) -> int:
        dim = len(self.windows)  # rolling_mean
        if self.include_stability:
            dim += len(self.windows)  # rolling_std
        if self.include_slope:
            dim += len(self.windows)  # slope
        return dim
    
    def get_name(self) -> str:
        parts = ["rolling"]
        if self.include_slope:
            parts.append("slope")
        if self.include_stability:
            parts.append("std")
        return f"{'_'.join(parts)}_w{'_'.join(map(str, self.windows))}"