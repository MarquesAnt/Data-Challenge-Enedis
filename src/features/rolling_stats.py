from .base import Feature
import numpy as np
import pandas as pd

class RollingStatsFeature(Feature):
    """
    Features statistiques glissantes :
    - rolling_mean (3h, 6h)
    - rolling_stability (mean/std)
    - slope_6h
    - slope_6h_rel
    """
    
    def __init__(self, rolling_windows=[3, 6], slope_window=6):
        """
        Args:
            rolling_windows: fenêtres pour rolling_mean et stability (en heures)
            slope_window: fenêtre pour le slope (en heures)
        """
        self.rolling_windows = rolling_windows
        self.slope_window = slope_window
    
    def extract(self, X: pd.DataFrame, scaler=None) -> np.ndarray:
        features = []
        series = X.iloc[:, 0]
        n = len(series)
        
        # === 1. Rolling mean et stability (VECTORISÉ) ===
        rolling_means = {}
        
        for win in self.rolling_windows:
            # Rolling mean avec pandas (optimisé C)
            rolling_mean = series.rolling(
                window=2*win, 
                center=True, 
                min_periods=1
            ).mean()
            rolling_mean = rolling_mean.bfill().ffill().values
            
            # Rolling std
            rolling_std = series.rolling(
                window=2*win, 
                center=True, 
                min_periods=1
            ).std()
            rolling_std = rolling_std.bfill().ffill().values
            
            # Stability = mean / std (vectorisé avec protection)
            rolling_stability = np.divide(
                rolling_mean,
                rolling_std,
                out=np.zeros_like(rolling_mean),
                where=(rolling_std != 0) & (~np.isnan(rolling_std))
            )
            
            rolling_means[win] = rolling_mean
            
            features.append(rolling_mean.reshape(-1, 1))
            features.append(rolling_stability.reshape(-1, 1))
        
        # === 2. Slope 6h (VECTORISÉ) ===
        diffs = series.diff()
        slope = diffs.rolling(
            window=self.slope_window, 
            min_periods=2
        ).mean()
        slope = slope.fillna(0).values
        
        features.append(slope.reshape(-1, 1))
        
        # === 3. Slope 6h relative ===
        rolling_mean_6h = rolling_means.get(6, rolling_means[self.rolling_windows[-1]])
        slope_rel = slope / (np.abs(rolling_mean_6h) + 1e-6)  # déjà protégé avec 1e-6
        
        features.append(slope_rel.reshape(-1, 1))
        
        return np.concatenate(features, axis=1)
    
    def get_dim(self) -> int:
        # rolling_mean + rolling_stability pour chaque fenêtre + slope + slope_rel
        return len(self.rolling_windows) * 2 + 2
    
    def get_name(self) -> str:
        return f"rolling_stats_w{'_'.join(map(str, self.rolling_windows))}"