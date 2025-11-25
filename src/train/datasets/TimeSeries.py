# train/datasets/TimeSeries.py

import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from models.feature_extractors import FeatureExtractor
import numpy as np
import pandas as pd


class TimeSeriesDataset(Dataset):
    """
    Dataset flexible qui utilise un FeatureExtractor
    """
    
    def __init__(self, X, y=None, feature_extractor=None, 
                 scaler=None, fit_scaler=False):
        """
        Args:
            X: DataFrame (timestamps × courbes)
            y: DataFrame avec vraies valeurs (optionnel)
            feature_extractor: FeatureExtractor pour composer les features
            scaler: StandardScaler
            fit_scaler: si True, fit le scaler
        """
        self.X = X.values.T  # (n_series, n_timesteps)
        self.y = y.values.T if y is not None else None
        self.feature_extractor = feature_extractor
        
        # Scaler
        if fit_scaler:
            self.scaler = StandardScaler()
            valid_values = self.X[~np.isnan(self.X)].reshape(-1, 1)
            self.scaler.fit(valid_values)
        else:
            self.scaler = scaler
        
        # Normaliser Y
        if self.y is not None:
            self.y_scaled = self.scaler.transform(
                self.y.reshape(-1, 1)
            ).reshape(self.y.shape)
        else:
            self.y_scaled = None
        
        # Index pour retrouver les timestamps
        self.index = X.index
        self.columns = X.columns
    
    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, idx):
        # Reconstruire un DataFrame pour cette série
        col_name = self.columns[idx]
        X_series = pd.DataFrame(
            self.X[idx], 
            index=self.index, 
            columns=[col_name]
        )
        
        # Extraire les features via le FeatureExtractor
        features = self.feature_extractor.extract(X_series, self.scaler)
        x = torch.FloatTensor(features)  # (timesteps, feature_dim)
        
        # Mask (toujours utile pour la loss)
        mask = torch.FloatTensor(
            (~np.isnan(self.X[idx])).astype(np.float32)
        ).unsqueeze(-1)
        
        if self.y_scaled is not None:
            y = torch.FloatTensor(self.y_scaled[idx]).unsqueeze(-1)
            return x, mask, y
        
        return x, mask