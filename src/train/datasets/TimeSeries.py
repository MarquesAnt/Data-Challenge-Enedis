# train/datasets/TimeSeries.py

import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from models.feature_extractors import FeatureExtractor
import numpy as np
import pandas as pd
from tqdm import tqdm



class TimeSeriesDataset(Dataset):
    def __init__(self, X, y=None, feature_extractor=None, scaler=None, fit_scaler=False):
        """Dataset avec features précalculées"""
        
        self.X = X.values.T
        self.y = y.values.T if y is not None else None
        
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
        
        #  NOUVEAU : PRÉCALCULER TOUTES LES FEATURES
        print(f"   Précalcul des features pour {self.X.shape[0]} séries...")
        self.features_precalculated = self._precalculate_features(
            X, feature_extractor, self.scaler
        )
        print(f"  ✓ Features précalculées : {self.features_precalculated.shape}")
        
        self.index = X.index
        self.columns = X.columns
    
    def _precalculate_features(self, X, feature_extractor, scaler):
        """Précalculer toutes les features une seule fois"""
        all_features = []
        
        for col in tqdm(X.columns, desc="Extracting features"):
            X_series = pd.DataFrame(X[col], columns=[col])
            features = feature_extractor.extract(X_series, scaler)
            all_features.append(features)
        
        # Shape: (n_series, n_timesteps, n_features)
        return np.array(all_features, dtype=np.float32)
    
    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, idx):
        #  Récupérer les features précalculées (très rapide!)
        x = torch.FloatTensor(self.features_precalculated[idx])
        
        # Mask
        mask = torch.FloatTensor(
            (~np.isnan(self.X[idx])).astype(np.float32)
        ).unsqueeze(-1)
        
        if self.y_scaled is not None:
            y = torch.FloatTensor(self.y_scaled[idx]).unsqueeze(-1)
            return x, mask, y
        
        return x, mask
