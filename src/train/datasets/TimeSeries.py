# ============= Dataset =============

import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle


class TimeSeriesDataset(Dataset):
    """Dataset pour les séries temporelles avec valeurs manquantes"""
    
    def __init__(self, X, y=None, scaler=None, fit_scaler=False):
        """
        X: DataFrame (timestamps × courbes)
        y: DataFrame avec les vraies valeurs (même format que X)
        """
        self.X = X.values.T  # Shape: (n_series, n_timesteps)
        self.y = y.values.T if y is not None else None
        
        # Normalisation
        if fit_scaler:
            self.scaler = StandardScaler()
            # Fit sur les valeurs non-NaN uniquement
            valid_values = self.X[~np.isnan(self.X)].reshape(-1, 1)
            self.scaler.fit(valid_values)
        else:
            self.scaler = scaler
            
        # Normaliser X
        self.X_scaled = self.X.copy()
        mask = ~np.isnan(self.X)
        self.X_scaled[mask] = self.scaler.transform(self.X[mask].reshape(-1, 1)).flatten()
        
        # Remplacer les NaN par 0 pour le modèle (on utilisera le mask)
        self.X_scaled = np.nan_to_num(self.X_scaled, nan=0.0)
        
        # Normaliser Y aussi !
        if self.y is not None:
            self.y_scaled = self.scaler.transform(self.y.reshape(-1, 1)).reshape(self.y.shape)
        else:
            self.y_scaled = None
        
        # Créer le masque (1 = valeur observée, 0 = manquante)
        self.mask = (~np.isnan(self.X)).astype(np.float32)

        # Enregistrer le Standard Scaler

        with open("scaler.pkl","wb") as f:
            pickle.dump(scaler,f)
    
    def __len__(self):
        return self.X.shape[0]  # Nombre de séries
    
    def __getitem__(self, idx):
        x = torch.FloatTensor(self.X_scaled[idx]).unsqueeze(-1)  # (timesteps, 1)
        mask = torch.FloatTensor(self.mask[idx]).unsqueeze(-1)   # (timesteps, 1)
        
        if self.y_scaled is not None:
            y = torch.FloatTensor(self.y_scaled[idx]).unsqueeze(-1)
            return x, mask, y
        return x, mask
    
    
# ============= Dataset avec interpolation comme feature =============

    
class TimeSeriesDatasetWithInterp(Dataset):
    """Dataset avec interpolation linéaire comme feature additionnelle"""
        
    def __init__(self, X, y=None, scaler=None, fit_scaler=False):
        
        """
        X: DataFrame (timestamps × courbes) AVEC valeurs manquantes (masquées)
        y: DataFrame avec les vraies valeurs complètes
        """
        self.X = X.values.T  # (n_series, n_timesteps)
        self.y = y.values.T if y is not None else None
            
        print("  Préparation des données...")
            
        # === 1. NORMALISATION ===
        if fit_scaler:
            self.scaler = StandardScaler()
            # Fit sur les valeurs non-NaN uniquement
            valid_values = self.X[~np.isnan(self.X)].reshape(-1, 1)
            self.scaler.fit(valid_values)
            print(f"  ✓ Scaler fitté sur {len(valid_values):,} valeurs observées")
        else:
            self.scaler = scaler
        
        # === 2. CALCULER L'INTERPOLATION LINÉAIRE ===
        print("  Calcul de l'interpolation linéaire...")
        
        # Convertir en DataFrame pour utiliser interpolate()
        X_df = pd.DataFrame(self.X.T)  # (n_timesteps, n_series)
        
        # Interpoler chaque colonne (série)
        X_interp_df = X_df.interpolate(
            method='linear',
            limit_direction='both',  # Interpole dans les deux sens
            axis=0  # Interpoler le long des timesteps
        )
        
        # Si des NaN restent (série entièrement vide), forward/backward fill
        X_interp_df = X_interp_df.fillna(method='ffill').fillna(method='bfill')
        
        self.X_interp = X_interp_df.values.T  # Retour en (n_series, n_timesteps)
        
        print(f"  ✓ Interpolation calculée : shape {self.X_interp.shape}")
        
        # === 3. NORMALISER X (avec NaN) ===
        self.X_scaled = self.X.copy()
        mask = ~np.isnan(self.X)
        self.X_scaled[mask] = self.scaler.transform(
            self.X[mask].reshape(-1, 1)
        ).flatten()
        
        # Remplacer NaN par 0 pour le modèle (le mask indiquera les vraies valeurs)
        self.X_scaled = np.nan_to_num(self.X_scaled, nan=0.0)
        
        # === 4. NORMALISER L'INTERPOLATION ===
        self.X_interp_scaled = self.scaler.transform(
            self.X_interp.reshape(-1, 1)
        ).reshape(self.X_interp.shape)
        
        print("  ✓ Données normalisées")
        
        # === 5. NORMALISER Y (target) ===
        if self.y is not None:
            self.y_scaled = self.scaler.transform(
                self.y.reshape(-1, 1)
            ).reshape(self.y.shape)
            print("  ✓ Target normalisé")
        else:
            self.y_scaled = None
        
        # === 6. CRÉER LE MASQUE ===
        self.mask = (~np.isnan(self.X)).astype(np.float32)
        
        print(f"  ✓ Dataset prêt : {self.X.shape[0]} séries × {self.X.shape[1]} timesteps")
        print(f"    - Valeurs masquées : {np.isnan(self.X).sum():,} ({np.isnan(self.X).mean()*100:.1f}%)")
    
    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, idx):
        # Valeur observée (ou 0 si masqué)
        x = torch.FloatTensor(self.X_scaled[idx]).unsqueeze(-1)  # (timesteps, 1)
        
        # Masque (1=observé, 0=manquant)
        mask = torch.FloatTensor(self.mask[idx]).unsqueeze(-1)   # (timesteps, 1)
        
        # Interpolation linéaire (toujours disponible)
        x_interp = torch.FloatTensor(self.X_interp_scaled[idx]).unsqueeze(-1)  # (timesteps, 1)
        
        # Concatener les 3 features : [valeur, mask, interpolation]
        x_full = torch.cat([x, mask, x_interp], dim=-1)  # (timesteps, 3)
        
        if self.y_scaled is not None:
            y = torch.FloatTensor(self.y_scaled[idx]).unsqueeze(-1)
            return x_full, mask, y
        return x_full, mask
