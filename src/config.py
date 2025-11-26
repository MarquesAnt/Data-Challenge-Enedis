# config.py

from features.base import BaseValueFeature, MaskFeature
from features.interpolation import InterpolationFeature
from features.temporal import TemporalFeatures
from features.rolling_stats import RollingStatsFeature
from features.profile import HourlyProfileFeature
from models.feature_extractors import FeatureExtractor
import torch

class ModelConfig:
    """Configuration centralisée du modèle"""
    
    def __init__(self):
        # ==============================
        # ARCHITECTURE MODÈLE
        # ==============================
        self.hidden_size = 256  # Augmenté pour plus de capacité
        self.num_layers = 3
        self.dropout = 0.3
        
        # ==============================
        # ENTRAÎNEMENT
        # ==============================
        self.batch_size = 64
        self.learning_rate = 0.0005
        self.num_epochs = 50
        self.patience = 10
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # ==============================
        # PIPELINE
        # ==============================
        self.do_finetuning = False
        
        # ==============================
        # FLAGS FEATURES
        # ==============================
        # IMPORTANT : Définir AVANT de construire feature_extractor
        self.use_interpolation = True
        self.use_temporal = True
        self.use_rolling_stats = True
        self.use_hourly_profile = True
        self.use_clusters = False
        
        # ==============================
        # FEATURE EXTRACTOR
        # ==============================
        # Sera construit plus tard avec build_feature_extractor()
        self.feature_extractor = None
        self.features = []
    
    def build_feature_extractor(self, X_train_clean=None):
        """
        Construire le feature_extractor selon les flags activés
        
        Args:
            X_train_clean: DataFrame des courbes complètes (pour hourly profile)
        
        Returns:
            FeatureExtractor configuré
        """
        features = []
        
        # ==============================
        # BASE (toujours présent)
        # ==============================
        features.append(BaseValueFeature())
        features.append(MaskFeature())
        
        # ==============================
        # INTERPOLATION
        # ==============================
        if self.use_interpolation:
            features.append(InterpolationFeature())
        
        # ==============================
        # ROLLING STATS (top LightGBM)
        # ==============================
        if self.use_rolling_stats:
            features.append(RollingStatsFeature(
                windows=[6, 12],  # 3h et 6h (timestep=30min)
                include_slope=True,
                include_stability=True
            ))
        
        # ==============================
        # HOURLY PROFILE
        # ==============================
        if self.use_hourly_profile:
            if X_train_clean is not None:
                # Fit le profil sur les données complètes
                profile_feature = HourlyProfileFeature(fit_data=X_train_clean)
            else:
                # Pas de fit, utilisera la moyenne globale
                print(" HourlyProfileFeature : pas de X_train_clean fourni, utilisation moyenne globale")
                profile_feature = HourlyProfileFeature()
            features.append(profile_feature)
        
        # ==============================
        # TEMPORAL (basique)
        # ==============================
        if self.use_temporal:
            features.append(TemporalFeatures(
                include_hour=True,
                include_day=True,
                include_weekend=True,
                cyclic=True  # sin/cos encoding
            ))
        
        # ==============================
        # CLUSTERS
        # ==============================
        if self.use_clusters:
            from features.clustering import ClusterFeature
            features.append(ClusterFeature(n_clusters=10))
        
        # ==============================
        # CONSTRUIRE L'EXTRACTEUR
        # ==============================
        self.features = features
        self.feature_extractor = FeatureExtractor(features)
        
        # Log
        print(f"\n✓ Feature extractor construit :")
        print(f"  {self.feature_extractor}")
        for i, feat in enumerate(features):
            print(f"  [{i+1}] {feat.get_name()} (dim={feat.get_dim()})")
        
        return self.feature_extractor