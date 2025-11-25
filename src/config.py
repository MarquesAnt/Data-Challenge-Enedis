# config.py

from features.base import BaseValueFeature, MaskFeature
from features.interpolation import InterpolationFeature
from features.temporal import TemporalFeatures
from models.feature_extractors import FeatureExtractor
import torch

class ModelConfig:
    """Configuration centralisée du modèle"""
    
    def __init__(self):
        # Modèle
        self.hidden_size = 128
        self.num_layers = 3
        self.dropout = 0.3
        
        # Entraînement
        self.batch_size = 64
        self.learning_rate = 0.0005
        self.num_epochs = 15
        self.patience = 10
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Pipeline
        self.do_finetuning = False
        
        # === FEATURES (modulaire!) ===
        self.features = self._build_features()
        self.feature_extractor = FeatureExtractor(self.features)
    
    def _build_features(self):
        """Construire la liste de features à utiliser"""
        features = [
            BaseValueFeature(),              # Valeur observée
            MaskFeature(),                   # Mask
        ]
        
        # Ajouter interpolation si souhaité
        if self.use_interpolation:
            features.append(InterpolationFeature())
        
        # Ajouter features temporelles si souhaité
        if self.use_temporal:
            features.append(TemporalFeatures(
                include_hour=True,
                include_day=True,
                include_weekend=True,
                cyclic=True  # sin/cos encoding
            ))
        
        return features
    
    # Flags pour activer/désactiver features
    use_interpolation = True
    use_temporal = False
    use_clusters = False