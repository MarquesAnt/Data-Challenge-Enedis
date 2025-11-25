import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
from models.BiLSTM import BiLSTMImputer
from train.datasets.TimeSeries import TimeSeriesDataset, TimeSeriesDatasetWithInterp
from train.train_core import train_model   


def predict(model, X_df, scaler, config, batch_size=64):
    """
    Prédire les valeurs manquantes avec le FeatureExtractor de la config
    
    Args:
        model: modèle BiLSTM entraîné
        X_df: DataFrame avec colonnes à prédire
        scaler: StandardScaler utilisé pendant l'entraînement
        config: Config object (contient feature_extractor)
        batch_size: taille des batchs pour prédiction
    
    Returns:
        result: DataFrame avec valeurs imputées
    """
    from train.datasets.TimeSeries import TimeSeriesDataset
    
    print("\n" + "="*60)
    print("PRÉDICTION")
    print("="*60)
    print(f"Features utilisées : {config.feature_extractor}")
    print(f"Colonnes à prédire : {len(X_df.columns)}")

    # 1. Créer le dataset avec FeatureExtractor
    dataset = TimeSeriesDataset(
        X=X_df,
        y=None,
        feature_extractor=config.feature_extractor,  # ← Depuis config
        scaler=scaler,
        fit_scaler=False
    )

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # 2. Prédiction
    model.eval()
    all_predictions = []

    with torch.no_grad():
        for batch_data in tqdm(loader, desc="Predicting"):
            # Gérer le cas où on a (x, mask) ou juste x
            if isinstance(batch_data, (list, tuple)):
                if len(batch_data) == 2:
                    x, mask = batch_data
                else:
                    x = batch_data[0]
                    mask = None
            else:
                x = batch_data
                mask = None
            
            x = x.to(config.device)
            if mask is not None:
                mask = mask.to(config.device)
            
            pred = model(x, mask)
            all_predictions.append(pred.cpu().numpy())

    # 3. Recomposition batchs → matrice complète
    predictions = np.concatenate(all_predictions, axis=0)  # (n_series, timesteps, 1)
    predictions = predictions.squeeze(-1).T                # → (timesteps, n_series)

    # 4. Dénormalisation
    predictions_denorm = scaler.inverse_transform(predictions.reshape(-1, 1))
    predictions_denorm = predictions_denorm.reshape(predictions.shape)

    pred_df = pd.DataFrame(
        predictions_denorm,
        index=X_df.index,
        columns=X_df.columns
    )

    # 5. Remplacement des NaNs uniquement
    result = X_df.copy()
    nan_mask = X_df.isna()
    result[nan_mask] = pred_df[nan_mask]
    
    n_imputed = nan_mask.sum().sum()
    print(f"\n✓ Prédiction terminée : {n_imputed:,} valeurs imputées")

    return result


# ============= Fonction principale =============
def run_bilstm_imputation(X_tr, Y_tr, holed_cols, clean_cols, config):
    """
    Pipeline complet d'entraînement et prédiction
    
    Returns:
        model: modèle entraîné
        scaler: scaler utilisé
        train_loader: pour analyse
        val_loader: pour analyse
    """
    
    # Séparer train/val (80/20 sur les courbes holed)
    n_holed = len(holed_cols)
    n_train = int(0.8 * n_holed)
    
    train_cols = holed_cols[:n_train]
    val_cols = holed_cols[n_train:]
    
    print(f"Training on {len(train_cols)} series, validating on {len(val_cols)} series")
    
    # Créer datasets
    train_dataset = TimeSeriesDataset(
        X_tr[train_cols], 
        Y_tr[train_cols], 
        fit_scaler=True
    )
    
    val_dataset = TimeSeriesDataset(
        X_tr[val_cols], 
        Y_tr[val_cols], 
        scaler=train_dataset.scaler
    )
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.batch_size, 
        shuffle=False
    )
    
    # Créer modèle
    model = BiLSTMImputer(
        input_size=2,  # valeur + mask
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        dropout=config.dropout
    ).to(config.device)
    
    print(f"\nModel architecture:")
    print(model)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Entraîner
    print("\nStarting training...")
    model = train_model(model, train_loader, val_loader, config, train_dataset.scaler)
    
    return model, train_dataset.scaler, train_loader, val_loader