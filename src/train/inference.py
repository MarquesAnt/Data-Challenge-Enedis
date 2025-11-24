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
    Prédire les valeurs manquantes avec ou sans interpolation automatiquement.

    - Si config.use_interpolation = True → TimeSeriesDatasetWithInterp
    - Sinon → TimeSeriesDataset
    """

    # 1. Choix automatique du dataset
    if config.use_interpolation:
        DatasetClass = TimeSeriesDatasetWithInterp
        print("\nPrédiction : interpolation activée")
    else:
        DatasetClass = TimeSeriesDataset
        print("\nPrédiction : modèle standard")

    # 2. Dataset
    dataset = DatasetClass(
        X=X_df,
        y=None,
        scaler=scaler,
        fit_scaler=False
    )

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # 3. Prédiction
    model.eval()
    all_predictions = []

    with torch.no_grad():
        for x, mask in tqdm(loader, desc="Predicting"):
            x, mask = x.to(config.device), mask.to(config.device)
            pred = model(x, mask)
            all_predictions.append(pred.cpu().numpy())

    # 4. Recomposition batchs → matrice complète
    predictions = np.concatenate(all_predictions, axis=0)  # (n_series, timesteps, 1)
    predictions = predictions.squeeze(-1).T                # → (timesteps, n_series)

    # 5. Dénormalisation
    predictions_denorm = scaler.inverse_transform(predictions.reshape(-1, 1))
    predictions_denorm = predictions_denorm.reshape(predictions.shape)

    pred_df = pd.DataFrame(
        predictions_denorm,
        index=X_df.index,
        columns=X_df.columns
    )

    # 6. Remplacement des NaNs uniquement
    result = X_df.copy()
    nan_mask = X_df.isna()
    result[nan_mask] = pred_df[nan_mask]

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