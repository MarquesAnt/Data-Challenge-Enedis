# ============= PRÉ-ENTRAÎNEMENT =============

import numpy as np
import pandas as pd 
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from train.train_core import train_model


def analyze_missing_patterns(X_holed):
    """
    Analyser la distribution des trous dans les données réelles
    
    Returns:
        hole_sizes: array des tailles de trous
        hole_rates: array des taux de NA par courbe
    """
    hole_sizes = []
    hole_rates = []
    
    for col in X_holed.columns:
        series = X_holed[col]
        is_na = series.isna()
        
        # Taux de NA pour cette courbe
        na_rate = is_na.mean()
        hole_rates.append(na_rate)
        
        # Trouver les tailles de blocs consécutifs de NaN
        in_hole = False
        current_hole_size = 0
        
        for val in is_na:
            if val:  # NaN
                if not in_hole:
                    in_hole = True
                    current_hole_size = 1
                else:
                    current_hole_size += 1
            else:  # Pas NaN
                if in_hole:
                    hole_sizes.append(current_hole_size)
                    in_hole = False
                    current_hole_size = 0
        
        # Si la série se termine par un trou
        if in_hole:
            hole_sizes.append(current_hole_size)
    
    return np.array(hole_sizes), np.array(hole_rates)


def create_masked_data_realistic(X_clean, X_holed_reference, seed=42, oversample_large=True):
    """
    Créer des données masquées en imitant la distribution réelle des trous
    avec sur-échantillonnage optionnel des gros trous
    
    Args:
        X_clean: DataFrame avec courbes complètes (pour pré-entraînement)
        X_holed_reference: DataFrame avec vraies courbes à trous (pour analyser les patterns)
        seed: graine aléatoire
        oversample_large: si True, sur-échantillonner les gros trous (>48 timesteps)
    
    Returns:
        X_masked: DataFrame avec masquage réaliste
        Y_true: DataFrame avec vraies valeurs
    """
    np.random.seed(seed)

    # Analyse des données réelles
    hole_sizes, hole_rates = analyze_missing_patterns(X_holed_reference)

    # Découpage par taille
    small_holes = hole_sizes[hole_sizes <= 12]
    medium_holes = hole_sizes[(hole_sizes > 12) & (hole_sizes < 48)]
    large_holes = hole_sizes[hole_sizes >= 48]

    print(f"\nAnalyse des trous réels :")
    print(f"  - Petits (1-12) : {len(small_holes)} ({len(small_holes)/len(hole_sizes)*100:.1f}%)")
    print(f"  - Moyens (13-47) : {len(medium_holes)} ({len(medium_holes)/len(hole_sizes)*100:.1f}%)")
    print(f"  - Gros (48+) : {len(large_holes)} ({len(large_holes)/len(hole_sizes)*100:.1f}%)")

    # Sur-échantillonnage des gros trous
    if oversample_large and len(large_holes) > 0:
        large_holes_repeated = np.repeat(large_holes, 5)
        hole_sizes_augmented = np.concatenate([hole_sizes, large_holes_repeated])
        print(f"\n Sur-échantillonnage ×5 des gros trous")
        print(f"  - Gros trous après : {len(large_holes)*6} ({len(large_holes)*6/len(hole_sizes_augmented)*100:.1f}%)")
    else:
        hole_sizes_augmented = hole_sizes

    # Nettoyage X_clean si besoin
    if X_clean.isna().any().any():
        print(" X_clean contient des NaN, nettoyage en cours...")
        X_clean = (X_clean.interpolate(method='linear', axis=0)
                           .fillna(method='bfill')
                           .fillna(method='ffill'))
        print("✓ NaN nettoyés")

    X_masked = X_clean.copy()
    Y_true = X_clean.copy()

    n_rows = X_clean.shape[0]
    columns = X_clean.columns
    created_hole_sizes = []

    print("\nCréation des masques réalistes (version optimisée NumPy)...")

    # Boucle optimisée sur les colonnes
    for col in tqdm(columns, desc="Masking"):
        
        # NumPy array pour vitesse
        col_values = X_masked[col].values.astype(float)
        
        # Tirer un taux de NA réaliste
        target_na_rate = np.random.choice(hole_rates)
        target_na_count = int(n_rows * target_na_rate)

        if target_na_count == 0:
            continue
        
        # Masque booléen (beaucoup plus rapide que Pandas)
        mask = np.zeros(n_rows, dtype=bool)
        masked_count = 0
        attempts = 0
        max_attempts = 1000

        # Génération optimisée des blocs
        while masked_count < target_na_count and attempts < max_attempts:
            attempts += 1

            # Tirer une taille de trou depuis la distribution réaliste
            hole_size = int(np.random.choice(hole_sizes_augmented))

            # Limiter si nécessaire
            remaining = target_na_count - masked_count
            if hole_size > remaining:
                hole_size = remaining
            if hole_size <= 0:
                break
            
            # Tirer un début possible
            start = np.random.randint(0, n_rows - hole_size)
            end = start + hole_size

            # Vérifier overlap avec NumPy (ultra rapide)
            if not mask[start:end].any():
                mask[start:end] = True
                masked_count += hole_size
                created_hole_sizes.append(hole_size)

        # Appliquer le masque en une seule opération NumPy
        col_values[mask] = np.nan
        X_masked[col] = col_values

    # Statistiques finales
    created_hole_sizes = np.array(created_hole_sizes)
    total_values = X_clean.size
    masked_values = X_masked.isna().sum().sum()

    print(f"\n✓ Masqué {masked_values:,} valeurs sur {total_values:,} "
          f"({masked_values/total_values*100:.1f}%)")

    if len(created_hole_sizes) > 0:
        created_small = np.sum(created_hole_sizes <= 12)
        created_medium = np.sum((created_hole_sizes > 12) & (created_hole_sizes < 48))
        created_large = np.sum(created_hole_sizes >= 48)
        
        print(f"✓ Distribution créée :")
        print(f"  - Petits (1-12) : {created_small} ({created_small/len(created_hole_sizes)*100:.1f}%)")
        print(f"  - Moyens (13-47) : {created_medium} ({created_medium/len(created_hole_sizes)*100:.1f}%)")
        print(f"  - Gros (48+) : {created_large} ({created_large/len(created_hole_sizes)*100:.1f}%)")
    
    return X_masked, Y_true


def pretrain_model(
    model,
    X_clean_all,
    X_holed_reference,
    config,
    DatasetClass,
    n_epochs_pretrain
):
    """
    Pré-entraîne un modèle en utilisant soit :
      - TimeSeriesDataset (classique)
      - TimeSeriesDatasetWithInterp (interpolation)
    selon ce que le pipeline passe.
    
    Args:
        model: modèle BiLSTM non entraîné
        X_clean_all: DataFrame avec TOUTES les courbes complètes (60k)
        X_holed_reference: DataFrame avec courbes à trous (pour analyser patterns)
        config: Config object
        DatasetClass: TimeSeriesDataset ou TimeSeriesDatasetWithInterp
        n_epochs_pretrain: nombre d'epochs
    
    Returns:
        model: modèle pré-entraîné
        scaler: scaler utilisé
    """

    print("\n" + "="*60)
    print("PHASE 1 : PRÉ-ENTRAÎNEMENT GÉNÉRIQUE")
    print("="*60)
    print(f"Nombre de courbes complètes : {len(X_clean_all.columns):,}")
    print(f"Dataset utilisé : {DatasetClass.__name__}")

    # 1. Masquage réaliste
    X_masked, Y_true = create_masked_data_realistic(
        X_clean=X_clean_all,
        X_holed_reference=X_holed_reference,
        seed=42,
        oversample_large=True
    )

    # 2. Split 90/10
    cols = list(X_masked.columns)
    n_train = int(0.9 * len(cols))
    train_cols = cols[:n_train]
    val_cols = cols[n_train:]
    
    print(f"\nPré-entraînement : {len(train_cols):,} train, {len(val_cols):,} val")

    # 3. Dataset TRAIN
    train_dataset = DatasetClass(
        X=X_masked[train_cols],
        y=Y_true[train_cols],
        fit_scaler=True
    )

    # 4. Dataset VAL
    val_dataset = DatasetClass(
        X=X_masked[val_cols],
        y=Y_true[val_cols],
        scaler=train_dataset.scaler
    )

    # 5. Loaders
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

    # 6. Ajuster config pour pré-entraînement
    orig_lr = config.learning_rate
    orig_epochs = config.num_epochs

    # Learning rate adaptatif
    config.learning_rate = 0.0005 if config.use_interpolation else 0.0001
    config.num_epochs = n_epochs_pretrain

    print(f"\nHyperparamètres pré-entraînement :")
    print(f"  - Learning rate : {config.learning_rate:.6f}")
    print(f"  - Epochs max : {config.num_epochs}")
    print(f"  - Patience : {config.patience}")

    # 7. Entraînement
    model = train_model(
        model, train_loader, val_loader, config, train_dataset.scaler
    )

    # 8. Reset config
    config.learning_rate = orig_lr
    config.num_epochs = orig_epochs

    print("✓ Pré-entraînement terminé")
    return model, train_dataset.scaler


def finetune_model(
    model,
    scaler,
    X_tr,
    Y_tr,
    holed_cols,
    config,
    DatasetClass
):
    """
    Fine-tuning générique compatible avec ou sans interpolation.
    
    Args:
        model: modèle pré-entraîné
        scaler: scaler du pré-entraînement
        X_tr: DataFrame X_train
        Y_tr: vraies valeurs
        holed_cols: colonnes à trous
        config: Config object
        DatasetClass: TimeSeriesDataset ou TimeSeriesDatasetWithInterp
    
    Returns:
        model: modèle fine-tuné
    """

    print("\n" + "="*60)
    print("PHASE 2 : FINE-TUNING GÉNÉRIQUE")
    print("="*60)
    print(f"Dataset utilisé : {DatasetClass.__name__}")

    # Split 80/20
    n = len(holed_cols)
    n_train = int(0.8 * n)
    train_cols = holed_cols[:n_train]
    val_cols = holed_cols[n_train:]
    
    print(f"Fine-tuning : {len(train_cols)} train, {len(val_cols)} val")

    # Datasets
    train_dataset = DatasetClass(
        X=X_tr[train_cols],
        y=Y_tr[train_cols],
        scaler=scaler,
        fit_scaler=False
    )
    val_dataset = DatasetClass(
        X=X_tr[val_cols],
        y=Y_tr[val_cols],
        scaler=scaler,
        fit_scaler=False
    )

    # Loaders
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

    # Sauvegarder config original
    orig_lr = config.learning_rate
    orig_epochs = config.num_epochs
    orig_patience = config.patience

    # Ajuster hyperparams pour fine-tuning
    config.learning_rate = orig_lr * 0.3  # ×0.3 au lieu de ×0.1
    config.num_epochs = 15  # Plus court que pré-train
    config.patience = 5     # Plus agressif (800 courbes seulement)

    print(f"\nHyperparamètres fine-tuning :")
    print(f"  - Learning rate : {config.learning_rate:.6f} (x0.3)")
    print(f"  - Epochs max : {config.num_epochs}")
    print(f"  - Patience : {config.patience}")
    print("\n Attention : Le fine-tuning peut dégrader les performances")
    print("    si le modèle pré-entraîné est déjà excellent.")

    # Fine-tuning
    model = train_model(model, train_loader, val_loader, config, scaler)

    # Restaurer config
    config.learning_rate = orig_lr
    config.num_epochs = orig_epochs
    config.patience = orig_patience

    print("✓ Fine-tuning terminé")
    return model