import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# ============= PRÉ-ENTRAÎNEMENT =============

def analyze_missing_patterns(X_holed):
    """
    Analyser la distribution des trous dans les données réelles
    
    Returns:
        hole_size_distribution: distribution des tailles de trous
        hole_rate_distribution: distribution des taux de NA par courbe
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
        oversample_large: si True, sur-échantillonner les gros trous (>30 timesteps)
    
    Returns:
        X_masked: DataFrame avec masquage réaliste
        Y_true: DataFrame avec vraies valeurs
    """
    np.random.seed(seed)
    
    print("Analyse des patterns de trous dans les données réelles...")
    hole_sizes, hole_rates = analyze_missing_patterns(X_holed_reference)
    
    print(f"✓ Trous analysés :")
    print(f"  - Nombre total de trous : {len(hole_sizes)}")
    print(f"  - Taille moyenne : {hole_sizes.mean():.1f} timesteps")
    print(f"  - Taille médiane : {np.median(hole_sizes):.0f} timesteps")
    print(f"  - Min/Max : {hole_sizes.min()}/{hole_sizes.max()}")
    
    # Analyser la distribution
    small_holes = hole_sizes[hole_sizes <= 12]
    medium_holes = hole_sizes[(hole_sizes > 12) & (hole_sizes < 48)]
    large_holes = hole_sizes[hole_sizes >= 48]
    
    print(f"\n  Distribution :")
    print(f"  - Petits trous (1-12) : {len(small_holes)} ({len(small_holes)/len(hole_sizes)*100:.1f}%)")
    print(f"  - Moyens (13-47) : {len(medium_holes)} ({len(medium_holes)/len(hole_sizes)*100:.1f}%)")
    print(f"  - Gros (48+) : {len(large_holes)} ({len(large_holes)/len(hole_sizes)*100:.1f}%)")
    
    # Sur-échantillonnage des gros trous
    if oversample_large and len(large_holes) > 0:
        print(f"\n  ⚡ Sur-échantillonnage des gros trous activé !")
        # Multiplier par 5 les trous de 48+
        large_holes_repeated = np.repeat(large_holes, 5)
        hole_sizes_augmented = np.concatenate([hole_sizes, large_holes_repeated])
        
        print(f"  - Gros trous avant : {len(large_holes)} ({len(large_holes)/len(hole_sizes)*100:.1f}%)")
        print(f"  - Gros trous après : {len(large_holes)*6} ({len(large_holes)*6/len(hole_sizes_augmented)*100:.1f}%)")
    else:
        hole_sizes_augmented = hole_sizes
    
    print(f"  - Taux de NA moyen : {hole_rates.mean()*100:.1f}%")
    
    # Vérifier qu'il n'y a pas de NaN dans X_clean
    if X_clean.isna().any().any():
        print("⚠️  ATTENTION : X_clean contient des NaN !")
        print(f"Nombre de NaN : {X_clean.isna().sum().sum()}")
        X_clean = X_clean.interpolate(method='linear', axis=0).fillna(method='bfill').fillna(method='ffill')
        print("✓ NaN remplacés par interpolation")
    
    X_masked = X_clean.copy()
    Y_true = X_clean.copy()
    
    print(f"\nCréation de masques réalistes...")
    
    # Statistiques pour validation
    created_hole_sizes = []
    
    # Pour chaque courbe
    for col in tqdm(X_masked.columns, desc="Masking"):
        series_length = len(X_masked)
        
        # Tirer un taux de NA depuis la distribution réelle
        target_na_rate = np.random.choice(hole_rates)
        target_na_count = int(series_length * target_na_rate)
        
        if target_na_count == 0:
            continue
        
        # Créer des trous en suivant la distribution augmentée
        masked_count = 0
        attempts = 0
        max_attempts = 1000
        
        while masked_count < target_na_count and attempts < max_attempts:
            attempts += 1
            
            # Tirer une taille de trou depuis la distribution augmentée
            hole_size = np.random.choice(hole_sizes_augmented)
            
            # Limiter la taille si nécessaire
            remaining = target_na_count - masked_count
            hole_size = min(hole_size, remaining, series_length - masked_count)
            
            # Choisir une position de départ aléatoire
            if series_length - hole_size <= 0:
                break
            start_idx = np.random.randint(0, series_length - hole_size)
            end_idx = start_idx + hole_size
            
            # Vérifier que la zone n'est pas déjà masquée
            zone = X_masked.iloc[start_idx:end_idx][col]
            if not zone.isna().any():
                # Masquer ce bloc
                X_masked.iloc[start_idx:end_idx, X_masked.columns.get_loc(col)] = np.nan
                masked_count += hole_size
                created_hole_sizes.append(hole_size)
    
    # Vérifications
    assert not Y_true.isna().any().any(), "Y_true ne doit PAS contenir de NaN !"
    
    # Statistiques finales
    total_values = X_clean.shape[0] * X_clean.shape[1]
    masked_values = X_masked.isna().sum().sum()
    
    print(f"\n✓ Masqué {masked_values:,} valeurs sur {total_values:,} ({masked_values/total_values*100:.1f}%)")
    print(f"✓ Distribution des trous imitée depuis les données réelles")
    
    # Vérifier la distribution créée
    created_hole_sizes = np.array(created_hole_sizes)
    if len(created_hole_sizes) > 0:
        created_small = created_hole_sizes[created_hole_sizes <= 12]
        created_medium = created_hole_sizes[(created_hole_sizes > 12) & (created_hole_sizes < 48)]
        created_large = created_hole_sizes[created_hole_sizes >= 48]
        
        print(f"\n  Distribution créée :")
        print(f"  - Petits (1-12) : {len(created_small)} ({len(created_small)/len(created_hole_sizes)*100:.1f}%)")
        print(f"  - Moyens (13-47) : {len(created_medium)} ({len(created_medium)/len(created_hole_sizes)*100:.1f}%)")
        print(f"  - Gros (48+) : {len(created_large)} ({len(created_large)/len(created_hole_sizes)*100:.1f}%)")
    
    return X_masked, Y_true


def pretrain_model(model, X_tr, clean_cols, holed_cols, config, n_epochs_pretrain=20):
    """
    Pré-entraîner le modèle sur les courbes complètes
    
    Args:
        model: BiLSTMImputer non entraîné
        X_tr: DataFrame complet
        clean_cols: colonnes des courbes complètes
        holed_cols: colonnes des courbes à trous (pour analyser les patterns)
        config: Config object
        n_epochs_pretrain: nombre d'epochs de pré-entraînement
    
    Returns:
        model: modèle pré-entraîné
        scaler: scaler utilisé
    """
    print("\n" + "="*60)
    print("PHASE 1 : PRÉ-ENTRAÎNEMENT sur les 20k courbes complètes")
    print("="*60)
    
    # Créer données masquées avec distribution réaliste
    X_masked, Y_true = create_masked_data_realistic(
        X_clean=X_tr[clean_cols],
        X_holed_reference=X_tr[holed_cols],  # Analyser les vrais trous
        seed=42
    )
    
    # Split train/val (90/10 car on a beaucoup de données)
    n_clean = len(clean_cols)
    n_train = int(0.9 * n_clean)
    
    train_cols_pretrain = clean_cols[:n_train]
    val_cols_pretrain = clean_cols[n_train:]
    
    print(f"\nPré-entraînement : {len(train_cols_pretrain)} train, {len(val_cols_pretrain)} val")
    
    # Créer datasets
    from bilstm_model import TimeSeriesDataset  # Importer depuis ton code
    
    train_dataset = TimeSeriesDataset(
        X_masked[train_cols_pretrain],
        Y_true[train_cols_pretrain],
        fit_scaler=True
    )
    
    val_dataset = TimeSeriesDataset(
        X_masked[val_cols_pretrain],
        Y_true[val_cols_pretrain],
        scaler=train_dataset.scaler
    )
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,  # Changé de 2 à 0 pour Colab
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,  # Changé de 2 à 0 pour Colab
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # Entraîner
    from bilstm_model import train_model  # Importer depuis ton code
    
    print(f"\nDébut du pré-entraînement ({n_epochs_pretrain} epochs)...")
    
    # Sauvegarder config original et réduire le LR pour stabilité
    original_epochs = config.num_epochs
    original_lr = config.learning_rate
    config.num_epochs = n_epochs_pretrain
    config.learning_rate = 0.0001  # LR plus faible pour le pré-entraînement
    
    print(f"Learning rate réduit à {config.learning_rate} pour stabilité")
    
    model = train_model(model, train_loader, val_loader, config, train_dataset.scaler)
    
    # Restaurer config
    config.num_epochs = original_epochs
    config.learning_rate = original_lr
    
    print("\n✓ Pré-entraînement terminé !")
    print("Le modèle a appris les patterns généraux de consommation électrique")
    
    return model, train_dataset.scaler


def finetune_model(model, scaler, X_tr, Y_tr, holed_cols, config):
    """
    Fine-tuner le modèle pré-entraîné sur les vraies courbes à trous
    
    Args:
        model: modèle pré-entraîné
        scaler: scaler du pré-entraînement
        X_tr: DataFrame avec courbes à trous
        Y_tr: vraies valeurs
        holed_cols: colonnes des courbes à trous
        config: Config object
    
    Returns:
        model: modèle fine-tuné
    """
    print("\n" + "="*60)
    print("PHASE 2 : FINE-TUNING sur les 1000 courbes à trous réels")
    print("="*60)
    
    # Split train/val
    n_holed = len(holed_cols)
    n_train = int(0.8 * n_holed)
    
    train_cols = holed_cols[:n_train]
    val_cols = holed_cols[n_train:]
    
    print(f"Fine-tuning : {len(train_cols)} train, {len(val_cols)} val")
    
    # Créer datasets (réutiliser le scaler du pré-entraînement!)
    from bilstm_model import TimeSeriesDataset
    
    train_dataset = TimeSeriesDataset(
        X_tr[train_cols],
        Y_tr[train_cols],
        scaler=scaler,
        fit_scaler=False  # Important : ne pas refitter le scaler!
    )
    
    val_dataset = TimeSeriesDataset(
        X_tr[val_cols],
        Y_tr[val_cols],
        scaler=scaler,
        fit_scaler=False
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
    
    # Fine-tuning avec learning rate plus faible
    print(f"\nDébut du fine-tuning avec LR réduit...")
    
    from bilstm_model import train_model
    
    # Réduire le learning rate pour le fine-tuning
    original_lr = config.learning_rate
    config.learning_rate = original_lr * 0.1  # 10x plus petit
    
    model = train_model(model, train_loader, val_loader, config, scaler)
    
    # Restaurer learning rate
    config.learning_rate = original_lr
    
    print("\n✓ Fine-tuning terminé !")
    
    return model


# ============= FONCTION PRINCIPALE =============

def train_with_pretraining(X_tr, Y_tr, holed_cols, clean_cols, config, 
                           n_epochs_pretrain=20):
    """
    Pipeline complet : pré-entraînement + fine-tuning
    
    Args:
        X_tr: DataFrame avec toutes les courbes
        Y_tr: vraies valeurs des courbes à trous
        holed_cols: colonnes des courbes à trous
        clean_cols: colonnes des courbes complètes
        config: Config object
        n_epochs_pretrain: epochs de pré-entraînement
    
    Returns:
        model: modèle entraîné
        scaler: scaler utilisé
    """
    print("\n" + "="*60)
    print("🚀 ENTRAÎNEMENT AVEC PRÉ-ENTRAÎNEMENT")
    print("="*60)
    print(f"Courbes complètes (pré-entraînement) : {len(clean_cols)}")
    print(f"Courbes à trous (fine-tuning) : {len(holed_cols)}")
    print("="*60)
    
    # Créer le modèle
    from bilstm_model import BiLSTMImputer
    
    model = BiLSTMImputer(
        input_size=2,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        dropout=config.dropout
    ).to(config.device)
    
    print(f"\nModèle créé : {sum(p.numel() for p in model.parameters()):,} paramètres")
    
    # PHASE 1 : Pré-entraînement
    model, scaler = pretrain_model(
        model, 
        X_tr, 
        clean_cols,
        holed_cols,  # Ajouté pour analyser les patterns
        config,
        n_epochs_pretrain=n_epochs_pretrain
    )
    
    # PHASE 2 : Fine-tuning
    model = finetune_model(
        model,
        scaler,
        X_tr,
        Y_tr,
        holed_cols,
        config
    )
    
    print("\n" + "="*60)
    print("✅ ENTRAÎNEMENT COMPLET TERMINÉ !")
    print("="*60)
    
    return model, scaler


# ============= UTILISATION =============
"""
# Importer les classes du code précédent
from bilstm_model import Config, BiLSTMImputer, TimeSeriesDataset, train_model

# Configuration
config = Config()

# Entraîner avec pré-entraînement
model, scaler = train_with_pretraining(
    X_tr=X_tr,
    Y_tr=Y_tr,
    holed_cols=holed_cols,
    clean_cols=clean_cols,
    config=config,
    n_epochs_pretrain=15  # Ajuster selon le temps disponible
)

# Prédire sur X_test
from bilstm_model import predict

X_test_holed_cols = [c for c in X_test.columns if c.startswith("holed")]
X_test_imputed = predict(model, X_test[X_test_holed_cols], scaler, config)

# Sauvegarder
X_test_imputed.to_csv("predictions_bilstm_pretrained.csv")
print("\n✅ Prédictions sauvegardées dans 'predictions_bilstm_pretrained.csv'")
"""