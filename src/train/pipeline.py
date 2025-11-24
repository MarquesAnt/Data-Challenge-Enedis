import json
import pandas as pd

from train.pre_training import (
    pretrain_model,
    finetune_model,
)

from models.BiLSTM import (
    BiLSTMImputer,
    BiLSTMImputerWithInterp
)

from train.datasets.TimeSeries import (
    TimeSeriesDataset,
    TimeSeriesDatasetWithInterp
)


def train_bilstm(X_tr, X_test, Y_tr, holed_cols, config,
                 n_epochs_pretrain=20):
    """
    Pipeline complet : pré-entraînement (+ fine-tuning optionnel),
    avec ou sans interpolation (selon config.use_interpolation).
    
    Args:
        X_tr: DataFrame train
        X_test: DataFrame test
        Y_tr: Vraies valeurs des courbes à trous
        holed_cols: Liste des colonnes à trous
        config: Config object avec use_interpolation, do_finetuning
        n_epochs_pretrain: Nombre d'epochs de pré-entraînement
    
    Returns:
        model: Modèle entraîné
        scaler: Scaler utilisé
    """

    print("\n" + "="*60)
    print(" ENTRAÎNEMENT BiLSTM")
    print(f"   Interpolation comme feature : {config.use_interpolation}")
    print(f"   Fine-tuning activé : {config.do_finetuning}")
    print("="*60)

    # ==============================
    # 1. Sélection dynamique
    # ==============================
    if config.use_interpolation:
        ModelClass = BiLSTMImputerWithInterp
        DatasetClass = TimeSeriesDatasetWithInterp    
        PretrainFn = pretrain_model
        FinetuneFn = finetune_model
        input_size = 3
    
    else:
        ModelClass = BiLSTMImputer
        DatasetClass = TimeSeriesDataset               
        PretrainFn = pretrain_model
        FinetuneFn = finetune_model
        input_size = 2

    # ==============================
    # 2. Créer le modèle
    # ==============================
    model = ModelClass(
        input_size=input_size,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        dropout=config.dropout
    ).to(config.device)

    print(f"\n✓ Modèle créé : {model.__class__.__name__}")
    print(f"  Paramètres : {sum(p.numel() for p in model.parameters()):,}")

    # ==============================
    # 3. Préparer les données complètes
    # ==============================
    
    # Identifier automatiquement les colonnes complètes
    clean_cols_train = [c for c in X_tr.columns if not c.startswith("holed")]
    clean_cols_test = [c for c in X_test.columns if not c.startswith("holed")]

    # Combiner toutes les courbes complètes (60k)
    X_all_clean = pd.concat([
        X_tr[clean_cols_train],
        X_test[clean_cols_test]
    ], axis=1)

    print(f"\n Données préparées :")
    print(f"  - Courbes complètes train : {len(clean_cols_train):,}")
    print(f"  - Courbes complètes test : {len(clean_cols_test):,}")
    print(f"  - TOTAL pour pré-entraînement : {X_all_clean.shape[1]:,}")
    print(f"  - Courbes à trous (fine-tuning) : {len(holed_cols):,}")

    # ==============================
    # 4. PRÉ-ENTRAÎNEMENT
    # ==============================
    print("\n" + "-"*60)
    print("PHASE 1 : PRÉ-ENTRAÎNEMENT")
    print("-"*60)

    # SIGNATURE UNIFIÉE pour les deux fonctions
    model, scaler = PretrainFn(
        model=model,
        X_clean_all=X_all_clean,
        X_holed_reference=X_tr[holed_cols],  # DataFrame pour analyser patterns
        config=config,
        DatasetClass=DatasetClass,
        n_epochs_pretrain=n_epochs_pretrain
    )

    print("✓ Pré-entraînement terminé")

    # ==============================
    # 5. FINE-TUNING (optionnel)
    # ==============================
    if config.do_finetuning:
        print("\n" + "-"*60)
        print("PHASE 2 : FINE-TUNING")
        print("-"*60)
        print(" Attention : Le fine-tuning peut dégrader les performances")
        print("    si mal calibré. Assure-toi que le LR est adapté.")
        
        model = FinetuneFn(
            model=model,
            scaler=scaler,
            X_tr=X_tr,
            Y_tr=Y_tr,
            holed_cols=holed_cols,
            config=config
        )
        
        print("✓ Fine-tuning terminé")
    else:
        print("\n Fine-tuning SKIP (config.do_finetuning=False)")
        print("   → Utilisation du modèle pré-entraîné seul")

    # ==============================
    # 6. Résumé
    # ==============================
    print("\n" + "="*60)
    print("✅ ENTRAÎNEMENT COMPLET TERMINÉ")
    print("="*60)
    print(f"Modèle final : {model.__class__.__name__}")
    print(f"Input size : {input_size}")
    print(f"Fine-tuning : {'Oui' if config.do_finetuning else 'Non'}")

    return model, scaler