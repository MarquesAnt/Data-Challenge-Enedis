import pandas as pd
from models.BiLSTM import BiLSTMImputer
from train.pre_training import pretrain_model, finetune_model   


def train_bilstm(X_tr, X_test, Y_tr, holed_cols, config,
                 n_epochs_pretrain=20):
    """
    Pipeline complet : pré-entraînement (+ fine-tuning optionnel)
    """

    print("\n" + "="*60)
    print("🚀 ENTRAÎNEMENT BiLSTM")
    print(f"   Features : {config.feature_extractor}")
    print(f"   Fine-tuning activé : {config.do_finetuning}")
    print("="*60)

    # ==============================
    # 1. Créer le modèle
    # ==============================
    input_size = config.feature_extractor.get_input_size()  # ← Calculé auto
    
    model = BiLSTMImputer(
        input_size=input_size,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        dropout=config.dropout
    ).to(config.device)

    print(f"\n✓ Modèle : BiLSTMImputer")
    print(f"  Input size : {input_size}")
    print(f"  Paramètres : {sum(p.numel() for p in model.parameters()):,}")

    # ==============================
    # 2. Préparer les données
    # ==============================
    clean_cols_train = [c for c in X_tr.columns if not c.startswith("holed")]
    clean_cols_test = [c for c in X_test.columns if not c.startswith("holed")]

    X_all_clean = pd.concat([
        X_tr[clean_cols_train],
        X_test[clean_cols_test]
    ], axis=1)

    print(f"\n📂 Données :")
    print(f"  - Total courbes complètes : {X_all_clean.shape[1]:,}")
    print(f"  - Courbes à trous : {len(holed_cols):,}")

    # ==============================
    # 3. PRÉ-ENTRAÎNEMENT
    # ==============================
    model, scaler = pretrain_model(
        model=model,
        X_clean_all=X_all_clean,
        X_holed_reference=X_tr[holed_cols],
        config=config,
        n_epochs_pretrain=n_epochs_pretrain
    )  

    # ==============================
    # 4. FINE-TUNING (optionnel)
    # ==============================
    if config.do_finetuning:
        model = finetune_model(
            model=model,
            scaler=scaler,
            X_tr=X_tr,
            Y_tr=Y_tr,
            holed_cols=holed_cols,
            config=config
        )  
    else:
        print("\n Fine-tuning SKIP")

    print("\n" + "="*60)
    print("✅ ENTRAÎNEMENT TERMINÉ")
    print("="*60)

    return model, scaler