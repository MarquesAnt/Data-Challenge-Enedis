import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

def train_validate_imputer(X):
    holed_cols = [c for c in X.columns if c.startswith("holed_")]
    print(f"Found {len(holed_cols)} columns with holes")

    # --- split des colonnes trouées 80/20 ---
    train_cols, val_cols = train_test_split(holed_cols, test_size=0.2, random_state=42)

    # --- features temporelles ---
    X_feat = X.copy()
    X_feat["hour"] = X_feat.index.hour
    X_feat["dayofweek"] = X_feat.index.dayofweek

    # --- normalisation (par colonne) ---
    scaler = StandardScaler()
    for c in X.columns:
        X_feat[c] = scaler.fit_transform(X_feat[[c]])

    # --- uniquement les lignes où les colonnes d’entraînement sont connues ---
    mask_train = ~X_feat[train_cols].isna().any(axis=1)
    X_train = X_feat.loc[mask_train].drop(columns=holed_cols)
    y_train = X_feat.loc[mask_train, train_cols]

    # --- modèle global multi-sorties ---
    model = MultiOutputRegressor(
        RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    )
    model.fit(X_train, y_train)
    print("Model trained on 80% of holed columns")

    # --- validation : prédire les colonnes jamais vues ---
    mask_val = ~X_feat[val_cols].isna().any(axis=1)
    X_val = X_feat.loc[mask_val].drop(columns=holed_cols)
    y_val = X_feat.loc[mask_val, val_cols]

    # prédiction séparée pour chaque colonne de validation
    y_pred = pd.DataFrame(index=y_val.index)
    for col in val_cols:
        y_pred[col] = model.estimator.predict(X_val)[:, train_cols.index(col)] \
            if col in train_cols else np.nan

    # --- métrique ---
    common_idx = y_val.index.intersection(y_pred.index)
    rmse = np.sqrt(mean_squared_error(y_val.loc[common_idx], y_pred.loc[common_idx]))
    print(f"RMSE (validation on 20% unseen columns) = {rmse:.3f}")
