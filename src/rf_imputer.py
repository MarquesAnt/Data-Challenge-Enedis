import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
import argparse
import os
from time import time


def impute_with_global_rf(X: pd.DataFrame, output_path: str = None) -> pd.DataFrame:
    """
    Impute missing values in columns starting with 'holed_' using one global RandomForestRegressor
    trained on rows with no missing values.
    """
    start = time()

    # Colonnes à trous
    holed_cols = [c for c in X.columns if c.startswith("holed_")]
    print(f" Found {len(holed_cols)} columns to impute: {holed_cols}")

    # Ajout des features temporelles
    X_feat = X.copy()
    X_feat["hour"] = X_feat.index.hour
    X_feat["dayofweek"] = X_feat.index.dayofweek

    # Normalisation colonne par colonne
    scaler = StandardScaler()
    for c in X.columns:
        X_feat[c] = scaler.fit_transform(X_feat[[c]])

    # Données d'entraînement = lignes sans NA dans les colonnes trouées
    mask_complete = ~X_feat[holed_cols].isna().any(axis=1)
    X_train = X_feat.loc[mask_complete]
    y_train = X_train[holed_cols]
    X_train = X_train.drop(columns=holed_cols)

    print(f" Training set size: {len(X_train)} rows")

    # Modèle unique multi-sorties
    base_model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )
    model = MultiOutputRegressor(base_model)
    model.fit(X_train, y_train)

    print(" Global RandomForest model trained!")

    # Prédiction des valeurs manquantes
    mask_missing = X_feat[holed_cols].isna().any(axis=1)
    X_pred = X_feat.loc[mask_missing].drop(columns=holed_cols)

    y_pred = model.predict(X_pred)
    X_filled = X_feat.copy()
    X_filled.loc[mask_missing, holed_cols] = y_pred

    print(f" Imputed {mask_missing.sum()} rows with missing values in {time() - start:.1f}s")

    # Sauvegarde
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        X_filled.to_csv(output_path)
        print(f" Saved filled data to {output_path}")

    return X_filled


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Global RandomForest imputation for holed_ columns")
    parser.add_argument("--input", required=True, help="Path to input CSV file")
    parser.add_argument("--output", required=True, help="Path to save imputed CSV file")
    args = parser.parse_args()

    X = pd.read_csv(args.input, index_col=0, parse_dates=True)
    impute_with_global_rf(X, args.output)
