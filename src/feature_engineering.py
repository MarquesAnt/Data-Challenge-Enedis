# src/feature_engineering.py
import numpy as np
import pandas as pd
from tqdm import tqdm


def make_XY_val(df: pd.DataFrame, Y: pd.DataFrame, holed_cols: list,
                n_lags: int = 3, n_leads: int = 1,
                rolling_windows: list = [6, 12, 24, 48]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Crée X_val et Y_val enrichis pour imputation temporelle :
        - lags / leads configurables
        - stats globales (range, std_rel)
        - stats locales (moyennes/écarts-types glissants)
        - deltas lag/lead
        - encodage temporel (hour_sin/cos)
        - profil horaire moyen
        - ajout de isweekend
        - ajout de trend_12h (moyenne glissante 12h - 48h)
        - ajout de slope_6h (pente locale sur 6 pas)
    """
    all_rows, y_val_rows = [], []

    # ====== Features temporelles communes ======
    df_feat_time = pd.DataFrame({
        'weekday': df.index.weekday,
        'hour': df.index.hour,
        'minute': df.index.minute
    }, index=df.index)

    # ====== Stats globales par colonne ======
    df_stats = df.describe().T[['mean', 'std', 'min', 'max']]
    df_stats['range'] = df_stats['max'] - df_stats['min']
    df_stats['std_rel'] = df_stats['std'] / df_stats['mean']

    # ====== Profil horaire moyen ======
    profiles = df.groupby([df.index.weekday, df.index.hour]).mean()

    # ====== Construction des features pour chaque trou ======
    for col in tqdm(holed_cols, desc="Création de X_val enrichi"):
        series = df[col]
        missing_idx = series[series.isna()].index
        if missing_idx.empty:
            continue

        col_mean, col_std = series.mean(skipna=True), series.std(skipna=True)

        # --- calculs globaux par colonne ---
        rolling_mean_12h = series.rolling(window=12, min_periods=1, center=True).mean()
        rolling_mean_48h = series.rolling(window=48, min_periods=1, center=True).mean()
        trend_12h_series = rolling_mean_12h - rolling_mean_48h

        # pente sur 6h : diff moyenne sur 6 pas
        slope_6h_series = series.diff().rolling(window=6, min_periods=1).mean()

        for idx in missing_idx:
            if idx not in df.index:
                continue
            i = df.index.get_loc(idx)
            hour_val = df_feat_time.at[idx, 'hour'] + df_feat_time.at[idx, 'minute'] / 60.0

            row = {
                'col': col,
                'timestamp': idx,
                'mean': col_mean,
                'std': col_std,
                'range': df_stats.loc[col, 'range'],
                'std_rel': df_stats.loc[col, 'std_rel'],
                'weekday': df_feat_time.at[idx, 'weekday'],
                'isweekend': int(df_feat_time.at[idx, 'weekday'] in [5, 6]),
                'hour': hour_val,
                'hour_sin': np.sin(2 * np.pi * hour_val / 24),
                'hour_cos': np.cos(2 * np.pi * hour_val / 24),
                'profile_hour_mean': profiles.at[
                    (df_feat_time.at[idx, 'weekday'], df_feat_time.at[idx, 'hour']), col
                ],
                'trend_12h': trend_12h_series.loc[idx] if idx in trend_12h_series.index else np.nan,
                'slope_6h': slope_6h_series.loc[idx] if idx in slope_6h_series.index else np.nan,
            }

            # ---- Lags et Leads ----
            for l in range(1, n_lags + 1):
                row[f'lag_{l}'] = series.iloc[i - l] if i - l >= 0 else np.nan
            for l in range(1, n_leads + 1):
                row[f'lead_{l}'] = series.iloc[i + l] if i + l < len(series) else np.nan

            # ---- Deltas ----
            if 'lag_1' in row and 'lag_2' in row:
                row['delta_lag_1'] = (
                    row['lag_1'] - row['lag_2']
                    if pd.notna(row['lag_1']) and pd.notna(row['lag_2'])
                    else np.nan
                )
            else:
                row['delta_lag_1'] = np.nan

            if 'lead_1' in row and 'lead_2' in row:
                row['delta_lead_1'] = (
                    row['lead_2'] - row['lead_1']
                    if pd.notna(row['lead_2']) and pd.notna(row['lead_1'])
                    else np.nan
                )
            else:
                row['delta_lead_1'] = np.nan

            # ---- Moyennes/écarts glissants ----
            for win in rolling_windows:
                window_vals = series[max(0, i - win):min(len(series), i + win)].dropna()
                row[f'rolling_mean_{win}h'] = window_vals.mean() if len(window_vals) else col_mean
                row[f'rolling_std_{win}h'] = window_vals.std() if len(window_vals) else col_std
                mean_val = row.get(f'rolling_mean_{win}h', np.nan)
                std_val = row.get(f'rolling_std_{win}h', np.nan)
                row[f'rolling_stability_{win}h'] = (
                    mean_val / std_val if pd.notna(mean_val) and pd.notna(std_val) and std_val != 0 else np.nan
                )
            row["slope_6h_rel"] = row["slope_6h"] / (abs(row["rolling_mean_6h"]) + 1e-6)


            # ---- Append final ----
            y_val_rows.append({'timestamp': idx, 'col': col, 'y_true': Y.loc[idx, col]})
            all_rows.append(row)

    # ====== Assemblage final ======
    X_val = pd.DataFrame(all_rows).set_index('timestamp')
    Y_val = pd.DataFrame(y_val_rows).set_index('timestamp')

    print(f"X_val : {len(X_val):,} lignes | {X_val.shape[1]} features")
    return X_val, Y_val




def make_train_fast(
    df: pd.DataFrame,
    clean_cols: list,
    n_samples: int = 40000,
    n_lags: int = 3,
    n_leads: int = 1,
    rolling_windows: list = [6, 12, 24, 48],
    random_state: int = 42
) -> pd.DataFrame:
    """
    Crée un dataset d'entraînement enrichi, aligné avec make_XY_val :
        - tirage aléatoire de n_samples points parmi les colonnes propres
        - features : mean, std, range, std_rel, weekday, hour, hour_sin/cos
        - profil horaire moyen
        - lags, leads, deltas
        - moyennes/écarts-types glissants
        - stats locales (local_mean, local_std)
        - nouvelle feature : trend_12h (tendance locale)
        - nouvelle feature : slope_6h (pente moyenne sur 6 pas)
        - y (valeur vraie)
    """

    np.random.seed(random_state)
    all_rows = []

    # ====== Pré-calcul des features globales ======
    df_feat_time = pd.DataFrame({
        "weekday": df.index.weekday,
        "hour": df.index.hour,
        "minute": df.index.minute
    }, index=df.index)

    # Stats globales par colonne
    df_stats = df.describe().T[["mean", "std", "min", "max"]]
    df_stats["range"] = df_stats["max"] - df_stats["min"]
    df_stats["std_rel"] = df_stats["std"] / df_stats["mean"]

    # Profil horaire moyen
    profiles = df.groupby([df.index.weekday, df.index.hour]).mean()

    # ====== Préparation des positions valides ======
    print(" Préparation des positions valides...")
    valid_positions = []
    for col in tqdm(clean_cols, desc="Scan des colonnes propres"):
        notna_idx = np.where(~df[col].isna())[0]
        for i in notna_idx:
            valid_positions.append((col, i))

    total_positions = len(valid_positions)
    print(f"\n {total_positions:,} valeurs disponibles pour tirage aléatoire.")
    n_samples = min(n_samples, total_positions)

    # ====== Tirage aléatoire ======
    sampled_idx = np.random.choice(total_positions, size=n_samples, replace=False)
    sampled_positions = np.array(valid_positions, dtype=object)[sampled_idx]

    # ====== Construction du dataset ======
    print("\n Construction du dataset d'entraînement enrichi...")
    for col, i in tqdm(sampled_positions, desc="Création des lignes", total=n_samples):
        ts = df.index[i]
        series = df[col]

        hour_val = ts.hour + ts.minute / 60.0

        # ---- Calcul de la tendance locale globale pour la colonne ----
        rolling_mean_12h = series.rolling(window=12, min_periods=1, center=True).mean()
        rolling_mean_48h = series.rolling(window=48, min_periods=1, center=True).mean()
        trend_12h_series = rolling_mean_12h - rolling_mean_48h

        # ---- Features globales ----
        col_mean = series.mean(skipna=True)
        col_std = series.std(skipna=True)

        row = {
            "col": col,
            "timestamp": ts,
            "weekday": ts.weekday(),
            "isweekend": int(ts.weekday() in [5,6]),
            "hour": hour_val,
            "hour_sin": np.sin(2 * np.pi * hour_val / 24),
            "hour_cos": np.cos(2 * np.pi * hour_val / 24),
            "mean": col_mean,
            "std": col_std,
            "range": df_stats.loc[col, "range"],
            "std_rel": df_stats.loc[col, "std_rel"],
            "profile_hour_mean": profiles.at[(ts.weekday(), ts.hour), col],
            "y": series.iat[i],
            # ---- Nouvelle feature ----
            "trend_12h": trend_12h_series.iat[i] if not np.isnan(trend_12h_series.iat[i]) else 0.0,
        }

        # ---- Pente locale 6h (moyenne des diff sur 6 pas) ----
        past_vals = series.iloc[max(0, i-6):i+1].dropna()
        if len(past_vals) >= 2:
            row["slope_6h"] = past_vals.diff().mean()
        else:
            row["slope_6h"] = 0.0

        # ---- Lags ----
        for l in range(1, n_lags + 1):
            row[f"lag_{l}"] = series.iat[i - l] if i - l >= 0 else np.nan

        # ---- Leads ----
        for l in range(1, n_leads + 1):
            row[f"lead_{l}"] = series.iat[i + l] if i + l < len(series) else np.nan

        # ---- Deltas ----
        row["delta_lag_1"] = (
            row.get("lag_1", np.nan) - row.get("lag_2", np.nan)
            if "lag_2" in row and pd.notna(row.get("lag_1")) and pd.notna(row.get("lag_2"))
            else np.nan
        )
        row["delta_lead_1"] = (
            row.get("lead_2", np.nan) - row.get("lead_1", np.nan)
            if "lead_2" in row and pd.notna(row.get("lead_1")) and pd.notna(row.get("lead_2"))
            else np.nan
        )

        # ---- Moyennes et écarts-types glissants (6h, 12h, 24h, 48h) ----
        for win in rolling_windows:
            window_vals = series[max(0, i - win):min(len(series), i + win)].dropna()
            row[f"rolling_mean_{win}h"] = window_vals.mean() if len(window_vals) else row["mean"]
            row[f"rolling_std_{win}h"] = window_vals.std() if len(window_vals) else row["std"]
            mean_val = row.get(f'rolling_mean_{win}h', np.nan)
            std_val = row.get(f'rolling_std_{win}h', np.nan)
            row[f'rolling_stability_{win}h'] = (
                mean_val / std_val if pd.notna(mean_val) and pd.notna(std_val) and std_val != 0 else np.nan
            )

        # ---- Stats locales (±6 demi-heures) ----
        window_vals = series[max(0, i - 6):min(len(series), i + 6)].dropna()
        row["local_mean"] = window_vals.mean() if len(window_vals) else row["mean"]
        row["local_std"] = window_vals.std() if len(window_vals) else row["std"]
        row["slope_6h_rel"] = row["slope_6h"] / (abs(row["rolling_mean_6h"]) + 1e-6)


        all_rows.append(row)

    # ====== Construction finale ======
    X_train = pd.DataFrame(all_rows).set_index("timestamp")

    print(f"\n Jeu d'entraînement enrichi créé : {len(X_train):,} lignes × {X_train.shape[1]} colonnes")
    return X_train

def make_X_test(df: pd.DataFrame, holed_cols: list,
                n_lags: int = 3, n_leads: int = 1,
                rolling_windows: list = [6, 12, 24, 48]) -> pd.DataFrame:
    """
    Crée X_test enrichi pour imputation temporelle :
        - mêmes features que make_XY_val, sans Y connu
        - conçu pour prédire les trous de df (NaN)

    Args:
        df (pd.DataFrame): Données principales (index = datetime)
        holed_cols (list): Colonnes contenant des trous
        n_lags (int): Nombre de lags à inclure
        n_leads (int): Nombre de leads à inclure
        rolling_windows (list): Fenêtres glissantes (en "unités horaires")

    Returns:
        X_test (pd.DataFrame): Dataset de features pour les trous à prédire
    """

    all_rows = []

    # ====== Features temporelles communes ======
    df_feat_time = pd.DataFrame({
        'weekday': df.index.weekday,
        'hour': df.index.hour,
        'minute': df.index.minute
    }, index=df.index)

    # ====== Stats globales par colonne ======
    df_stats = df.describe().T[['mean', 'std', 'min', 'max']]
    df_stats['range'] = df_stats['max'] - df_stats['min']
    df_stats['std_rel'] = df_stats['std'] / df_stats['mean']

    # ====== Profil horaire moyen ======
    profiles = df.groupby([df.index.weekday, df.index.hour]).mean()

    # ====== Construction des features pour chaque trou ======
    for col in tqdm(holed_cols, desc="Création de X_test enrichi"):
        series = df[col]
        missing_idx = series[series.isna()].index
        col_mean, col_std = series.mean(skipna=True), series.std(skipna=True)

        for idx in missing_idx:
            i = df.index.get_loc(idx)
            hour_val = df_feat_time.at[idx, 'hour'] + df_feat_time.at[idx, 'minute'] / 60.0

            row = {
                'col': col,
                'timestamp': idx,
                'mean': col_mean,
                'std': col_std,
                'range': df_stats.loc[col, 'range'],
                'std_rel': df_stats.loc[col, 'std_rel'],
                'weekday': df_feat_time.at[idx, 'weekday'],
                'hour': hour_val,
                'hour_sin': np.sin(2 * np.pi * hour_val / 24),
                'hour_cos': np.cos(2 * np.pi * hour_val / 24),
                'profile_hour_mean': profiles.at[
                    (df_feat_time.at[idx, 'weekday'], df_feat_time.at[idx, 'hour']), col
                ],
            }

            # ---- Lags et Leads ----
            for l in range(1, n_lags + 1):
                row[f'lag_{l}'] = series.iloc[i - l] if i - l >= 0 else np.nan
            for l in range(1, n_leads + 1):
                row[f'lead_{l}'] = series.iloc[i + l] if i + l < len(series) else np.nan

            # ---- Deltas ----
            if 'lag_1' in row and 'lag_2' in row:
                row['delta_lag_1'] = (
                    row['lag_1'] - row['lag_2']
                    if pd.notna(row['lag_1']) and pd.notna(row['lag_2'])
                    else np.nan
                )
            else:
                row['delta_lag_1'] = np.nan

            if 'lead_1' in row and 'lead_2' in row:
                row['delta_lead_1'] = (
                    row['lead_2'] - row['lead_1']
                    if pd.notna(row['lead_2']) and pd.notna(row['lead_1'])
                    else np.nan
                )
            else:
                row['delta_lead_1'] = np.nan

            # ---- Moyennes/écarts glissants ----
            for win in rolling_windows:
                window_vals = series[max(0, i - win):min(len(series), i + win)].dropna()
                row[f'rolling_mean_{win}h'] = window_vals.mean() if len(window_vals) else col_mean
                row[f'rolling_std_{win}h'] = window_vals.std() if len(window_vals) else col_std
                mean_val = row.get(f'rolling_mean_{win}h', np.nan)
                std_val = row.get(f'rolling_std_{win}h', np.nan)
                row[f'rolling_stability_{win}h'] = (
                    mean_val / std_val if pd.notna(mean_val) and pd.notna(std_val) and std_val != 0 else np.nan
                )

            all_rows.append(row)

    # ====== Assemblage final ======
    X_test = pd.DataFrame(all_rows).set_index('timestamp')

    print(f" X_test : {len(X_test):,} lignes | {X_test.shape[1]} features")
    return X_test


