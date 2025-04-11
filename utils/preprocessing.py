import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

def load_data(path):
    """Charge et prétraite les données depuis un fichier CSV"""
    df = pd.read_csv(path, sep=";")
    
    # Conversion des types
    df["pression"] = df["pression"].str.replace(",", ".").astype(float)
    df["debit"] = df["debit"].str.replace(",", ".").astype(float)
    df['Time'] = pd.to_datetime(df['Time'], format='%d.%m.%Y %H:%M:%S.%f')
    
    # Nettoyage
    return df.dropna()

def create_features(df, dates_echec=None):
    """
    Création des features pour les modèles
    Si dates_echec est fourni, crée aussi la target
    """
    # Copie du dataframe pour éviter les modifications en place
    df_features = df.copy()
    
    # Création de la target si dates d'échec fournies
    if dates_echec:
        df_features["target"] = 0
        for date in dates_echec:
            if isinstance(date, str):
                date = pd.to_datetime(date, format='%d.%m.%Y %H:%M:%S.%f')
            window = (df_features["Time"] >= date - pd.Timedelta(hours=30)) & (df_features["Time"] <= date)
            df_features.loc[window, "target"] = 1

    # Caractéristiques temporelles
    df_features['hour'] = df_features['Time'].dt.hour
    df_features['day'] = df_features['Time'].dt.day
    df_features['month'] = df_features['Time'].dt.month

    # Features techniques
    df_features['pression_diff'] = df_features['pression'].diff()
    df_features['debit_diff'] = df_features['debit'].diff()
    
    # Fenêtres glissantes
    window_sizes = [5, 10, 800]
    for window in window_sizes:
        for col in ['pression', 'debit']:
            df_features[f'{col}_rolling_mean_{window}'] = df_features[col].rolling(window=window).mean()
            df_features[f'{col}_rolling_std_{window}'] = df_features[col].rolling(window=window).std()
    
    # Ratio pression/débit
    df_features['pression_debit_ratio'] = df_features['pression'] / df_features['debit']
    
    # Différence avec la moyenne mobile
    df_features['pression_mean_diff_800'] = df_features['pression'] - df_features['pression_rolling_mean_800']
    df_features['debit_mean_diff_800'] = df_features['debit'] - df_features['debit_rolling_mean_800']
    
    return df_features.dropna()

def get_selected_features():
    """Renvoie la liste des features utilisées par les modèles"""
    return [
        'pression', 'debit', 'pression_diff', 'debit_diff',
        'pression_rolling_mean_800', 'pression_rolling_std_800',
        'debit_rolling_mean_800', 'debit_rolling_std_10',
       
        'pression_debit_ratio', 'hour', 'day', 'month'
    ]

def create_sequences(data, targets=None, seq_length=60):
    """
    Crée des séquences pour le modèle LSTM
    Si targets n'est pas fourni, renvoie seulement X
    """
    X = []
    for i in range(len(data) - seq_length + 1):
        X.append(data[i:i+seq_length])
    
    if targets is not None:
        y = targets[seq_length-1:]
        return np.array(X), np.array(y)
    else:
        return np.array(X)

def preprocess_data(df, scaler_path, with_target=False):
    """
    Prétraite les données pour les prédictions
    
    Args:
        df: DataFrame contenant les données brutes
        scaler_path: Chemin vers le fichier du scaler
        with_target: Indique si la target est présente dans les données
        
    Returns:
        Données prétraitées pour les modèles
    """
    # Chargement du scaler
    scaler = joblib.load(scaler_path)
    
    # Application des features
    if with_target:
        dates_echec = []  # À remplir si on a des dates d'échec connues
        df_features = create_features(df, dates_echec)
    else:
        df_features = create_features(df)
    
    # Extraction des features pertinentes
    features = get_selected_features()
    X = df_features[features]
    
    # Normalisation
    X_scaled = scaler.transform(X)
    
    if with_target:
        y = df_features['target']
        return X_scaled, y, df_features
    else:
        return X_scaled, None, df_features