# train.py
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from tensorflow.keras.models import Sequential, save_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# Configuration
RANDOM_STATE = 42
SEQ_LENGTH = 60
EPOCHS = 50
BATCH_SIZE = 32

# 1. Chargement et prétraitement des données
def load_data(path):
    df = pd.read_csv(path, sep=";")
    
    # Conversion des types
    df["pression"] = df["pression"].str.replace(",", ".").astype(float)
    df["debit"] = df["debit"].str.replace(",", ".").astype(float)
    df['Time'] = pd.to_datetime(df['Time'], format='%d.%m.%Y %H:%M:%S.%f')
    
    # Nettoyage
    return df.dropna()

# 2. Feature Engineering
def create_features(df, dates_echec):
    # Création de la target
    df["target"] = 0
    for date in dates_echec:
        window = (df["Time"] >= date - pd.Timedelta(hours=30)) & (df["Time"] <= date)
        df.loc[window, "target"] = 1

    # Caractéristiques temporelles
    df['hour'] = df['Time'].dt.hour
    df['day'] = df['Time'].dt.day
    df['month'] = df['Time'].dt.month

    # Features techniques
    df['pression_diff'] = df['pression'].diff()
    df['debit_diff'] = df['debit'].diff()
    
    # Fenêtres glissantes
    window_sizes = [5, 10, 800]
    for window in window_sizes:
        for col in ['pression', 'debit']:
            df[f'{col}_rolling_mean_{window}'] = df[col].rolling(window=window).mean()
            df[f'{col}_rolling_std_{window}'] = df[col].rolling(window=window).std()
    
    df['pression_debit_ratio'] = df['pression'] / df['debit']
    
    return df.dropna()

# 3. Préparation des séquences LSTM
def create_sequences(data, targets, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(targets[i+seq_length-1])
    return np.array(X), np.array(y)

if __name__ == "__main__":
    # Chargement des données
    df = load_data("/kaggle/input/klklkl/ba9.csv")
    dates_echec = [
        "23.11.2024 7:50:18.870000", 
        "13.11.2024 14:00:13.960000", 
        "05.11.2024 23:00:19.725000"
    ]
    dates_echec = [pd.to_datetime(d, format='%d.%m.%Y %H:%M:%S.%f') for d in dates_echec]
    
    # Feature Engineering
    df = create_features(df, dates_echec)
    
    # Sélection des features
    features = [
        'pression', 'debit', 'pression_diff', 'debit_diff',
        'pression_rolling_mean_800', 'pression_rolling_std_800',
        'debit_rolling_mean_800', 'debit_rolling_std_10',
        
        'pression_debit_ratio', 'hour', 'day', 'month'
    ]
    
    X = df[features]
    y = df['target']
    
    # Split des données
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.3, 
        random_state=RANDOM_STATE, 
        stratify=y
    )
    
    # Normalisation
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Sauvegarde du scaler
    joblib.dump(scaler, 'scaler.pkl')
    
    # 4. Entraînement Random Forest
    rf_model = RandomForestClassifier(
        n_estimators=100,
        class_weight='balanced',
        random_state=RANDOM_STATE
    )
    rf_model.fit(X_train_scaled, y_train)
    joblib.dump(rf_model, 'modele_random_forest.pkl')
    
    # 5. Entraînement LSTM
    # Création des séquences
    X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train.values, SEQ_LENGTH)
    X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test.values, SEQ_LENGTH)
    
    # Construction du modèle
    model_lstm = Sequential([
        LSTM(64, activation='relu', input_shape=(SEQ_LENGTH, len(features)), return_sequences=True),
        Dropout(0.2),
        LSTM(32, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    
    model_lstm.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Entraînement
    history = model_lstm.fit(
        X_train_seq, y_train_seq,
        validation_data=(X_test_seq, y_test_seq),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[EarlyStopping(patience=10, restore_best_weights=True)],
        verbose=1
    )
    
    # Sauvegarde du modèle LSTM
    save_model(model_lstm, 'modele_lstm.h5')
    
    print("✅ Entraînement terminé et modèles sauvegardés !")

    # 6. Évaluation
    print("\nÉvaluation Random Forest:")
    y_pred = rf_model.predict(X_test_scaled)
    print(classification_report(y_test, y_pred))
    
    print("\nÉvaluation LSTM:")
    lstm_pred = (model_lstm.predict(X_test_seq) > 0.5).astype(int)
    print(classification_report(y_test_seq, lstm_pred))