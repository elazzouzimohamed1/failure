pip install -r requirements.txt


import streamlit as st
import pandas as pd
import numpy as np
import os
import datetime
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from utils.preprocessing import load_data, preprocess_data, create_sequences, get_selected_features
from utils.model_utils import (
    load_models, predict_rf, predict_lstm, 
    evaluate_model, visualize_predictions, get_feature_importance
)

# Configuration de la page
st.set_page_config(page_title="Maintenance Prédictive", page_icon="🔍", layout="wide")

# Chemins des fichiers
MODELS_DIR = "models"
RF_MODEL_PATH = os.path.join(MODELS_DIR, "modele_random_forest.pkl")
LSTM_MODEL_PATH = os.path.join(MODELS_DIR, "modele_lstm.h5")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")

# Fonction pour vérifier si les modèles existent
def check_models_exist():
    return all(os.path.exists(p) for p in [RF_MODEL_PATH, LSTM_MODEL_PATH, SCALER_PATH])

# Titre de l'application
st.title("🔍 Application de Maintenance Prédictive")
st.markdown("""
Cette application utilise des modèles d'apprentissage automatique (Random Forest et LSTM) 
pour prédire les défaillances d'équipements à partir de données de capteurs (pression et débit).
""")

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Sélectionnez une page", ["Accueil", "Analyse de données", "Prédictions", "Évaluation des modèles", "À propos"])

# Vérifier l'existence des modèles
models_exist = check_models_exist()
if not models_exist:
    st.warning("⚠️ Les modèles entraînés n'ont pas été trouvés. Veuillez exécuter le script d'entraînement d'abord.")

# Page d'accueil
if page == "Accueil":
    st.header("📊 Système de Maintenance Prédictive")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Objectifs du projet")
        st.markdown("""
        - **Prédire les défaillances** avant qu'elles ne se produisent
        - **Réduire les temps d'arrêt** des équipements
        - **Optimiser les interventions** de maintenance
        - **Améliorer la fiabilité** des systèmes
        """)
        
        st.subheader("Modèles utilisés")
        st.markdown("""
        1. **Random Forest**: Un modèle robuste pour la classification qui utilise plusieurs arbres de décision
        2. **LSTM (Long Short-Term Memory)**: Un réseau de neurones récurrent adapté aux séquences temporelles
        """)
    
    with col2:
        st.subheader("Comment utiliser cette application")
        st.markdown("""
        1. **Analyse de données**: Visualisez et explorez les données des capteurs
        2. **Prédictions**: Téléchargez de nouvelles données pour prédire les défaillances
        3. **Évaluation des modèles**: Examinez les performances des modèles
        4. **À propos**: Informations sur le développement de ce projet
        """)
        
        st.info("💡 Pour commencer, chargez des données dans l'onglet 'Analyse de données' ou 'Prédictions'")

# Page d'analyse de données
elif page == "Analyse de données":
    st.header("📈 Analyse de données")
    
    # Upload de fichier
    uploaded_file = st.file_uploader("Chargez un fichier CSV", type=["csv"])
    
    if uploaded_file is not None:
        try:
            # Charger les données
            df = load_data(uploaded_file)
            st.success(f"✅ Données chargées avec succès: {df.shape[0]} lignes et {df.shape[1]} colonnes")
            
            # Afficher les informations des données
            st.subheader("Aperçu des données")
            st.dataframe(df.head())
            
            # Statistiques descriptives
            st.subheader("Statistiques descriptives")
            st.dataframe(df.describe())
            
            # Visualisation des données temporelles
            st.subheader("Évolution temporelle de la pression et du débit")
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df['Time'], y=df['pression'], name='Pression', line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=df['Time'], y=df['debit'], name='Débit', line=dict(color='green'), yaxis='y2'))
            
            fig.update_layout(
                title='Évolution de la pression et du débit au fil du temps',
                xaxis_title='Date et heure',
                yaxis_title='Pression',
                yaxis2=dict(
                    title='Débit',
                    overlaying='y',
                    side='right'
                ),
                height=600
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Analyse de corrélation
            st.subheader("Corrélation entre pression et débit")
            fig = go.Figure(go.Scatter(
                x=df['pression'],
                y=df['debit'],
                mode='markers',
                marker=dict(
                    size=6,
                    color=df['Time'].astype(int),
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title='Temps')
                )
            ))
            
            fig.update_layout(
                title='Corrélation entre pression et débit',
                xaxis_title='Pression',
                yaxis_title='Débit',
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Distributions
            st.subheader("Distribution des variables")
            col1, col2 = st.columns(2)
            
            with col1:
                fig = go.Figure()
                fig.add_trace(go.Histogram(x=df['pression'], nbinsx=30, name='Pression'))
                fig.update_layout(title='Distribution de la pression', xaxis_title='Pression', yaxis_title='Fréquence')
                st.plotly_chart(fig, use_container_width=True)
                
            with col2:
                fig = go.Figure()
                fig.add_trace(go.Histogram(x=df['debit'], nbinsx=30, name='Débit'))
                fig.update_layout(title='Distribution du débit', xaxis_title='Débit', yaxis_title='Fréquence')
                st.plotly_chart(fig, use_container_width=True)
                
        except Exception as e:
            st.error(f"Erreur lors du chargement des données: {str(e)}")
    else:
        st.info("⬆️ Veuillez charger un fichier CSV pour commencer l'analyse")

# Page de prédictions
elif page == "Prédictions":
    st.header("🔮 Prédictions de défaillances")
    
    if not models_exist:
        st.error("Modèles non disponibles. Veuillez exécuter le script d'entraînement.")
    else:
        # Upload de fichier
        uploaded_file = st.file_uploader("Chargez un fichier CSV pour les prédictions", type=["csv"])
        
        if uploaded_file is not None:
            try:
                # Charger les données
                df = load_data(uploaded_file)
                st.success(f"✅ Données chargées avec succès: {df.shape[0]} lignes")
                
                # Prétraitement des données
                with st.spinner("Prétraitement des données..."):
                    X_scaled, _, df_features = preprocess_data(df, SCALER_PATH, with_target=False)
                
                # Chargement des modèles
                with st.spinner("Chargement des modèles..."):
                    rf_model, lstm_model = load_models(RF_MODEL_PATH, LSTM_MODEL_PATH)
                
                # Prédictions
                with st.spinner("Génération des prédictions..."):
                    # Random Forest
                    y_pred_rf, y_pred_proba_rf = predict_rf(rf_model, X_scaled)
                    
                    # LSTM (séquence de 60 points)
                    y_pred_lstm, y_pred_proba_lstm = predict_lstm(lstm_model, X_scaled, seq_length=60)
                
                st.success("✅ Prédictions générées avec succès")
                
                # Affichage des résultats
                st.subheader("Résultats des prédictions")
                
                # Nombre d'alertes détectées
                rf_alerts = sum(y_pred_rf == 1)
                lstm_alerts = sum(~np.isnan(y_pred_lstm) & (y_pred_lstm == 1))
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Alertes Random Forest", rf_alerts)
                with col2:
                    st.metric("Alertes LSTM", lstm_alerts)
                
                # Visualisation des prédictions
                st.subheader("Visualisation des prédictions")
                fig = visualize_predictions(df_features, y_pred_rf, y_pred_proba_rf, y_pred_lstm, y_pred_proba_lstm)
                st.plotly_chart(fig, use_container_width=True)
                
                # Tableau des alertes détectées
                st.subheader("Détails des alertes")
                
                # Combiner les alertes des deux modèles
                all_alerts = []
                
                # Alertes RF
                rf_alert_indices = np.where(y_pred_rf == 1)[0]
                for idx in rf_alert_indices:
                    all_alerts.append({
                        'Date': df_features['Time'].iloc[idx],
                        'Modèle': 'Random Forest',
                        'Probabilité': y_pred_proba_rf[idx],
                        'Pression': df_features['pression'].iloc[idx],
                        'Débit': df_features['debit'].iloc[idx]
                    })
                
                # Alertes LSTM
                lstm_alert_indices = np.where((~np.isnan(y_pred_lstm)) & (y_pred_lstm == 1))[0]
                for idx in lstm_alert_indices:
                    all_alerts.append({
                        'Date': df_features['Time'].iloc[idx],
                        'Modèle': 'LSTM',
                        'Probabilité': y_pred_proba_lstm[idx],
                        'Pression': df_features['pression'].iloc[idx],
                        'Débit': df_features['debit'].iloc[idx]
                    })
                
                if all_alerts:
                    alerts_df = pd.DataFrame(all_alerts)
                    alerts_df = alerts_df.sort_values('Date')
                    st.dataframe(alerts_df)
                    
                    # Exportation des alertes
                    csv = alerts_df.to_csv(index=False)
                    st.download_button(
                        label="Télécharger les alertes (CSV)",
                        data=csv,
                        file_name="alertes_maintenance.csv",
                        mime="text/csv",
                    )
                else:
                    st.info("Aucune alerte détectée dans les données.")
                
                # Importance des caractéristiques (Random Forest)
                st.subheader("Importance des caractéristiques (Random Forest)")
                feature_names = get_selected_features()
                fig = get_feature_importance(rf_model, feature_names)
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Erreur lors de la génération des prédictions: {str(e)}")
                st.exception(e)
        else:
            st.info("⬆️ Veuillez charger un fichier CSV pour générer des prédictions")

# Page d'évaluation des modèles
elif page == "Évaluation des modèles":
    st.header("📊 Évaluation des modèles")
    
    if not models_exist:
        st.error("Modèles non disponibles. Veuillez exécuter le script d'entraînement.")
    else:
        # Upload de fichier avec la vérité terrain
        st.warning("⚠️ Pour évaluer les modèles, veuillez charger un fichier contenant la vérité terrain (target)")
        uploaded_file = st.file_uploader("Chargez un fichier CSV avec la vérité terrain", type=["csv"])
        
        # Option pour définir les dates d'échec
        use_dates = st.checkbox("Définir manuellement les dates d'échec")
        dates_echec = []
        
        if use_dates:
            st.subheader("Dates d'échec")
            
            # Interface pour ajouter des dates d'échec
            col1, col2 = st.columns([3, 1])
            with col1:
                new_date = st.text_input("Nouvelle date d'échec (format: DD.MM.YYYY HH:MM:SS.ffffff)", 
                                         placeholder="ex: 23.11.2024 7:50:18.870000")
            with col2:
                add_button = st.button("Ajouter")
            
            if add_button and new_date:
                try:
                    dates_echec.append(pd.to_datetime(new_date, format='%d.%m.%Y %H:%M:%S.%f'))
                    st.success(f"Date ajoutée: {new_date}")
                except:
                    st.error("Format de date invalide")
            
            # Afficher les dates ajoutées
            if dates_echec:
                st.write("Dates d'échec définies:")
                for i, date in enumerate(dates_echec):
                    st.write(f"{i+1}. {date.strftime('%d.%m.%Y %H:%M:%S.%f')}")
        
        if uploaded_file is not None:
            try:
                # Charger les données
                df = load_data(uploaded_file)
                st.success(f"✅ Données chargées avec succès: {df.shape[0]} lignes")
                
                # Prétraitement
                if use_dates and dates_echec:
                    # Utiliser les dates d'échec définies manuellement
                    X_scaled, y, df_features = preprocess_data(df, SCALER_PATH, with_target=False)
                    df_features = create_features(df, dates_echec)
                    y = df_features['target']
                else:
                    # Vérifier si les données contiennent déjà la colonne target
                    if 'target' in df.columns:
                        X_scaled, y, df_features = preprocess_data(df, SCALER_PATH, with_target=True)
                    else:
                        st.error("Le fichier ne contient pas de colonne 'target' et aucune date d'échec n'a été définie.")
                        st.stop()
                
                # Chargement des modèles
                rf_model, lstm_model = load_models(RF_MODEL_PATH, LSTM_MODEL_PATH)
                
                # Prédictions
                # Random Forest
                y_pred_rf, y_pred_proba_rf = predict_rf(rf_model, X_scaled)
                
                # LSTM
                y_pred_lstm, y_pred_proba_lstm = predict_lstm(lstm_model, X_scaled, seq_length=60)
                
                # Évaluation des modèles
                st.subheader("Évaluation du modèle Random Forest")
                fig_rf, metrics_rf = evaluate_model(y, y_pred_rf, y_pred_proba_rf, "Random Forest")
                st.plotly_chart(fig_rf, use_container_width=True)
                st.dataframe(metrics_rf)
                
                # Filtrer les données pour le LSTM (ignorer les NaN)
                valid_indices = ~np.isnan(y_pred_lstm)
                if valid_indices.any():
                    st.subheader("Évaluation du modèle LSTM")
                    fig_lstm, metrics_lstm = evaluate_model(
                        y[valid_indices], 
                        y_pred_lstm[valid_indices], 
                        y_pred_proba_lstm[valid_indices], 
                        "LSTM"
                    )
                    st.plotly_chart(fig_lstm, use_container_width=True)
                    st.dataframe(metrics_lstm)
                else:
                    st.warning("Impossible d'évaluer le modèle LSTM car aucune prédiction valide n'a été générée.")
                
                # Visualisation des prédictions
                st.subheader("Visualisation des prédictions vs vérité terrain")
                
                fig = go.Figure()
                
                # Données de vérité terrain
                fig.add_trace(go.Scatter(x=df_features['Time'], y=y, name='Vérité terrain', mode='markers',
                                       marker=dict(color='black', size=10, symbol='x')))
                
                # Prédictions RF
                fig.add_trace(go.Scatter(x=df_features['Time'], y=y_pred_proba_rf, name='Probabilité RF', 
                                       line=dict(color='red')))
                
                # Prédictions LSTM
                fig.add_trace(go.Scatter(x=df_features['Time'][valid_indices], y=y_pred_proba_lstm[valid_indices], 
                                       name='Probabilité LSTM', line=dict(color='blue')))
                
                # Seuil de décision
                fig.add_shape(type="line", x0=df_features['Time'].iloc[0], x1=df_features['Time'].iloc[-1], 
                             y0=0.5, y1=0.5, line=dict(color="gray", width=1, dash="dash"))
                
                fig.update_layout(
                    title="Prédictions vs vérité terrain",
                    xaxis_title="Date et heure",
                    yaxis_title="Probabilité / Vérité terrain",
                    height=600
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Erreur lors de l'évaluation des modèles: {str(e)}")
                st.exception(e)
        else:
            st.info("⬆️ Veuillez charger un fichier CSV pour évaluer les modèles")

# Page À propos
elif page == "À propos":
    st.header("ℹ️ À propos du projet")
    
    st.subheader("Projet de maintenance prédictive")
    st.markdown("""
    Cette application a été développée dans le cadre d'un projet de fin d'études en maintenance prédictive.
    Elle utilise des techniques d'apprentissage automatique pour prédire les défaillances d'équipements à partir
    de données de capteurs (pression et débit).
    """)
    
    st.subheader("Modèles utilisés")
    st.markdown("""
    #### 1. Random Forest
    Le modèle Random Forest est un ensemble d'arbres de décision qui:
    - Est robuste contre le surapprentissage
    - Peut gérer des données non linéaires
    - Fournit une mesure d'importance des caractéristiques
    
    #### 2. LSTM (Long Short-Term Memory)
    Le modèle LSTM est un type de réseau de neurones récurrent qui:
    - Est spécialement conçu pour les séquences temporelles
    - Peut capturer des dépendances à long terme
    - Est adapté aux problèmes de prédiction de séries temporelles
    """)
    
    st.subheader("Structure du projet")
    st.markdown("""
    ```
    maintenance_predictive/
    ├── requirements.txt          # Dépendances du projet
    ├── train.py                  # Script d'entraînement
    ├── models/                   # Dossier contenant les modèles entraînés
    │   ├── scaler.pkl
    │   ├── modele_random_forest.pkl
    │   └── modele_lstm.h5
    ├── data/                     # Dossier pour les données
    │   └── ba9.csv               # Données d'origine
    ├── utils/                    # Utilitaires
    │   ├── __init__.py
    │   ├── preprocessing.py      # Fonctions de prétraitement
    │   └── model_utils.py        # Fonctions liées aux modèles
    └── app.py                    # Application principale avec interface
    ```
    """)
    
    st.subheader("Comment utiliser cette application")
    st.markdown("""
    1. **Entraînement des modèles**: Exécutez `train.py` pour entraîner les modèles
    2. **Lancement de l'application**: Exécutez `streamlit run app.py` pour démarrer l'application
    3. **Exploration**: Utilisez les différentes pages pour analyser les données et générer des prédictions
    """)
    
    st.subheader("Prérequis")
    st.markdown("""
    L'application nécessite les bibliothèques Python suivantes:
    - scikit-learn==1.2.2
    - joblib==1.2.0
    - tensorflow==2.13.0
    - pandas==2.0.3
    - numpy==1.24.3
    - matplotlib==3.7.2
    - seaborn==0.12.2
    - streamlit==1.27.0
    - plotly==5.16.1
    
    Installation: `pip install -r requirements.txt`
    """)