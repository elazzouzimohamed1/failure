import joblib
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils.preprocessing import create_sequences

def load_models(rf_model_path, lstm_model_path):
    """Charge les modèles entraînés"""
    rf_model = joblib.load(rf_model_path)
    lstm_model = load_model(lstm_model_path)
    return rf_model, lstm_model

def predict_rf(model, X_scaled):
    """Effectue des prédictions avec le modèle Random Forest"""
    y_pred_proba = model.predict_proba(X_scaled)[:, 1]
    y_pred = model.predict(X_scaled)
    return y_pred, y_pred_proba

def predict_lstm(model, X_scaled, seq_length=60):
    """Effectue des prédictions avec le modèle LSTM"""
    X_seq = create_sequences(X_scaled, targets=None, seq_length=seq_length)
    y_pred_proba = model.predict(X_seq).flatten()
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    # Ajustement des prédictions pour correspondre à la longueur de X_scaled
    # Les seq_length-1 premières valeurs sont remplies par NaN car pas assez de données
    full_y_pred = np.full(X_scaled.shape[0], np.nan)
    full_y_pred_proba = np.full(X_scaled.shape[0], np.nan)
    
    full_y_pred[seq_length-1:] = y_pred
    full_y_pred_proba[seq_length-1:] = y_pred_proba
    
    return full_y_pred, full_y_pred_proba

def evaluate_model(y_true, y_pred, y_pred_proba, model_name):
    """Évalue les performances d'un modèle et génère des visualisations"""
    # Filtrer les valeurs non-NaN
    mask = ~np.isnan(y_pred)
    y_true_filtered = y_true[mask]
    y_pred_filtered = y_pred[mask]
    y_pred_proba_filtered = y_pred_proba[mask]
    
    # Rapport de classification
    report = classification_report(y_true_filtered, y_pred_filtered, output_dict=True)
    
    # Matrice de confusion
    cm = confusion_matrix(y_true_filtered, y_pred_filtered)
    
    # Courbe ROC
    fpr, tpr, _ = roc_curve(y_true_filtered, y_pred_proba_filtered)
    roc_auc = auc(fpr, tpr)
    
    # Courbe Precision-Recall
    precision, recall, _ = precision_recall_curve(y_true_filtered, y_pred_proba_filtered)
    
    # Création de la visualisation avec Plotly
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Matrice de confusion", 
            "Courbe ROC (AUC={:.3f})".format(roc_auc), 
            "Courbe Precision-Recall", 
            "Distribution des probabilités"
        )
    )
    
    # Matrice de confusion
    heatmap = go.Heatmap(
        z=cm, 
        x=["Négatif", "Positif"], 
        y=["Négatif", "Positif"],
        colorscale="Blues",
        showscale=False,
        text=cm,
        texttemplate="%{text}",
    )
    fig.add_trace(heatmap, row=1, col=1)
    
    # Courbe ROC
    fig.add_trace(
        go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC curve'),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash'), name='Random'),
        row=1, col=2
    )
    
    # Courbe Precision-Recall
    fig.add_trace(
        go.Scatter(x=recall, y=precision, mode='lines', name='Precision-Recall'),
        row=2, col=1
    )
    
    # Distribution des probabilités
    fig.add_trace(
        go.Histogram(
            x=y_pred_proba_filtered, 
            nbinsx=50,
            histnorm='probability density',
            name='Probabilités'
        ),
        row=2, col=2
    )
    
    # Personnalisation de la mise en page
    fig.update_layout(
        title=f"Évaluation du modèle {model_name}",
        height=800,
        showlegend=False
    )
    
    # Mise à jour des axes
    fig.update_xaxes(title_text="Faux positifs", row=1, col=2)
    fig.update_yaxes(title_text="Vrais positifs", row=1, col=2)
    fig.update_xaxes(title_text="Recall", row=2, col=1)
    fig.update_yaxes(title_text="Precision", row=2, col=1)
    fig.update_xaxes(title_text="Probabilité", row=2, col=2)
    fig.update_yaxes(title_text="Densité", row=2, col=2)
    
    # Tableau des métriques
    metrics_df = pd.DataFrame({
        'Précision': [report['0']['precision'], report['1']['precision'], report['macro avg']['precision']],
        'Recall': [report['0']['recall'], report['1']['recall'], report['macro avg']['recall']],
        'F1-score': [report['0']['f1-score'], report['1']['f1-score'], report['macro avg']['f1-score']],
        'Support': [report['0']['support'], report['1']['support'], report['macro avg']['support']]
    }, index=['Classe 0', 'Classe 1', 'Moyenne'])
    
    return fig, metrics_df

def visualize_predictions(df_features, y_pred_rf, y_pred_proba_rf, y_pred_lstm, y_pred_proba_lstm):
    """Visualise les prédictions des deux modèles sur les données temporelles"""
    fig = make_subplots(rows=3, cols=1, 
                        subplot_titles=("Pression et Débit", "Prédictions Random Forest", "Prédictions LSTM"),
                        shared_xaxes=True, 
                        vertical_spacing=0.1)
    
    # Données de pression et débit
    fig.add_trace(
        go.Scatter(x=df_features['Time'], y=df_features['pression'], name='Pression', line=dict(color='blue')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=df_features['Time'], y=df_features['debit'], name='Débit', line=dict(color='green'), yaxis='y2'),
        row=1, col=1
    )
    
    # Prédictions du Random Forest
    fig.add_trace(
        go.Scatter(x=df_features['Time'], y=y_pred_proba_rf, name='Proba RF', line=dict(color='red')),
        row=2, col=1
    )
    
    # Mise en évidence des prédictions positives
    pos_indices = np.where(y_pred_rf == 1)[0]
    if len(pos_indices) > 0:
        fig.add_trace(
            go.Scatter(
                x=df_features['Time'].iloc[pos_indices], 
                y=y_pred_proba_rf[pos_indices], 
                mode='markers',
                marker=dict(color='red', size=10, symbol='circle'),
                name='RF Alertes'
            ),
            row=2, col=1
        )
    
    # Prédictions du LSTM
    valid_indices = ~np.isnan(y_pred_proba_lstm)
    fig.add_trace(
        go.Scatter(
            x=df_features['Time'].iloc[valid_indices], 
            y=y_pred_proba_lstm[valid_indices], 
            name='Proba LSTM', 
            line=dict(color='purple')
        ),
        row=3, col=1
    )
    
    # Mise en évidence des prédictions positives du LSTM
    pos_indices_lstm = np.where((y_pred_lstm == 1) & valid_indices)[0]
    if len(pos_indices_lstm) > 0:
        fig.add_trace(
            go.Scatter(
                x=df_features['Time'].iloc[pos_indices_lstm], 
                y=y_pred_proba_lstm[pos_indices_lstm], 
                mode='markers',
                marker=dict(color='purple', size=10, symbol='circle'),
                name='LSTM Alertes'
            ),
            row=3, col=1
        )
    
    # Ajout d'une ligne horizontale à 0.5 pour les seuils de décision
    fig.add_shape(type="line", x0=df_features['Time'].iloc[0], x1=df_features['Time'].iloc[-1], 
                 y0=0.5, y1=0.5, line=dict(color="black", width=1, dash="dash"), row=2, col=1)
    fig.add_shape(type="line", x0=df_features['Time'].iloc[0], x1=df_features['Time'].iloc[-1], 
                 y0=0.5, y1=0.5, line=dict(color="black", width=1, dash="dash"), row=3, col=1)
    
    # Configuration du deuxième axe Y pour le débit
    fig.update_layout(
        yaxis2=dict(
            title="Débit",
            overlaying="y",
            side="right"
        ),
        height=900,
        title_text="Visualisation des prédictions de défaillance",
        showlegend=True
    )
    
    fig.update_xaxes(title_text="Date et heure", row=3, col=1)
    fig.update_yaxes(title_text="Pression", row=1, col=1)
    fig.update_yaxes(title_text="Probabilité défaillance (RF)", row=2, col=1)
    fig.update_yaxes(title_text="Probabilité défaillance (LSTM)", row=3, col=1)
    
    return fig

def get_feature_importance(rf_model, feature_names):
    """Extrait l'importance des features du modèle RandomForest"""
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)
    
    # Créer un graphique
    fig = go.Figure(go.Bar(
        x=importances[indices],
        y=[feature_names[i] for i in indices],
        orientation='h'
    ))
    
    fig.update_layout(
        title='Importance des caractéristiques',
        xaxis_title='Importance relative',
        yaxis_title='Caractéristiques',
        yaxis={'categoryorder':'total ascending'},
        height=500
    )
    
    return fig