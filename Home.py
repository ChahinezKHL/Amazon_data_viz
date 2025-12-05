import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

# ==============================
# Config Streamlit
# ==============================
st.set_page_config(
    page_title="Dashboard Amazon - Analyse des Ventes",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================
# Style CSS personnalisé
# ==============================
st.markdown("""
<style>
    /* Style pour la page principale */
    .main-container {
        max-width: 1200px;
        margin: 0 auto;
        padding: 2rem;
    }
    
    .hero-section {
        text-align: center;
        padding: 3rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        color: white;
        margin-bottom: 2rem;
    }
    
    .card {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border-left: 4px solid #FF9900;
        transition: transform 0.3s ease;
        color: #000000 !important;
    }
    
    .card:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
    }
    
    .card p, .card li, .card span, .card div {
        color: #000000 !important;
    }
    
    .plan-container {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        border: 2px solid #e0e0e0;
        transition: all 0.3s ease;
        color: #000000 !important;
        height: 100%;
    }
    
    .plan-container:hover {
        border-color: #FF9900;
        background-color: #fff8e1;
        transform: translateY(-3px);
    }
    
    .plan-number {
        background: #FF9900;
        color: white !important;
        width: 42px;
        height: 42px;
        border-radius: 50%;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-size: 20px;
        font-weight: bold;
        margin-bottom: 12px;
    }
    
    .plan-title {
        color: #1E3A8A !important;
        margin-top: 0.5rem;
        margin-bottom: 0.5rem;
        font-size: 1.3rem;
    }
    
    .plan-description {
        color: #666666 !important;
        font-size: 0.9rem;
        line-height: 1.4;
    }
    
    /* Bouton Amazon orange */
    .amazon-button {
        display: block;
        width: 100%;
        padding: 0.8rem;
        margin-top: 1rem;
        background-color: #FF9900 !important;
        color: white !important;
        border: none;
        border-radius: 5px;
        font-weight: bold;
        cursor: pointer;
        transition: all 0.3s ease;
        text-align: center;
        text-decoration: none;
        font-size: 0.9rem;
    }
    
    .amazon-button:hover {
        background-color: #E68A00 !important;
        color: white !important;
        text-decoration: none;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(255, 153, 0, 0.3);
    }
    
    .amazon-button:active {
        transform: translateY(0);
        box-shadow: 0 2px 4px rgba(255, 153, 0, 0.3);
    }
    
    /* Style pour les boutons Streamlit personnalisés */
    .stButton button {
        background-color: #FF9900 !important;
        color: white !important;
        border: none !important;
        border-radius: 5px !important;
        font-weight: bold !important;
        transition: all 0.3s ease !important;
        width: 100%;
    }
    
    .stButton button:hover {
        background-color: #E68A00 !important;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(255, 153, 0, 0.3);
    }
    
    .stButton button:active {
        transform: translateY(0);
        box-shadow: 0 2px 4px rgba(255, 153, 0, 0.3);
    }
    
    /* Cacher le branding Streamlit */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Stats cards */
    .stats-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 10px;
        padding: 1.5rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ==================== FONCTIONS ====================
@st.cache_data
def load_data(path: str = "Amazon.csv") -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        st.error("Fichier 'Amazon.csv' non trouvé. Place-le dans le même répertoire que ce script.")
        return pd.DataFrame()

    for col in ["TotalAmount", "Quantity", "UnitPrice", "Tax", "ShippingCost"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "OrderDate" in df.columns:
        df["OrderDate"] = pd.to_datetime(df["OrderDate"], errors="coerce")

    df["ShippingCost"] = df.get("ShippingCost", 0).fillna(0)
    df["Estimated_Cost"] = (df.get("UnitPrice", 0) * df.get("Quantity", 0)).fillna(0)
    denom = df.get("TotalAmount", 0).replace(0, np.nan)
    df["Estimated_Profit"] = (df.get("TotalAmount", 0) - df["Estimated_Cost"] - df["ShippingCost"] - df.get("Tax", 0)).fillna(0)
    df["Profit_Margin"] = (df["Estimated_Profit"] / denom * 100).replace([np.inf, -np.inf], np.nan).fillna(0)
    return df

@st.cache_data
def quick_outlier_rate(df: pd.DataFrame) -> float:
    try:
        from sklearn.preprocessing import StandardScaler
        from sklearn.covariance import MinCovDet
        from scipy.stats import chi2
    except Exception:
        return np.nan
    vars_ = [v for v in ["Quantity", "UnitPrice", "Tax", "TotalAmount"] if v in df.columns]
    if len(vars_) < 2:
        return np.nan
    X = df[vars_].dropna()
    if len(X) < 50:
        return np.nan
    X_std = StandardScaler().fit_transform(X)
    mcd = MinCovDet(random_state=123).fit(X_std)
    md2 = mcd.mahalanobis(X_std)
    thr = chi2.ppf(0.975, df=X_std.shape[1])
    return float((md2 > thr).mean())

# ==================== PAGE HOME ====================
def main():
    # ==============================
    # Header avec logo Amazon
    # ==============================
    st.markdown("""
    <div class="hero-section">
        <img src="https://upload.wikimedia.org/wikipedia/commons/a/a9/Amazon_logo.svg" width="250" style="margin-bottom: 1rem;">
        <h1 style="color: white; font-size: 2.5rem; margin-bottom: 0.5rem;">Dashboard d'Analyse des Ventes</h1>
        <p style="color: rgba(255, 255, 255, 0.9); font-size: 1.2rem;">
            Analyse approfondie des données de transactions pour optimiser la performance commerciale
        </p>
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("Chargement des données..."):
        df = load_data()

    if not df.empty:
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Total commandes", f"{len(df):,}")
        with col2:
            st.metric("Produits uniques", f"{df['ProductID'].nunique():,}" if "ProductID" in df.columns else "—")
        with col3:
            st.metric("Clients uniques", f"{df['CustomerID'].nunique():,}" if "CustomerID" in df.columns else "—")
        with col4:
            st.metric("Chiffre d'affaires", f"{df['TotalAmount'].sum():,.0f} €" if "TotalAmount" in df.columns else "—")
        with col5:
            rate = quick_outlier_rate(df)
            st.metric("Transactions atypiques (MCD)", f"{rate*100:.2f} %" if not np.isnan(rate) else "—")

    # ==============================
    # Plan du projet avec cases cliquables
    # ==============================
    st.markdown("## 📋 Plan du Projet")
    st.markdown("Cliquez sur un bouton pour accéder à l'analyse détaillée :")

    # Ligne 1 : Partie 1
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="plan-container">
            <div class="plan-number">1</div>
            <h3 class="plan-title">Exploration des Données</h3>
            <p class="plan-description">Analyse exploratoire complète avec visualisations, statistiques descriptives et préparation des données.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Bouton avec fond orange Amazon
        if st.button("📊 Accéder à la Partie 1", key="btn_part1", type="primary"):
            st.switch_page("pages/01_Exploration_des_donnees.py")
    
    with col2:
        st.markdown("""
        <div class="plan-container">
            <div class="plan-number">2</div>
            <h3 class="plan-title">Analyse & Problématique</h3>
            <p class="plan-description">Segmentation clients, détection d'anomalies et analyse multivariée pour identifier les problématiques clés.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Bouton avec fond orange Amazon
        if st.button("🔍 Accéder à la Partie 2", key="btn_part2", type="primary"):
            st.switch_page("pages/02_Analyse_Problematique.py")
    
    with col3:
        st.markdown("""
        <div class="plan-container">
            <div class="plan-number">3</div>
            <h3 class="plan-title">Synthèse & Solutions</h3>
            <p class="plan-description">Recommandations stratégiques, feuille de route et plan d'action basés sur les insights découverts.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Bouton avec fond orange Amazon
        if st.button("🚀 Accéder à la Partie 3", key="btn_part3", type="primary"):
            st.switch_page("pages/03_Synthese_Solutions.py")

    # ==============================
    # Description du projet
    # ==============================
    st.markdown("## 📖 À propos de ce projet")
    
    st.markdown("""
    <div class="card">
    <h3>🎯 Objectif</h3>
    <p>Ce dashboard fournit une analyse complète des données de ventes Amazon, de l'exploration initiale à la formulation de recommandations stratégiques actionnables pour optimiser la performance commerciale.</p>
    
    <h3>🔧 Méthodologie</h3>
    <ul>
    <li><strong>Partie 1</strong> : Exploration et nettoyage des données</li>
    <li><strong>Partie 2</strong> : Analyse avancée et détection de problématiques</li>
    <li><strong>Partie 3</strong> : Synthèse et plan d'action</li>
    </ul>
    
    <h3>📊 Données analysées</h3>
    <ul>
    <li>Transactions commerciales</li>
    <li>Informations clients</li>
    <li>Données produits</li>
    <li>Indicateurs financiers</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

    # ==============================
    # Guide d'utilisation
    # ==============================
    st.markdown("## 💡 Comment utiliser ce dashboard")
    
    col_guide1, col_guide2 = st.columns(2)
    
    with col_guide1:
        st.markdown("""
        <div class="card">
        <h4>🔄 Parcours recommandé</h4>
        <ol>
        <li>Commencez par la <strong>Partie 1</strong> pour comprendre la structure des données</li>
        <li>Explorez la <strong>Partie 2</strong> pour découvrir les insights et problématiques</li>
        <li>Consultez la <strong>Partie 3</strong> pour les solutions et recommandations</li>
        </ol>
        </div>
        """, unsafe_allow_html=True)
    
    with col_guide2:
        st.markdown("""
        <div class="card">
        <h4>⚡ Fonctionnalités clés</h4>
        <ul>
        <li><strong>Visualisations interactives</strong></li>
        <li><strong>Filtres dynamiques</strong></li>
        <li><strong>Export des résultats</strong></li>
        <li><strong>Navigation fluide</strong></li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    # ==============================
    # Footer
    # ==============================
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 2rem 0;">
        <p><strong>Dashboard Amazon - Analyse des Ventes</strong> | Version 1.0</p>
        <p>📧 Contact : chahinez.kehal@yahoo.fr | 📅 Dernière mise à jour : Décembre 2025</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()