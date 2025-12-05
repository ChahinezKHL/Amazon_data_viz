import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.covariance import MinCovDet
from scipy.stats import chi2

# ==============================
# Config Streamlit
# ==============================
st.set_page_config(page_title="Partie 2 — Analyse et problématique", page_icon="🔍", layout="wide")

# ==============================
# Style CSS personnalisé
# ==============================
st.markdown("""
<style>
    /* Zones avec fond clair manuel : texte noir */
    .interpretation-box {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 4px solid #4CAF50;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        color: #000000 !important;
    }
    
    .interpretation-box p, .interpretation-box li, .interpretation-box span {
        color: #000000 !important;
    }
    
    .outlier-box {
        background-color: #f0f8ff;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 4px solid #2196F3;
        color: #000000 !important;
    }
    
    .outlier-box p, .outlier-box li, .outlier-box span {
        color: #000000 !important;
    }
    
    .conclusion-box {
        background-color: #fff8e1;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 4px solid #FF9800;
        color: #000000 !important;
    }
    
    .conclusion-box p, .conclusion-box li, .conclusion-box span {
        color: #000000 !important;
    }
    
    /* Cacher le branding Streamlit */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ==============================
# Titre
# ==============================
st.title("🔍 Problématique qui se dégage")
st.caption("Objectif : Identifier des patterns et anomalies dans vos données Amazon")

# ==============================
# Chargement des données
# ==============================
@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    
    # Convertir les colonnes numériques
    numeric_cols = ['Quantity', 'UnitPrice', 'Discount', 'Tax', 'ShippingCost', 'TotalAmount']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Convertir la date
    if 'OrderDate' in df.columns:
        df['OrderDate'] = pd.to_datetime(df['OrderDate'], errors='coerce')
    
    # Supprimer les lignes sans TotalAmount
    df = df.dropna(subset=["TotalAmount"])
    
    return df

df = load_data("Amazon.csv")

# Afficher un résumé des données
st.sidebar.markdown("### 📊 Résumé des données")
st.sidebar.metric("Transactions", f"{len(df):,}")
st.sidebar.metric("Clients uniques", f"{df['CustomerID'].nunique():,}")
st.sidebar.metric("Produits uniques", f"{df['ProductID'].nunique():,}")
st.sidebar.metric("Catégories", f"{df['Category'].nunique():,}")

# ==============================
# SECTION 1: Sélection des variables pour l'analyse
# ==============================
st.markdown("---")
st.subheader("📌 1. Sélection des variables pour l'analyse")

# Identifier les colonnes numériques pertinentes
numeric_cols = ['Quantity', 'UnitPrice', 'Discount', 'Tax', 'ShippingCost', 'TotalAmount']
available_numeric = [col for col in numeric_cols if col in df.columns]

st.info(f"**Variables numériques disponibles :** {', '.join(available_numeric)}")

# Sélectionner les variables par défaut
default_vars = ['Quantity', 'UnitPrice', 'Tax', 'TotalAmount']
default_vars = [v for v in default_vars if v in available_numeric]

# Si moins de 2 variables par défaut, prendre les premières disponibles
if len(default_vars) < 2:
    default_vars = available_numeric[:min(4, len(available_numeric))]

# Widget de sélection
selected_vars = st.multiselect(
    "Sélectionnez les variables numériques à analyser (minimum 2 recommandé) :",
    options=available_numeric,
    default=default_vars,
    key="analysis_vars"
)

# ==============================
# SECTION 2: K-means clustering
# ==============================
st.markdown("---")
st.subheader("📊 2. Segmentation des transactions (K-means)")

if len(selected_vars) >= 2:
    # Préparation des données
    X = df[selected_vars].dropna()
    
    if len(X) > 10:  # Au moins 10 observations
        st.info(f"Analyse sur {len(X)} transactions avec les variables : {', '.join(selected_vars)}")
        
        # Normalisation
        scaler = StandardScaler()
        X_std = scaler.fit_transform(X)
        
        # ACP pour visualisation
        pca = PCA(n_components=2)
        pcs = pca.fit_transform(X_std)
        pca_df = pd.DataFrame(pcs, columns=["PC1", "PC2"])
        
        # K-means clustering
        col1, col2 = st.columns([3, 1])
        with col1:
            k = st.slider("Nombre de clusters (k) :", min_value=2, max_value=6, value=3, key="kmeans_k")
        
        kmeans = KMeans(n_clusters=k, random_state=123, n_init=10)
        clusters = kmeans.fit_predict(X_std)
        pca_df["Cluster"] = clusters
        
        # Ajouter des informations sur les transactions
        pca_df["TotalAmount"] = df.loc[X.index, "TotalAmount"].values
        if "Category" in df.columns:
            pca_df["Category"] = df.loc[X.index, "Category"].values
        
        # Visualisation
        fig_kmeans, ax_kmeans = plt.subplots(figsize=(10, 6))
        
        # Utiliser la taille des points pour représenter le montant total
        sizes = (pca_df["TotalAmount"] / pca_df["TotalAmount"].max() * 100) + 20
        
        # Créer le scatter plot
        scatter = ax_kmeans.scatter(pca_df["PC1"], pca_df["PC2"], 
                                   c=pca_df["Cluster"], cmap="Set2", 
                                   s=sizes, alpha=0.7, edgecolors='white', linewidth=0.5)
        
        ax_kmeans.set_title(f"Segmentation des transactions - K-means avec {k} clusters")
        ax_kmeans.set_xlabel(f"Composante Principale 1 ({pca.explained_variance_ratio_[0]:.1%} de variance)")
        ax_kmeans.set_ylabel(f"Composante Principale 2 ({pca.explained_variance_ratio_[1]:.1%} de variance)")
        
        # Légende pour les clusters
        legend1 = ax_kmeans.legend(*scatter.legend_elements(), title="Clusters", loc="upper right")
        ax_kmeans.add_artist(legend1)
        
        st.pyplot(fig_kmeans)
        
        # Caractéristiques des clusters
        st.subheader("📈 Caractéristiques des clusters")
        
        # Ajouter les clusters au dataframe original
        df_clustered = df.loc[X.index].copy()
        df_clustered["Cluster"] = clusters
        
        # Calculer les statistiques par cluster
        cluster_stats = df_clustered.groupby("Cluster")[selected_vars].agg(['mean', 'std', 'count'])
        
        # Afficher les statistiques
        for cluster_num in range(k):
            with st.expander(f"📋 Cluster {cluster_num} - {len(df_clustered[df_clustered['Cluster']==cluster_num])} transactions"):
                cluster_data = df_clustered[df_clustered['Cluster']==cluster_num]
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Nombre de transactions", len(cluster_data))
                    if "TotalAmount" in selected_vars:
                        st.metric("Montant moyen", f"${cluster_data['TotalAmount'].mean():.2f}")
                
                with col2:
                    if "Quantity" in selected_vars:
                        st.metric("Quantité moyenne", f"{cluster_data['Quantity'].mean():.1f}")
                    if "Category" in cluster_data.columns:
                        top_cat = cluster_data['Category'].mode()[0] if not cluster_data['Category'].mode().empty else "N/A"
                        st.metric("Catégorie principale", top_cat)
        
        # Interprétation
        st.markdown("""
        <div class='interpretation-box'>
        <strong>💡 Interprétation de la segmentation :</strong><br>
        
        <strong>Ce que révèle l'analyse K-means :</strong>
        <ul>
        <li><strong>Cluster 0 (Transactions standard)</strong> : Commandes typiques avec des montants moyens</li>
        <li><strong>Cluster 1 (Gros acheteurs)</strong> : Transactions importantes en quantité ou valeur</li>
        <li><strong>Cluster 2 (Petites commandes)</strong> : Achats de faible valeur mais potentiellement fréquents</li>
        </ul>
        
        <strong>Application business :</strong>
        <ul>
        <li><strong>Marketing ciblé</strong> : Offres différentes pour chaque segment</li>
        <li><strong>Service client</strong> : Priorisation des gros acheteurs</li>
        <li><strong>Gestion stock</strong> : Anticiper la demande par segment</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
    else:
        st.warning("Pas assez de données valides pour l'analyse. Veuillez vérifier vos données.")
else:
    st.info("👈 Veuillez sélectionner au moins 2 variables pour commencer l'analyse.")

# ==============================
# SECTION 3: Détection d'outliers MCD
# ==============================

# ==============================
# Outliers — Mahalanobis (robuste MCD) SEUL
# ==============================
st.subheader("📌 Détection d'outliers — Version robuste (MCD)")

# Estimation robuste
mcd = MinCovDet(random_state=123).fit(X_std)
md2_robust = mcd.mahalanobis(X_std)  # distances au carré
p = X_std.shape[1]
thr_robust = chi2.ppf(0.975, df=p)   # seuil théorique à 97.5%
out_robust = md2_robust > thr_robust

# Histogramme
fig_mr, ax_mr = plt.subplots(figsize=(9, 4))
sns.histplot(md2_robust, bins=60, ax=ax_mr, color="#2A9D8F")
ax_mr.axvline(thr_robust, color="red", linestyle="--", label=f"Seuil χ²(0.975, df={p}) = {thr_robust:.2f}")
ax_mr.set_title("Distances de Mahalanobis² (robuste MCD)")
ax_mr.set_xlabel("Mahalanobis²")
ax_mr.legend()
st.pyplot(fig_mr)

# Résumé + lecture business
st.markdown(
    f"**Outliers détectés (robuste)** : **{int(out_robust.sum())}** / {len(md2_robust)} "
    f"(≈ {(out_robust.mean()*100):.2f}%)."
)
       
        
        # Recommandations basées sur les résultats
st.markdown("""
        <div class='interpretation-box' style='background-color: #e8f5e9; border-left-color: #4CAF50;'>
        <strong>🚀 Actions recommandées :</strong>
        
        <strong>1. Pour l'équipe contrôle qualité :</strong>
        <ul>
        <li>Auditer les {outlier_count} transactions détectées</li>
        <li>Vérifier les erreurs potentielles (prix, quantités, remises)</li>
        <li>Documenter les cas légitimes mais exceptionnels</li>
        </ul>
        
        <strong>2. Pour l'équipe data science :</strong>
        <ul>
        <li>Exclure temporairement ces outliers des modèles prédictifs</li>
        <li>Analyser séparément les patterns des outliers</li>
        <li>Mettre en place une surveillance automatique</li>
        </ul>
        
        <strong>3. Pour l'équipe commerciale :</strong>
        <ul>
        <li>Identifier les opportunités business parmi les outliers</li>
        <li>Comprendre pourquoi certaines transactions sont exceptionnelles</li>
        <li>Adapter les stratégies commerciales</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        

# ==============================
# SECTION 4: Synthèse et conclusions
# ==============================
st.markdown("---")



# Conclusion finale
st.markdown("""
<div class='conclusion-box'>
<h4>🧠 Conclusion stratégique</h4>

<strong>Problématique principale identifiée :</strong>
<p>Vos données Amazon révèlent à la fois une <strong>structure segmentée</strong> (groupes homogènes de transactions) 
et la présence de <strong>transactions atypiques</strong> nécessitant investigation.</p>

<strong>Décisions à prendre :</strong>
<ol>
<li><strong>Valider la segmentation</strong> avec l'équipe commerciale pour adapter les stratégies</li>
<li><strong>Auditer les outliers</strong> pour distinguer erreurs, fraudes et opportunités</li>
<li><strong>Automatiser la surveillance</strong> pour une détection en temps réel</li>
<li><strong>Intégrer ces insights</strong> dans les processus décisionnels</li>
</ol>

<strong>Valeur business :</strong>
<p>Cette analyse permet d'optimiser les ressources commerciales, améliorer la qualité des données, 
et identifier des opportunités de croissance ciblées.</p>
</div>
""", unsafe_allow_html=True)

# ==============================
# Navigation
# ==============================
st.markdown("---")
st.markdown("## 🚀 Navigation entre les parties")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("📊 Retour à l'exploration", key="nav_part1", type="primary", use_container_width=True):
        st.switch_page("pages/01_Exploration_des_donnees.py")

with col2:
    st.markdown("""
    <div style='text-align: center; padding: 1rem; background-color: #f0f0f0; border-radius: 10px;'>
        <h4>🔍 Page actuelle</h4>
        <p><em>Analyse & Problématique</em></p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    if st.button("🚀 Vers les solutions", key="nav_part3", type="primary", use_container_width=True):
        st.switch_page("pages/03_Synthese_Solutions.py")

# ===== Footer =====
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem 0;">
    <p><strong>Dashboard Amazon - Analyse des Ventes</strong> | Version 1.0</p>
    <p>📧 Contact : chahinez.kehal@yahoo.fr | 📅 Dernière mise à jour : Décembre 2025</p>
</div>
""", unsafe_allow_html=True)