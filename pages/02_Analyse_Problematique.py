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

# Variables globales pour stocker les résultats
X_std = None
X = None

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
st.markdown("---")
st.subheader("🔍 3. Détection des transactions atypiques (MCD)")

# Vérifier que les conditions sont remplies pour exécuter le MCD
if len(selected_vars) >= 2 and X is not None and X_std is not None and len(X) > 0:
    try:
        # Estimation robuste avec Minimum Covariance Determinant
        mcd = MinCovDet(random_state=123).fit(X_std)
        md2_robust = mcd.mahalanobis(X_std)  # Distances de Mahalanobis au carré
        p = X_std.shape[1]  # Nombre de variables
        thr_robust = chi2.ppf(0.975, df=p)  # Seuil théorique à 97.5%
        out_robust = md2_robust > thr_robust
        
        outlier_count = int(out_robust.sum())
        total_count = len(md2_robust)
        outlier_percent = (outlier_count / total_count) * 100
        
        # Histogramme
        fig_mr, ax_mr = plt.subplots(figsize=(10, 5))
        
        # Définir un fond clair mais pas trop blanc
        fig_mr.patch.set_facecolor('#f0f0f0')
        ax_mr.set_facecolor('#f8f8f8')
        
        # Histogramme des distances
        sns.histplot(md2_robust, bins=60, ax=ax_mr, color="#2A9D8F", 
                    edgecolor='black', linewidth=0.5, alpha=0.8)
        ax_mr.axvline(thr_robust, color="red", linestyle="--", linewidth=2,
                     label=f"Seuil χ²(0.975, df={p}) = {thr_robust:.2f}")
        
        # Zone des outliers
        ax_mr.axvspan(thr_robust, md2_robust.max(), alpha=0.1, color='red')
        
        ax_mr.set_title("Distances de Mahalanobis² (MCD robuste)", 
                       fontsize=14, fontweight='bold', color='#333333')
        ax_mr.set_xlabel("Mahalanobis²", fontsize=12, color='#333333')
        ax_mr.set_ylabel("Fréquence", fontsize=12, color='#333333')
        
        # Personnaliser les ticks
        ax_mr.tick_params(colors='#333333', labelsize=10)
        
        # Personnaliser le cadre
        for spine in ax_mr.spines.values():
            spine.set_color('#666666')
            spine.set_linewidth(1.5)
        
        # Légende
        legend = ax_mr.legend(frameon=True, framealpha=0.9, loc='upper right')
        legend.get_frame().set_facecolor('#ffffff')
        legend.get_frame().set_edgecolor('#666666')
        
        # Ajouter une grille
        ax_mr.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, color='#666666')
        
        st.pyplot(fig_mr)
        
        # Résumé statistique
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Transactions analysées", f"{total_count:,}")
        with col2:
            st.metric("Transactions atypiques", f"{outlier_count:,}")
        with col3:
            st.metric("Pourcentage", f"{outlier_percent:.1f}%")
        
        # Boîte d'interprétation
        st.markdown(f"""
        <div class='outlier-box'>
        <strong>🔍 Interprétation et impact business :</strong><br>
        <ul>
        <li><strong>Points atypiques multi‑variables</strong> : Ces {outlier_count} transactions présentent des combinaisons inhabituelles des variables sélectionnées.</li>
        <li><strong>À auditer en priorité</strong> : Pourraient correspondre à des remises excessives, erreurs prix/quantité, ou fraude possible.</li>
        <li><strong>Prochaine action</strong> : Exporter la liste pour contrôle et décider si on exclut/flag ces cas avant d'évaluer la rentabilité par segment.</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Tableau des outliers
        if outlier_count > 0:
            with st.expander(f"📋 Voir les {min(20, outlier_count)} transactions détectées comme outliers"):
                # Obtenir les indices des outliers
                outlier_indices = X.index[out_robust]
                
                # Créer un dataframe avec les colonnes disponibles
                available_cols = []
                for col in ['OrderID', 'OrderDate', 'CustomerName', 'Category', 'Quantity', 'UnitPrice', 'TotalAmount']:
                    if col in df.columns:
                        available_cols.append(col)
                
                if available_cols:
                    outlier_df = df.loc[outlier_indices, available_cols].copy()
                    outlier_df["Distance_Mahalanobis"] = md2_robust[out_robust]
                    outlier_df = outlier_df.sort_values("Distance_Mahalanobis", ascending=False)
                    
                    st.dataframe(outlier_df.head(20))
                    st.caption(f"Affichage des {min(20, outlier_count)} premières transactions sur {outlier_count} détectées comme outliers.")
                    
                    # Option pour télécharger
                    csv = outlier_df.to_csv(index=False)
                    st.download_button(
                        label=f"📥 Télécharger tous les outliers ({outlier_count} transactions)",
                        data=csv,
                        file_name="outliers_amazon.csv",
                        mime="text/csv",
                        type="primary"
                    )
                else:
                    st.info("Aucune colonne d'identification disponible dans les données.")
        
        # Recommandations basées sur les résultats
        st.markdown(f"""
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
        
    except Exception as e:
        st.error(f"Erreur lors de la détection des outliers: {str(e)}")
        st.info("""
        **Causes possibles :**
        - Variables trop corrélées entre elles
        - Pas assez de données (minimum 20 observations recommandé)
        - Valeurs extrêmes qui perturbent les calculs
        """)
else:
    if len(selected_vars) < 2:
        st.info("👈 Veuillez d'abord sélectionner au moins 2 variables dans la section 1.")
    elif X is None or X_std is None:
        st.info("👈 Veuillez d'abord exécuter l'analyse K-means pour préparer les données.")
    elif len(X) == 0:
        st.warning("Pas assez de données valides après nettoyage.")

# ==============================
# SECTION 4: Synthèse et conclusions
# ==============================
st.markdown("---")
st.subheader("🎯 4. Synthèse des insights")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div style='background-color: #f0f8ff; padding: 1.5rem; border-radius: 10px; border-left: 4px solid #2196F3;'>
    <h4 style='color: #333;'>📊 Segmentation (K-means)</h4>
    <p><strong style='color: #333;'>Points forts :</strong></p>
    <ul style='color: #333;'>
    <li>Identification de profils clients distincts</li>
    <li>Base pour le marketing personnalisé</li>
    <li>Visualisation intuitive via ACP</li>
    </ul>
    <p><strong style='color: #333;'>Limitations :</strong></p>
    <ul style='color: #333;'>
    <li>Choix du nombre k subjectif</li>
    <li>Sensible aux outliers</li>
    <li>Interprétation parfois complexe</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div style='background-color: #fff0f0; padding: 1.5rem; border-radius: 10px; border-left: 4px solid #ff6b6b;'>
    <h4 style='color: #333;'>🔍 Détection outliers (MCD)</h4>
    <p><strong style='color: #333;'>Points forts :</strong></p>
    <ul style='color: #333;'>
    <li>Méthode robuste et statistique</li>
    <li>Multidimensionnelle</li>
    <li>Seuil objectif basé sur χ²</li>
    </ul>
    <p><strong style='color: #333;'>Limitations :</strong></p>
    <ul style='color: #333;'>
    <li>Requiert des données "propres"</li>
    <li>Calcul intensif pour grands datasets</li>
    <li>Interprétation business nécessaire</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

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