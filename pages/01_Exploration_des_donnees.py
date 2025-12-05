import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ==============================
# Config Streamlit
# ==============================
st.set_page_config(page_title="Exploration des données", page_icon="📊", layout="wide")

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
    
    .indicator-box {
        background-color: #f0f8ff;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 3px solid #2196F3;
        color: #000000 !important;
    }
    
    .indicator-box p, .indicator-box li, .indicator-box span {
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
    
    .pca-explanation {
        background-color: #e8f5e9;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 3px solid #4CAF50;
        color: #000000 !important;
    }
    
    .pca-explanation p, .pca-explanation li, .pca-explanation span {
        color: #000000 !important;
    }
    
    /* Cacher le branding Streamlit */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Style pour les titres de section */
    .section-header {
        margin-top: 2rem;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e0e0e0;
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
</style>
""", unsafe_allow_html=True)

# ==============================
# Titre 
# ==============================

st.title("📊 Exploration des données")
st.markdown("<hr>", unsafe_allow_html=True)

# ==============================
# Chargement des données
# ==============================
@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    df["TotalAmount"] = pd.to_numeric(df["TotalAmount"], errors="coerce")
    df = df.dropna(subset=["TotalAmount"])
    return df

df = load_data("Amazon.csv")

# ==============================
# Aperçu du dataset
# ==============================
st.markdown("<h3 class='section-header'>📋 Aperçu du dataset</h3>", unsafe_allow_html=True)

with st.expander(" Aperçu des données (5 premières lignes)"):
        st.write(df.head(5))

with st.expander(" Statistiques descriptives"):
        st.write(df.describe())

# ==============================
# Histogramme des montants totaux
# ==============================
st.markdown("<h3 class='section-header'> Distribution des montants totaux</h3>", unsafe_allow_html=True)

bins = st.slider("Nombre de classes (bins)", min_value=20, max_value=120, value=50, step=5, key="hist_bins")

fig_hist, ax_hist = plt.subplots(figsize=(8, 4))
sns.histplot(df["TotalAmount"], bins=bins, kde=True, ax=ax_hist, color="#3A86FF")
ax_hist.set_xlabel("TotalAmount")
ax_hist.set_ylabel("Fréquence")
ax_hist.set_title("Distribution des montants totaux")
st.pyplot(fig_hist)

# Indicateurs dans une boîte avec fond clair
q_low, q_high = np.percentile(df["TotalAmount"], [5, 95])
st.markdown(f"""
<div class='indicator-box'>
<strong> Indicateurs clés :</strong><br>
• Moyenne = {df['TotalAmount'].mean():.2f}<br>
• Médiane = {df['TotalAmount'].median():.2f}<br>
• 5ème percentile = {q_low:.2f}<br>
• 95ème percentile = {q_high:.2f}
</div>
""", unsafe_allow_html=True)

# Interprétation dans une boîte avec fond clair
st.markdown("""
<div class='interpretation-box'>
<strong>💡 Interprétation :</strong><br>
La majorité des commandes ont des montants faibles à moyens, avec quelques très grosses commandes qui tirent la moyenne vers le haut. 
Cette distribution typiquement asymétrique suggère une clientèle hétérogène avec des comportements d'achat variés.
</div>
""", unsafe_allow_html=True)

# ==============================
# Heatmap des corrélations
# ==============================
st.markdown("<h3 class='section-header'>🔥 Analyse des corrélations</h3>", unsafe_allow_html=True)

st.markdown("""
<div class='interpretation-box' style='margin-bottom: 1.5rem;'>
<strong>Objectif :</strong> Comprendre comment les variables numériques interagissent entre elles.
</div>
""", unsafe_allow_html=True)

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
corr = df[numeric_cols].corr(method="pearson")

fig_corr, ax_corr = plt.subplots(figsize=(10,6))
sns.heatmap(corr, cmap="coolwarm", annot=True, fmt=".2f", vmin=-1, vmax=1, ax=ax_corr)
ax_corr.set_title("Corrélations entre variables numériques")
st.pyplot(fig_corr)

# Interprétation des corrélations
st.markdown("""
<div class='interpretation-box'>
<strong>💡 Interprétation :</strong><br>
<ul>
<li><strong>Corrélations positives fortes</strong> : Le prix unitaire et les taxes sont fortement corrélés au montant total, ce qui est logique.</li>
<li><strong>Corrélations négatives</strong> : Les remises agissent en sens inverse du montant total, ce qui correspond à l'intuition commerciale.</li>
<li><strong>Corrélations faibles</strong> : Certaines variables comme les frais de port montrent peu de corrélation avec les autres, suggérant une logique indépendante.</li>
</ul>
</div>
""", unsafe_allow_html=True)

# ==============================
# ACP
# ==============================
st.markdown("<h3 class='section-header'> Analyse en Composantes Principales (ACP)</h3>", unsafe_allow_html=True)

st.markdown("""
<div class='interpretation-box' style='margin-bottom: 1.5rem;'>
<strong>Objectif :</strong> Réduire la dimensionnalité des données et visualiser leur structure sous-jacente.
</div>
""", unsafe_allow_html=True)

selected_vars = st.multiselect(
    "Sélectionne les variables pour l'ACP", 
    options=numeric_cols, 
    default=["Quantity", "UnitPrice", "Tax", "TotalAmount"],
    key="pca_vars"
)

if len(selected_vars) >= 2:
    X = df[selected_vars].dropna()
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)

    pca = PCA(n_components=2)
    pcs = pca.fit_transform(X_std)
    pca_df = pd.DataFrame(pcs, columns=["PC1", "PC2"])
    
    if "Category" in df.columns:
        pca_df["Category"] = df.loc[X.index, "Category"]

    fig_pca, ax_pca = plt.subplots(figsize=(8,5))
    sns.scatterplot(data=pca_df, x="PC1", y="PC2", hue="Category", palette="tab10", s=50, ax=ax_pca)
    ax_pca.set_title("Projection ACP (PC1 vs PC2)")
    st.pyplot(fig_pca)

    ve = pca.explained_variance_ratio_
    
    # Explications ACP dans une boîte avec fond clair
    st.markdown(f"""
    <div class='indicator-box'>
    <strong> Variance expliquée :</strong><br>
    • PC1 = {ve[0]:.1%}<br>
    • PC2 = {ve[1]:.1%}<br>
    • Total = {(ve[0]+ve[1]):.1%}
    </div>
    """, unsafe_allow_html=True)
    
    # Interprétation détaillée ACP
    st.markdown("""
    <div class='pca-explanation'>
    <strong> Explication des composantes principales :</strong><br>
    
    <strong>PC1</strong> : Principalement lié au montant total et au prix unitaire.<br>
    <strong>PC2</strong> : Principalement lié à la quantité et aux taxes.
    
    <p><strong>Pourquoi ces combinaisons ?</strong></p>
    <p>L'ACP identifie les directions dans lesquelles les données varient le plus. 
    Ici, elle nous montre que certaines variables évoluent ensemble naturellement 
    (par exemple, quand le prix augmente, le montant total augmente généralement aussi), 
    ce qui crée ces "bandes" inclinées dans la visualisation.</p>
    
    <p><strong>Insight business :</strong> Cette structure suggère des comportements d'achat cohérents 
    qui peuvent être exploités pour la segmentation client.</p>
    </div>
    """, unsafe_allow_html=True)
else:
    st.info("Sélectionne au moins 2 variables pour réaliser l'ACP.")

# ==============================
# Conclusion
# ==============================
st.markdown("---")
st.markdown("""
<div class='conclusion-box'>
<h3> Synthèse des insights</h3>

<strong> Histogramme : Distribution des ventes</strong>
<p>• Confirme la présence de quelques très grosses commandes qui influencent la moyenne<br>
• Suggère une segmentation naturelle entre petits, moyens et gros paniers</p>

<strong> Heatmap : Relations entre variables</strong>
<p>• Identifie les leviers qui influencent le montant total (prix, taxes, remises)<br>
• Montre des relations attendues qui valident la qualité des données</p>

<strong> ACP : Structure des données</strong>
<p>• Simplifie la complexité des données en 2 dimensions principales<br>
• Prépare le terrain pour la segmentation et l'analyse de patterns<br>
• Révéle des combinaisons naturelles de variables</p>

<strong> Amorce pour la suite :</strong>
<p>Cette exploration initiale nous donne une solide compréhension des données. 
Nous allons maintenant analyser les <strong>patterns et anomalies</strong> pour affiner la segmentation 
et détecter les transactions atypiques qui pourraient nécessiter une attention particulière.</p>
</div>
""", unsafe_allow_html=True)

# ==============================
# Navigation vers les autres pages
# ==============================
st.markdown("---")
st.markdown("##  Navigation")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div style='text-align: center; padding: 1rem;'>
        <h4>📊 Exploration des données</h4>
        <p><em>Page actuelle</em></p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div style='text-align: center; padding: 1rem;'>
        <h4>🔍 Analyse & Problématique</h4>
        <p>Segmentation et détection d'anomalies</p>
    </div>
    """, unsafe_allow_html=True)
    if st.button("➡️ Accéder à la Partie 2", key="goto_part2", type="primary"):
        st.switch_page("pages/02_Analyse_Problematique.py")

with col3:
    st.markdown("""
    <div style='text-align: center; padding: 1rem;'>
        <h4>🚀 Synthèse & Solutions</h4>
        <p>Recommandations et plan d'action</p>
    </div>
    """, unsafe_allow_html=True)
    if st.button("➡️ Accéder à la Partie 3", key="goto_part3", type="primary"):
        st.switch_page("pages/03_Synthese_Solutions.py")

# ===== Footer =====
st.markdown("---")
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 2rem 0;">
        <p><strong>Dashboard Amazon - Analyse des Ventes</strong> | Version 1.0</p>
        <p>📧 Contact : chahinez.kehal@yahoo.fr | 📅 Dernière mise à jour : Décembre 2025</p>
    </div>
    """, unsafe_allow_html=True)