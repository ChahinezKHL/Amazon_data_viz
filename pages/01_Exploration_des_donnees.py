import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots




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
# Histogramme interactif avec Plotly
# ==============================
st.markdown("<h3 class='section-header'> Distribution des montants totaux</h3>", unsafe_allow_html=True)

# Slider pour les bins
bins = st.slider("Nombre de classes (bins)", min_value=20, max_value=120, value=50, step=5, key="hist_bins")

# Créer l'histogramme interactif avec Plotly
fig = px.histogram(
    df, 
    x="TotalAmount",
    nbins=bins,
    title="Distribution des montants totaux",
    labels={"TotalAmount": "Montant total", "count": "Fréquence"},
    template="plotly_white",
    color_discrete_sequence=["#3A86FF"],
    opacity=0.8,
    marginal="box"  # Ajoute un boxplot en marge
)

# Ajouter une courbe de densité KDE
fig.update_traces(
    marker_line_width=1,
    marker_line_color="white",
    hovertemplate="<b>Intervalle:</b> %{x}<br>" +
                  "<b>Fréquence:</b> %{y}<br>" +
                  "<extra></extra>"
)

# Personnaliser le layout
fig.update_layout(
    height=500,
    hovermode="x unified",
    title_font_size=20,
    xaxis_title_font_size=14,
    yaxis_title_font_size=14,
    showlegend=False
)

# Ajouter des lignes verticales pour les indicateurs
mean_val = df['TotalAmount'].mean()
median_val = df['TotalAmount'].median()
q_low, q_high = np.percentile(df["TotalAmount"], [5, 95])

fig.add_vline(x=mean_val, line_dash="dash", line_color="red", 
              annotation_text=f"Moyenne: {mean_val:.2f}", 
              annotation_position="top right")
fig.add_vline(x=median_val, line_dash="dash", line_color="green",
              annotation_text=f"Médiane: {median_val:.2f}", 
              annotation_position="top left")
fig.add_vline(x=q_low, line_dash="dot", line_color="orange",
              annotation_text=f"5e percentile: {q_low:.2f}")
fig.add_vline(x=q_high, line_dash="dot", line_color="orange",
              annotation_text=f"95e percentile: {q_high:.2f}")

# Afficher le graphique
st.plotly_chart(fig, use_container_width=True)

# Indicateurs dans une boîte avec fond clair
st.markdown(f"""
<div class='indicator-box'>
<strong> Indicateurs clés :</strong><br>
• Moyenne = {df['TotalAmount'].mean():.2f}<br>
• Médiane = {df['TotalAmount'].median():.2f}<br>
• 5ème percentile = {q_low:.2f}<br>
• 95ème percentile = {q_high:.2f}<br>
• Écart-type = {df['TotalAmount'].std():.2f}<br>
• Coefficient de variation = {(df['TotalAmount'].std()/df['TotalAmount'].mean()*100):.1f}%
</div>
""", unsafe_allow_html=True)

# Interprétation dans une boîte avec fond clair
st.markdown("""
<div class='interpretation-box'>
<strong>💡 Interprétation :</strong><br>
La majorité des commandes ont des montants faibles à moyens, avec quelques très grosses commandes qui tirent la moyenne vers le haut. 
Cette distribution typiquement asymétrique suggère une clientèle hétérogène avec des comportements d'achat variés.<br><br>
<strong>Interactivité :</strong>
• Survolez les barres pour voir les détails<br>
• Zoom et dézoom avec la molette de la souris<br>
• Double-cliquez pour réinitialiser la vue<br>
• Utilisez les outils en haut à droite pour exporter
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
# Heatmap simple interactive
# ==============================
st.markdown("<h3 class='section-header'>🔥 Analyse des corrélations</h3>", unsafe_allow_html=True)

st.markdown("""
<div class='interpretation-box' style='margin-bottom: 1.5rem;'>
<strong>Objectif :</strong> Comprendre comment les variables numériques interagissent entre elles.
</div>
""", unsafe_allow_html=True)

# Calculer la matrice de corrélation
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
corr = df[numeric_cols].corr(method="pearson")

# Créer la heatmap interactive
fig = px.imshow(
    corr,
    text_auto=".2f",
    aspect="auto",
    color_continuous_scale="RdBu",
    range_color=[-1, 1],
    labels=dict(color="Corrélation"),
    title="Matrice de corrélation entre variables numériques"
)

# Personnaliser l'affichage
fig.update_traces(
    hovertemplate="<b>%{y}</b> vs <b>%{x}</b><br>" +
                  "Corrélation: %{z:.3f}<br>" +
                  "<extra></extra>"
)

fig.update_layout(
    xaxis=dict(tickangle=45),
    height=600,
    hovermode="closest"
)

st.plotly_chart(fig, use_container_width=True)

# Afficher les corrélations les plus fortes dans un tableau
st.markdown("### Corrélations les plus fortes")

# Créer un DataFrame des corrélations
corr_pairs = []
for i in range(len(corr.columns)):
    for j in range(i+1, len(corr.columns)):
        corr_pairs.append({
            'Variable 1': corr.columns[i],
            'Variable 2': corr.columns[j],
            'Corrélation': corr.iloc[i, j]
        })

corr_df = pd.DataFrame(corr_pairs).sort_values('Corrélation', key=abs, ascending=False)

# Afficher les top 10 corrélations
st.dataframe(
    corr_df.head(10).style.format({'Corrélation': '{:.3f}'})\
        .background_gradient(cmap='RdBu', subset=['Corrélation'], vmin=-1, vmax=1)\
        .bar(subset=['Corrélation'], color=['#d65f5f', '#5fba7d'], align='mid'),
    use_container_width=True
)

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
# ACP Interactive Simplifiée
# ==============================
st.markdown("<h3 class='section-header'>📊 Analyse en Composantes Principales (ACP)</h3>", unsafe_allow_html=True)

st.markdown("""
<div class='interpretation-box' style='margin-bottom: 1.5rem;'>
<strong>Objectif :</strong> Visualiser les données en 2D en conservant le maximum d'information.
</div>
""", unsafe_allow_html=True)

# Sélection rapide
selected_vars = st.multiselect(
    "Variables pour l'ACP", 
    options=numeric_cols,
    default=numeric_cols[:4] if len(numeric_cols) >= 4 else numeric_cols,
    key="pca_simple"
)

if len(selected_vars) >= 2:
    X = df[selected_vars].dropna()
    X_std = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2)
    pcs = pca.fit_transform(X_std)
    
    # DataFrame pour Plotly
    pca_df = pd.DataFrame({
        "PC1": pcs[:, 0],
        "PC2": pcs[:, 1],
        "Taille": np.abs(pcs[:, 0]) + np.abs(pcs[:, 1])  # Pour la taille des points
    })
    
    # Options de coloriage
    color_options = ["Aucun"] + [col for col in df.columns 
                                 if col not in selected_vars 
                                 and df[col].nunique() < 10]
    color_choice = st.selectbox("Colorier par", color_options)
    
    if color_choice != "Aucun":
        pca_df[color_choice] = df.loc[X.index, color_choice].astype(str)
    
    # Création du graphique
    ve = pca.explained_variance_ratio_
    
    if color_choice != "Aucun":
        fig = px.scatter(
            pca_df,
            x="PC1",
            y="PC2",
            color=color_choice,
            size="Taille",
            hover_name=pca_df.index,
            title=f"ACP - {color_choice}",
            labels={
                "PC1": f"PC1 ({ve[0]:.1%})",
                "PC2": f"PC2 ({ve[1]:.1%})"
            },
            opacity=0.6
        )
    else:
        fig = px.scatter(
            pca_df,
            x="PC1",
            y="PC2",
            size="Taille",
            hover_name=pca_df.index,
            title="Analyse en Composantes Principales",
            labels={
                "PC1": f"PC1 ({ve[0]:.1%})",
                "PC2": f"PC2 ({ve[1]:.1%})"
            },
            color_discrete_sequence=["#3A86FF"],
            opacity=0.6
        )
    
    fig.update_traces(
        marker=dict(line=dict(width=1, color='DarkSlateGrey'))
    )
    
    fig.update_layout(
        height=600,
        hovermode="closest",
        title_x=0.5
    )
    
    # Ajouter l'ellipse de confiance si coloré par catégorie
    if color_choice != "Aucun" and pca_df[color_choice].nunique() < 6:
        fig.update_traces(marker=dict(opacity=0.7))
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Indicateurs
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Variance PC1", f"{ve[0]:.1%}")
    with col2:
        st.metric("Variance PC2", f"{ve[1]:.1%}")
    with col3:
        st.metric("Total variance", f"{ve[0]+ve[1]:.1%}")
    
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