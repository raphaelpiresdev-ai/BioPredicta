"""
MultiTarget BioPredict — Plataforma de Triagem Virtual Multiobjetivo (A2A × hERG)
Aplicação Web Interativa em Streamlit com Explicabilidade SHAP e Editor Molecular 2D Ketcher.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st
from rdkit import Chem
from rdkit.Chem import Descriptors, Draw

from src.chem_utils import (
    DESCRIPTOR_NAMES,
    FEATURE_NAMES,
    FINGERPRINT_NAMES,
    canonicalize_smiles,
    compute_2d_descriptors,
    compute_morgan_fingerprints,
    featurize_dataframe,
)

# -----------------------------------------------------------------------------
# 1. Configuração da Página
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="MultiTarget BioPredict | A2A × hERG",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# 2. Estilização CSS Customizada (Modern Dark Theme & Glassmorphism)
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@400;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    h1, h2, h3, h4, h5, h6 {
        font-family: 'Outfit', sans-serif;
        letter-spacing: -0.02em;
    }

    .main-title {
        font-size: 2.3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #38bdf8 0%, #818cf8 50%, #c084fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }

    .sub-title {
        font-size: 1.05rem;
        color: #94a3b8;
        margin-bottom: 1.5rem;
    }

    .metric-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 1.2rem;
        margin-bottom: 1rem;
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.3);
    }

    .semaphore-green {
        background: rgba(16, 185, 129, 0.12);
        border: 1.5px solid #10b981;
        color: #34d399;
        padding: 0.5rem 1rem;
        border-radius: 9999px;
        font-weight: 700;
        display: inline-block;
        font-size: 1.1rem;
    }

    .semaphore-yellow {
        background: rgba(245, 158, 11, 0.12);
        border: 1.5px solid #f59e0b;
        color: #fbbf24;
        padding: 0.5rem 1rem;
        border-radius: 9999px;
        font-weight: 700;
        display: inline-block;
        font-size: 1.1rem;
    }

    .semaphore-red {
        background: rgba(239, 68, 68, 0.12);
        border: 1.5px solid #ef4444;
        color: #f87171;
        padding: 0.5rem 1rem;
        border-radius: 9999px;
        font-weight: 700;
        display: inline-block;
        font-size: 1.1rem;
    }

    .badge-chip {
        display: inline-block;
        padding: 0.25rem 0.6rem;
        font-size: 0.8rem;
        font-weight: 600;
        border-radius: 6px;
        background: #334155;
        color: #e2e8f0;
        margin-right: 0.3rem;
        margin-bottom: 0.3rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# 3. Carregamento dos Recursos em Cache Local
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_models_and_explainers():
    """Carrega os modelos treinados e instancia os explicadores SHAP em cache."""
    root = Path(__file__).resolve().parent
    model_a2a = joblib.load(root / "models" / "a2a_model.joblib")
    model_herg = joblib.load(root / "models" / "herg_model.joblib")

    with open(root / "models" / "a2a_model_metadata.json", "r", encoding="utf-8") as f:
        meta_a2a = json.load(f)

    with open(root / "models" / "herg_model_metadata.json", "r", encoding="utf-8") as f:
        meta_herg = json.load(f)

    explainer_a2a = shap.TreeExplainer(model_a2a)
    explainer_herg = shap.TreeExplainer(model_herg)

    return model_a2a, model_herg, meta_a2a, meta_herg, explainer_a2a, explainer_herg


try:
    model_a2a, model_herg, meta_a2a, meta_herg, explainer_a2a, explainer_herg = load_models_and_explainers()
    models_loaded = True
except Exception as e:
    models_loaded = False
    load_error_msg = str(e)


# -----------------------------------------------------------------------------
# 4. Dados dos Controles Farmacológicos
# -----------------------------------------------------------------------------
REFERENCE_CONTROLS = {
    "☕ Cafeína (CHEMBL113)": {
        "name": "Cafeína",
        "smiles": "Cn1c(=O)c2c(ncn2C)n(C)c1=O",
        "desc": "Antagonista não-seletivo clássico de adenosina (A1/A2A) com afinidade micromolar moderada e perfil cardiológico seguro.",
    },
    "💊 Istradefilina (CHEMBL223405)": {
        "name": "Istradefilina",
        "smiles": "CCN1C(=O)C2=C(N=C(N2C)/C=C/C3=CC(=C(C=C3)OC)OC)N(C1=O)CC",
        "desc": "Antagonista potente de A2A aprovado pela FDA para Doença de Parkinson. Seguro para hERG em faixas terapêuticas.",
    },
    "⚠️ Terfenadina (CHEMBL22)": {
        "name": "Terfenadina",
        "smiles": "CC(C)(C)c1ccc(cc1)C(O)CCCN2CCC(CC2)C(O)(c3ccccc3)c4ccccc4",
        "desc": "Anti-histamínico cardiotóxico clássico retirado do mercado por induzir prolongamento do intervalo QT e Torsades de Pointes.",
    },
    "🌿 Taurina (Endógeno)": {
        "name": "Taurina",
        "smiles": "NCCS(=O)(=O)O",
        "desc": "Composto endógeno com ação osmorreguladora, sem atividade purinérgica e inativo para bloqueio do canal hERG.",
    },
    "🍵 L-Teanina (Nutracêutico)": {
        "name": "L-Teanina",
        "smiles": "CCNC(=O)CCC(N)C(=O)O",
        "desc": "Aminoácido encontrado em folhas de chá verde, desprovido de ligação a A2A e completamente inócuo ao canal hERG.",
    },
}


# -----------------------------------------------------------------------------
# 5. Barra Lateral (Sidebar)
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🧬 MultiTarget BioPredict")
    st.caption("Triagem Virtual Multiobjetivo $A_{2A}$ × hERG")

    st.markdown("---")
    st.markdown("#### ⚙️ Modo de Entrada")
    input_mode = st.radio(
        "Selecione o método de submissão:",
        ["🧪 Controles de Referência (1-Clique)", "✏️ Inserir SMILES (Texto)", "🎨 Desenhar Molécula (Ketcher)"],
        index=0,
    )

    st.markdown("---")
    st.markdown("#### 📊 Modelos Preditivos Locais")
    st.markdown(
        """
        - **Eficácia Terapêutica ($A_{2A}$):**
          - Algoritmo: `XGBoost Regressor`
          - Métrica: $R^2 = 0{,}723$ | $\text{RMSE} = 0{,}578$
        - **Segurança Cardíaca (hERG):**
          - Algoritmo: `Random Forest Classifier`
          - Métrica: $\text{ROC-AUC} = 0{,}930$ | Brier: $0{,}112$
        """
    )

    st.markdown("---")
    st.markdown("#### 🚦 Semáforos Independentes")
    st.markdown(
        r"""
        **Eficácia $A_{2A}$ ($pIC_{50}$):**
        - 🟢 **Alta Potência:** $pIC_{50} \ge 6{,}5$
        - 🟡 **Moderada:** $5{,}0 \le pIC_{50} < 6{,}5$
        - 🔴 **Baixa/Inativo:** $pIC_{50} < 5{,}0$

        **Segurança hERG ($P_{\text{hERG}}$):**
        - 🟢 **Baixo Risco:** $P < 30\%$
        - 🟡 **Moderado:** $30\% \le P \le 70\%$
        - 🔴 **Alto Risco:** $P > 70\%$
        """
    )
    st.caption("100% Local · AMD Ryzen 7 CPU · 8 GB RAM")


# -----------------------------------------------------------------------------
# 6. Cabeçalho Principal
# -----------------------------------------------------------------------------
st.markdown('<div class="main-title">MultiTarget BioPredict</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Plataforma Computacional de Triagem Virtual Multiobjetivo: '
    'Afinidade ao Receptor de Adenosina A<sub>2A</sub> e Risco de Cardiotoxicidade por Bloqueio do Canal hERG</div>',
    unsafe_allow_html=True,
)

if not models_loaded:
    st.error(f"Erro ao carregar os modelos preditivos em `models/`: {load_error_msg}")
    st.stop()


# -----------------------------------------------------------------------------
# 7. Resolução da Molécula Submetida
# -----------------------------------------------------------------------------
active_smiles = ""
active_name = "Molécula Personalizada"
active_desc = ""

if input_mode == "🧪 Controles de Referência (1-Clique)":
    selected_ctrl = st.selectbox(
        "Selecione um composto de controle farmacológico pré-configurado:",
        list(REFERENCE_CONTROLS.keys()),
        index=0,
    )
    ctrl_data = REFERENCE_CONTROLS[selected_ctrl]
    active_smiles = ctrl_data["smiles"]
    active_name = ctrl_data["name"]
    active_desc = ctrl_data["desc"]
    st.info(f"**{active_name}:** {active_desc}")

elif input_mode == "✏️ Inserir SMILES (Texto)":
    col_input, col_preset = st.columns([3, 1])
    with col_input:
        user_smiles = st.text_input(
            "Insira o código SMILES da molécula:",
            value="Cn1c(=O)c2c(ncn2C)n(C)c1=O",
            placeholder="Ex: CCN1C(=O)C2=C(N=C(N2C)/C=C/C3=CC(=C(C=C3)OC)OC)N(C1=O)CC",
        )
    with col_preset:
        preset_choice = st.selectbox(
            "Exemplos rápidos:",
            ["(Personalizado)", "Cafeína", "Istradefilina", "Terfenadina"],
            index=0,
        )
        if preset_choice == "Cafeína":
            user_smiles = REFERENCE_CONTROLS["☕ Cafeína (CHEMBL113)"]["smiles"]
        elif preset_choice == "Istradefilina":
            user_smiles = REFERENCE_CONTROLS["💊 Istradefilina (CHEMBL223405)"]["smiles"]
        elif preset_choice == "Terfenadina":
            user_smiles = REFERENCE_CONTROLS["⚠️ Terfenadina (CHEMBL22)"]["smiles"]

    active_smiles = user_smiles.strip()

elif input_mode == "🎨 Desenhar Molécula (Ketcher)":
    try:
        from streamlit_ketcher import st_ketcher
        st.markdown("Desenhe a estrutura química no editor bidimensional abaixo:")
        ketcher_smiles = st_ketcher("Cn1c(=O)c2c(ncn2C)n(C)c1=O", height=450)
        active_smiles = ketcher_smiles.strip() if ketcher_smiles else ""
    except Exception as e:
        st.warning(f"Editor Ketcher indisponível ({e}). Por favor, utilize a entrada por código SMILES.")
        active_smiles = st.text_input("SMILES:", value="Cn1c(=O)c2c(ncn2C)n(C)c1=O")


# -----------------------------------------------------------------------------
# 8. Padronização e Featurização da Molécula
# -----------------------------------------------------------------------------
if not active_smiles:
    st.warning("Por favor, insira ou desenhe uma molécula para iniciar a triagem.")
    st.stop()

canonical_smi = canonicalize_smiles(active_smiles)

if canonical_smi is None:
    st.error(
        f"❌ **Estrutura Química Inválida:** O RDKit não pôde interpretar o código submetido (`{active_smiles}`). "
        "Verifique a valência e sintaxe do código SMILES."
    )
    st.stop()

# Criar objeto Mol e featurizar
mol_obj = Chem.MolFromSmiles(canonical_smi)
df_input = pd.DataFrame([{"canonical_smiles": canonical_smi}])
df_feat = featurize_dataframe(df_input, "canonical_smiles")
X_sample = df_feat[list(FEATURE_NAMES)]


# -----------------------------------------------------------------------------
# 9. Inferência Multiobjetivo Concorrente
# -----------------------------------------------------------------------------
# Modelo A2A (XGBoost Regressor)
pred_pIC50 = float(model_a2a.predict(X_sample)[0])
pred_IC50_nM = float(10 ** (9.0 - pred_pIC50))

# Modelo hERG (Random Forest Classifier)
prob_herg = float(model_herg.predict_proba(X_sample)[0, 1])

# Classificação dos Semáforos
if pred_pIC50 >= 6.5:
    a2a_color_class = "semaphore-green"
    a2a_label = "🟢 Alta Potência (Eficaz)"
    a2a_desc = "Composto de alta afinidade e potência nanomolar no receptor A2A. Candidato viável a progressão."
elif pred_pIC50 >= 5.0:
    a2a_color_class = "semaphore-yellow"
    a2a_label = "🟡 Potência Moderada"
    a2a_desc = "Potência intermediária (micromolar). Recomendada otimização de fragmentos (Hit-to-Lead)."
else:
    a2a_color_class = "semaphore-red"
    a2a_label = "🔴 Baixa Afinidade / Inativo"
    a2a_desc = "Composto considerado inativo ou fracamente ligante para o receptor A2A."

if prob_herg < 0.30:
    herg_color_class = "semaphore-green"
    herg_label = "🟢 Baixo Risco (Seguro)"
    herg_desc = "Baixa probabilidade de oclusão do canal de potássio. Excelente indicativo de segurança cardiológica."
elif prob_herg <= 0.70:
    herg_color_class = "semaphore-yellow"
    herg_label = "🟡 Risco Moderado (Atenção)"
    herg_desc = "Zona farmacológica intermediária. Recomenda-se ensaio eletrofisiológico in vitro (Patch-Clamp)."
else:
    herg_color_class = "semaphore-red"
    herg_label = "🔴 Alto Risco (Cardiotóxico)"
    herg_desc = "Elevada propensão ao bloqueio de hERG. Risco crítico de prolongamento QT e arritmias cardíacas letais."


# -----------------------------------------------------------------------------
# 10. Apresentação dos Resultados Principais
# -----------------------------------------------------------------------------
st.markdown("---")
st.subheader("🎯 Painel de Decisão Multiobjetivo")

col_mol, col_sem1, col_sem2 = st.columns([1.2, 1.4, 1.4])

with col_mol:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.markdown(f"**Estrutura Bidimensional (RDKit)**")
    # Renderização da imagem 2D da molécula
    try:
        img = Draw.MolToImage(mol_obj, size=(380, 260))
        st.image(img, use_container_width=True)
    except Exception:
        st.caption("Imagem não disponível para renderização.")
    st.code(canonical_smi, language="text")
    st.markdown("</div>", unsafe_allow_html=True)

with col_sem1:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.markdown("**1. Eficácia Terapêutica ($A_{2A}$)**")
    st.markdown(f'<div class="{a2a_color_class}">{a2a_label}</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    metric_c1, metric_c2 = st.columns(2)
    metric_c1.metric("Potência (pIC50)", f"{pred_pIC50:.2f}")
    if pred_IC50_nM < 1000:
        metric_c2.metric("IC50 Estimado", f"{pred_IC50_nM:.1f} nM")
    else:
        metric_c2.metric("IC50 Estimado", f"{pred_IC50_nM/1000:.2f} µM")

    # Termômetro de Afinidade
    gauge_val = np.clip((pred_pIC50 - 4.0) / (10.0 - 4.0), 0.0, 1.0)
    st.progress(gauge_val)
    st.caption(f"Faixa de Calibração: 4.1 a 11.7 log units | {a2a_desc}")
    st.markdown("</div>", unsafe_allow_html=True)

with col_sem2:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.markdown("**2. Segurança Cardíaca ($K_v11.1$ / hERG)**")
    st.markdown(f'<div class="{herg_color_class}">{herg_label}</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    hmetric_c1, hmetric_c2 = st.columns(2)
    hmetric_c1.metric("Probabilidade P(hERG)", f"{prob_herg:.1%}")
    hmetric_c2.metric("Índice de Segurança", f"{(1.0 - prob_herg):.1%}")

    # Barra de Probabilidade de Risco
    st.progress(prob_herg)
    st.caption(f"Limiares: Verde <30% · Amarelo 30-70% · Vermelho >70% | {herg_desc}")
    st.markdown("</div>", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 11. Propriedades Físico-Químicas & Regra de Lipinski
# -----------------------------------------------------------------------------
desc_dict = compute_2d_descriptors(mol_obj)
mol_wt = desc_dict["MolWt"]
mol_logp = desc_dict["MolLogP"]
tpsa = desc_dict["TPSA"]
hbd = desc_dict["NumHDonors"]
hba = desc_dict["NumHAcceptors"]
rot_bonds = desc_dict["NumRotatableBonds"]

# Violações da Regra de Lipinski
lipinski_violations = sum([
    mol_wt > 500,
    mol_logp > 5.0,
    hbd > 5,
    hba > 10,
])

with st.expander("📋 Propriedades Físico-Químicas 2D e Regra de Lipinski", expanded=False):
    dcols = st.columns(6)
    dcols[0].metric("Massa Molecular (Da)", f"{mol_wt:.2f}")
    dcols[1].metric("MolLogP (Lipofilicidade)", f"{mol_logp:.2f}")
    dcols[2].metric("TPSA (Área Polar Å²)", f"{tpsa:.2f}")
    dcols[3].metric("Doadores de H (HBD)", f"{hbd}")
    dcols[4].metric("Aceptores de H (HBA)", f"{hba}")
    dcols[5].metric("Ligações Rotacionáveis", f"{rot_bonds}")

    st.markdown("---")
    lip_status = "✅ 0 Violações (Alta Drogabilidade)" if lipinski_violations == 0 else f"⚠️ {lipinski_violations} Violação(ões) da Regra dos Cinco"
    st.markdown(f"**Avaliação da Regra dos Cinco (Lipinski):** {lip_status}")


# -----------------------------------------------------------------------------
# 12. Matriz Multiobjetivo 2D de Decisão Farmacológica
# -----------------------------------------------------------------------------
st.markdown("---")
st.subheader("🗺️ Matriz de Decisão Multiobjetivo (Eficácia × Segurança)")

col_quad, col_quad_desc = st.columns([2, 1.2])

with col_quad:
    fig_mat, ax_mat = plt.subplots(figsize=(8, 5.5), facecolor="#0e1117")
    ax_mat.set_facecolor("#161b22")

    x_cur = pred_pIC50
    y_cur = 1.0 - prob_herg  # Segurança cardiológica

    # Faixas de partição
    ax_mat.axvline(6.5, color="#64748b", linestyle="--", linewidth=1.2, alpha=0.7)
    ax_mat.axhline(0.70, color="#64748b", linestyle=":", linewidth=1.2, alpha=0.7)

    # Cores de quadrante suaves
    ax_mat.axvspan(6.5, 11.5, ymin=0.70, ymax=1.0, color="#10b981", alpha=0.12)
    ax_mat.axvspan(6.5, 11.5, ymin=0.0, ymax=0.30, color="#ef4444", alpha=0.12)

    ax_mat.text(9.0, 0.95, "QUADRANTE IDEAL 🏆\nPotente & Seguro", ha="center", va="top", fontsize=9, fontweight="bold", color="#34d399")
    ax_mat.text(9.0, 0.05, "DESCARTE CRÍTICO ⚠️\nPotente porém Cardiotóxico", ha="center", va="bottom", fontsize=9, fontweight="bold", color="#f87171")
    ax_mat.text(4.5, 0.95, "INATIVO & SEGURO\nBaixo Interesse", ha="center", va="top", fontsize=9, fontweight="bold", color="#94a3b8")
    ax_mat.text(4.5, 0.05, "INATIVO & TÓXICO\nReprovado", ha="center", va="bottom", fontsize=9, fontweight="bold", color="#64748b")

    # Ponto da molécula atual
    ax_mat.scatter(x_cur, y_cur, s=280, color="#38bdf8", edgecolors="#ffffff", linewidths=2.5, zorder=6)
    ax_mat.annotate(
        f"{active_name}\n(pIC50={x_cur:.2f} | Seg={y_cur:.1%})",
        (x_cur + 0.2, y_cur + 0.03),
        fontsize=9,
        fontweight="bold",
        color="#ffffff",
        bbox=dict(boxstyle="round,pad=0.3", fc="#1e293b", ec="#38bdf8", alpha=0.9),
    )

    ax_mat.set_xlabel("Afinidade A2A: Potência pIC50 (Maior = Mais Eficaz)", fontsize=10, color="#e2e8f0")
    ax_mat.set_ylabel("Segurança Cardiológica: 1 - P(hERG) (Maior = Mais Seguro)", fontsize=10, color="#e2e8f0")
    ax_mat.set_title("Posicionamento Cartesiano do Composto", fontsize=11, fontweight="bold", color="#f8fafc")
    ax_mat.set_xlim(3.5, 11.5)
    ax_mat.set_ylim(-0.05, 1.05)
    ax_mat.tick_params(colors="#94a3b8")
    ax_mat.grid(True, linestyle=":", alpha=0.3, color="#475569")

    plt.tight_layout()
    st.pyplot(fig_mat)

with col_quad_desc:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.markdown("#### 🧭 Interpretação do Quadrante")

    if x_cur >= 6.5 and y_cur >= 0.70:
        st.success("**🏆 Quadrante 1 — Ideal Terapêutico:** Alta afinidade e perfil de segurança cardiovascular favorável. Prioritário para síntese ou validação em bioensaios.")
    elif x_cur >= 6.5 and y_cur < 0.30:
        st.error("**⚠️ Quadrante 4 — Potente porém Cardiotóxico:** O composto possui alta potência no alvo terapêutico, mas apresenta risco iminente de induzir arritmia letal por bloqueio hERG. Descarte ou redesenho estrutural para atenuar lipofilicidade.")
    elif x_cur < 5.0 and y_cur >= 0.70:
        st.info("**🌿 Quadrante 2 — Inativo & Seguro:** Composto sem riscos de toxicidade cardiovascular, mas com afinidade purinérgica irrelevante.")
    elif x_cur < 5.0 and y_cur < 0.30:
        st.error("**❌ Quadrante 3 — Inativo & Tóxico:** Sem afinidade no receptor terapêutico e com potencial de cardiotoxicidade. Descarte sumário.")
    else:
        st.warning("**🟡 Região de Atenção / Moderada:** Composto com propriedades intermediárias, exigindo refinamento estrutural.")

    st.markdown(
        """
        - **Eixo X:** $pIC_{50}$ predito pelo XGBoost.
        - **Eixo Y:** Grau de segurança ($1 - P_{\\text{hERG}}$) predito pelo Random Forest.
        """
    )
    st.markdown("</div>", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 13. Explicabilidade Local SHAP (XAI - Top 10 Features)
# -----------------------------------------------------------------------------
st.markdown("---")
st.subheader("🔍 Explicabilidade Local com SHAP (Top 10 Atributos Decisivos)")

tab_shap_a2a, tab_shap_herg = st.tabs(["🎯 Explicabilidade da Potência A2A", "❤️ Explicabilidade do Risco hERG"])

with tab_shap_a2a:
    with st.spinner("Calculando atribuição local de Shapley para A2A..."):
        sv_a2a = explainer_a2a(X_sample)
        vals_a2a = sv_a2a.values[0]
        base_a2a = float(sv_a2a.base_values[0])

        top_idx_a2a = np.argsort(np.abs(vals_a2a))[::-1][:10]
        top_names_a2a = [FEATURE_NAMES[i] for i in top_idx_a2a]
        top_impacts_a2a = [vals_a2a[i] for i in top_idx_a2a]
        top_mol_vals_a2a = [X_sample.iloc[0, i] for i in top_idx_a2a]

        labels_a2a = []
        for n, v in zip(top_names_a2a, top_mol_vals_a2a):
            t = "2D" if n in DESCRIPTOR_NAMES else "ECFP4"
            labels_a2a.append(f"{n} ({t}) = {v:.2f}" if isinstance(v, float) else f"{n} ({t}) = {v}")

        fig_sa, ax_sa = plt.subplots(figsize=(9, 4.8), facecolor="#0e1117")
        ax_sa.set_facecolor("#161b22")
        bar_colors_a2a = ["#10b981" if imp >= 0 else "#ef4444" for imp in top_impacts_a2a]

        y_pos = np.arange(len(top_names_a2a))
        ax_sa.barh(y_pos, top_impacts_a2a, color=bar_colors_a2a, edgecolor="white", linewidth=0.5, alpha=0.9)
        ax_sa.set_yticks(y_pos)
        ax_sa.set_yticklabels(labels_a2a, fontsize=9, color="#e2e8f0")
        ax_sa.invert_yaxis()
        ax_sa.axvline(0, color="#94a3b8", linestyle="-", linewidth=1.0)
        ax_sa.set_xlabel("Impacto SHAP no pIC50 (Verde = Aumenta Potência | Vermelho = Reduz)", fontsize=9, color="#e2e8f0")
        ax_sa.set_title(f"Top 10 Features para Afinidade A2A (Base Esperada = {base_a2a:.2f})", fontsize=11, fontweight="bold", color="#f8fafc")
        ax_sa.tick_params(colors="#94a3b8")
        ax_sa.grid(axis="x", linestyle=":", alpha=0.3, color="#475569")
        plt.tight_layout()
        st.pyplot(fig_sa)

        # Tabela explicativa detalhada
        df_expl_a2a = pd.DataFrame({
            "Atributo": top_names_a2a,
            "Tipo": ["Descritor Macroscópico 2D" if n in DESCRIPTOR_NAMES else "Subestrutura Topológica ECFP4" for n in top_names_a2a],
            "Valor na Molécula": [f"{v:.2f}" if isinstance(v, float) else f"{v}" for v in top_mol_vals_a2a],
            "Impacto SHAP no pIC50": [f"{imp:+.3f}" for imp in top_impacts_a2a],
            "Efeito Biológico": ["Aumenta afinidade no receptor" if imp >= 0 else "Atenua afinidade no receptor" for imp in top_impacts_a2a],
        })
        st.dataframe(df_expl_a2a, use_container_width=True)

with tab_shap_herg:
    with st.spinner("Calculando atribuição local de Shapley para hERG..."):
        sv_herg = explainer_herg(X_sample)
        vals_herg = sv_herg.values[0, :, 1]  # Classe 1 = Bloqueador cardiotóxico
        base_herg = float(sv_herg.base_values[0, 1])

        top_idx_herg = np.argsort(np.abs(vals_herg))[::-1][:10]
        top_names_herg = [FEATURE_NAMES[i] for i in top_idx_herg]
        top_impacts_herg = [vals_herg[i] for i in top_idx_herg]
        top_mol_vals_herg = [X_sample.iloc[0, i] for i in top_idx_herg]

        labels_herg = []
        for n, v in zip(top_names_herg, top_mol_vals_herg):
            t = "2D" if n in DESCRIPTOR_NAMES else "ECFP4"
            labels_herg.append(f"{n} ({t}) = {v:.2f}" if isinstance(v, float) else f"{n} ({t}) = {v}")

        fig_sh, ax_sh = plt.subplots(figsize=(9, 4.8), facecolor="#0e1117")
        ax_sh.set_facecolor("#161b22")
        bar_colors_herg = ["#ef4444" if imp >= 0 else "#10b981" for imp in top_impacts_herg]

        y_pos = np.arange(len(top_names_herg))
        ax_sh.barh(y_pos, top_impacts_herg, color=bar_colors_herg, edgecolor="white", linewidth=0.5, alpha=0.9)
        ax_sh.set_yticks(y_pos)
        ax_sh.set_yticklabels(labels_herg, fontsize=9, color="#e2e8f0")
        ax_sh.invert_yaxis()
        ax_sh.axvline(0, color="#94a3b8", linestyle="-", linewidth=1.0)
        ax_sh.set_xlabel("Impacto SHAP no Risco hERG (Vermelho = Aumenta Toxicidade | Verde = Protege)", fontsize=9, color="#e2e8f0")
        ax_sh.set_title(f"Top 10 Features para Cardiotoxicidade hERG (Base Esperada = {base_herg:.2%})", fontsize=11, fontweight="bold", color="#f8fafc")
        ax_sh.tick_params(colors="#94a3b8")
        ax_sh.grid(axis="x", linestyle=":", alpha=0.3, color="#475569")
        plt.tight_layout()
        st.pyplot(fig_sh)

        df_expl_herg = pd.DataFrame({
            "Atributo": top_names_herg,
            "Tipo": ["Descritor Macroscópico 2D" if n in DESCRIPTOR_NAMES else "Subestrutura Topológica ECFP4" for n in top_names_herg],
            "Valor na Molécula": [f"{v:.2f}" if isinstance(v, float) else f"{v}" for v in top_mol_vals_herg],
            "Impacto SHAP em P(hERG)": [f"{imp:+.3f}" for imp in top_impacts_herg],
            "Efeito Clínico": ["Aumenta propensão ao bloqueio (cardiotóxico)" if imp >= 0 else "Atenua o risco de bloqueio (protetor)" for imp in top_impacts_herg],
        })
        st.dataframe(df_expl_herg, use_container_width=True)


# -----------------------------------------------------------------------------
# 14. Rodapé e Governança Técnica
# -----------------------------------------------------------------------------
st.markdown("---")
footer_c1, footer_c2 = st.columns([2, 1])
footer_c1.caption(
    "**MultiTarget BioPredict** — Plataforma Acadêmica de Triagem Virtual Multiobjetivo. "
    "Desenvolvido por Raphael Pires. Licença MIT. RDKit · scikit-learn · XGBoost · SHAP · Streamlit."
)
footer_c2.caption(
    "Ambiente 100% Local (Antigravity) · Restrições de 8 GB RAM respeitadas · Sem dependência de nuvem."
)
