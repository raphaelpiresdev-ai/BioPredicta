"""
MultiTarget BioPredict — Aplicação Streamlit (placeholder da Fase 0).

Execute com:
    streamlit run app.py

Nesta fase o app apenas valida a integração ponta a ponta do ambiente
(Streamlit + RDKit + src.chem_utils). Os modelos A2A/hERG serão conectados
nas fases seguintes.
"""

import streamlit as st

from src.chem_utils import canonicalize_smiles, compute_2d_descriptors, compute_morgan_fingerprints

st.set_page_config(page_title="MultiTarget BioPredict", page_icon="🧬", layout="wide")

st.title("🧬 MultiTarget BioPredict")
st.caption("Triagem Virtual Multiobjetivo · Afinidade A2A × Cardiotoxicidade hERG")

st.info(
    "**Fase 0 — Preparação do Ambiente concluída.** "
    "Os modelos preditivos serão integrados nas próximas fases.",
    icon="🚧",
)

st.subheader("Verificação rápida do pipeline químico")
smiles = st.text_input(
    "SMILES",
    value="CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # cafeína (antagonista A2A não seletivo)
    help="Insira um SMILES; ele será padronizado e featurizado com RDKit.",
)

if smiles:
    canonical = canonicalize_smiles(smiles)
    if canonical is None:
        st.error("SMILES inválido.")
    else:
        st.success(f"SMILES canônico: `{canonical}`")
        desc = compute_2d_descriptors(canonical)
        cols = st.columns(len(desc))
        for col, (name, value) in zip(cols, desc.items()):
            col.metric(name, f"{value:.2f}" if isinstance(value, float) else value)
        fp = compute_morgan_fingerprints(canonical)
        st.caption(f"ECFP4: {fp.size} bits · {int(fp.sum())} bits ativos")
