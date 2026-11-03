"""
Módulo da Página: Nova Análise Molecular (Etapa 2 — UX/UI Refactoring).

Responsável por:
1. Ingestão molecular unificada via inserção de SMILES ou desenho no editor Ketcher;
2. Validação estrutural em tempo real e canonicalização isomérica com RDKit;
3. Pré-visualização 2D em Dark Mode de alta fidelidade;
4. Identificação automatizada via PubChem (PUG REST API) com tratamento resiliente (404, offline, timeout);
5. Acionamento explícito da análise multiobjetivo (1030 atributos, XGBoost A2A, Random Forest hERG, AD e SHAP);
6. Armazenamento consistente do resultado em st.session_state["current_analysis"];
7. Garantia de integridade do estado e invalidação de resultados desatualizados.
"""

from __future__ import annotations

import datetime
import html
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
import shap
import streamlit as st
from streamlit.runtime.scriptrunner_utils.script_run_context import get_script_run_ctx
from streamlit_ketcher import st_ketcher

from src.ad_utils import compute_ad_diagnostics, get_ad_reference
from src.chem_utils import FEATURE_NAMES, canonicalize_smiles, compute_2d_descriptors, featurize_dataframe
from src.pubchem_utils import (
    STATUS_CONNECTION_ERROR,
    STATUS_INVALID_STRUCTURE,
    STATUS_NOT_FOUND,
    STATUS_SUCCESS,
    build_pubchem_card_html,
    identify_compound_pubchem,
)
from src.ui.components import (
    get_page_target,
    render_html,
    render_molecule_dark_base64,
    render_page_header,
)
from src.ui.history_manager import (
    MAX_HISTORY_ITEMS,
    add_to_history,
    get_history,
    get_history_summary_list,
    init_history,
    set_active_molecule,
)

logger = logging.getLogger(__name__)

# Moléculas exemplares de referência farmacológica para carregamento rápido
PRESET_REFERENCE_COMPOUNDS: Dict[str, Dict[str, str]] = {
    "☕ Cafeína (Controle A2A)": {
        "smiles": "Cn1c(=O)c2c(ncn2C)n(C)c1=O",
        "description": "Antagonista clássico purinérgico A1/A2A com potência micromolar moderada.",
        "category": "Controle Farmacológico",
    },
    "💊 Istradefilina (Antagonista A2A)": {
        "smiles": "CCn1c(=O)c2c(nc(/C=C/c3ccc(OC)c(OC)c3)n2C)n(CC)c1=O",
        "description": "Fármaco aprovado (Nourianz) com alta afinidade e seletividade pelo receptor A2A.",
        "category": "Antagonista Clínico",
    },
    "🫀 Haloperidol (Alto Risco hERG)": {
        "smiles": "OC1(CCN(CCCC(=O)c2ccc(F)cc2)CC1)c1ccc(Cl)cc1",
        "description": "Antipsicótico com alta probabilidade de bloqueio do canal de potássio hERG (Kv11.1).",
        "category": "Controle Cardiotóxico",
    },
    "🌿 Taurina (Controle Negativo)": {
        "smiles": "NCCS(=O)(=O)O",
        "description": "Metabólito osmorregulador endógeno sem afinidade por A2A nem cardiotoxicidade hERG.",
        "category": "Controle Negativo",
    },
    "🧪 2-Fenilbenzoxazol (Inédito/Sem PubChem)": {
        "smiles": "c1ccc(cc1)c2nc3ccccc3o2",
        "description": "Esqueleto heteroaromático simples para verificação de composto personalizado e 404 PubChem.",
        "category": "Composto Sintético",
    },
}


def select_preset_molecule(smiles: str) -> None:
    """Sincroniza os estados de sessão para carregar uma molécula de acesso rápido no campo SMILES."""
    st.session_state["input_smiles_field"] = smiles
    st.session_state["field_smiles_text"] = smiles
    st.session_state["input_mode_selector"] = "✏️ Inserir SMILES"
    st.session_state.pop("_pending_smiles_update", None)
    st.session_state.pop("ketcher_molecular_canvas", None)


# -----------------------------------------------------------------------------
# 1. Carregador Resiliente Local de Modelos e Explicadores
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_models_bundle_fallback() -> Dict[str, Any]:
    """Carrega modelos, metadados e explainers SHAP caso não tenham sido injetados pelo app.py."""
    root = Path(__file__).resolve().parent.parent.parent.parent
    model_a2a = joblib.load(root / "models" / "a2a_model.joblib")
    model_herg = joblib.load(root / "models" / "herg_model.joblib")

    with open(root / "models" / "a2a_model_metadata.json", "r", encoding="utf-8") as f:
        meta_a2a = json.load(f)

    with open(root / "models" / "herg_model_metadata.json", "r", encoding="utf-8") as f:
        meta_herg = json.load(f)

    explainer_a2a = shap.TreeExplainer(model_a2a)
    explainer_herg = shap.TreeExplainer(model_herg)

    return {
        "model_a2a": model_a2a,
        "model_herg": model_herg,
        "meta_a2a": meta_a2a,
        "meta_herg": meta_herg,
        "explainer_a2a": explainer_a2a,
        "explainer_herg": explainer_herg,
        "load_ad_reference": get_ad_reference,
    }


def classify_herg_risk_canonical(prob_herg: float) -> Dict[str, str]:
    """
    Classificação canônica de risco operacional hERG:
    - Baixo: P < 0.30
    - Moderado / Intermediário: 0.30 <= P <= 0.70 (0.70 pertence à classe moderada)
    - Alto: P > 0.70
    """
    if prob_herg < 0.30:
        return {
            "risk_class": "baixo",
            "color_class": "semaphore-green",
            "bar_color": "#00D2BE",
            "label": "🟢 Baixo Risco Operacional (P_hERG < 0,30)",
            "desc": "Escore estimado de probabilidade de inibição hERG inferior a 0,30.",
        }
    elif prob_herg <= 0.70:
        return {
            "risk_class": "intermediario",
            "color_class": "semaphore-yellow",
            "bar_color": "#F59E0B",
            "label": "🟡 Risco Intermediário / Moderado (0,30 ≤ P_hERG ≤ 0,70)",
            "desc": "Escore estimado de probabilidade na faixa operacional intermediária/moderada (0,30 ≤ P_hERG ≤ 0,70).",
        }
    else:
        return {
            "risk_class": "alto",
            "color_class": "semaphore-red",
            "bar_color": "#FF4B4B",
            "label": "🔴 Alto Risco Operacional (P_hERG > 0,70)",
            "desc": "Escore estimado de probabilidade superior ao limiar de alto risco (P_hERG > 0,70).",
        }


def classify_a2a_potency_canonical(pred_pIC50: float) -> Dict[str, str]:
    """Classificação canônica de potência A2A conforme thresholds canônicos (5.0 e 6.5)."""
    if pred_pIC50 >= 6.5:
        return {
            "potency_class": "alta",
            "color_class": "semaphore-green",
            "label": "🟢 Alta Potência Operacional (pIC50 ≥ 6,5)",
            "desc": "Faixa nanomolar compatível com alta afinidade pelo receptor A2A.",
        }
    elif pred_pIC50 >= 5.0:
        return {
            "potency_class": "moderada",
            "color_class": "semaphore-yellow",
            "label": "🟡 Potência Moderada Operacional (5,0 ≤ pIC50 < 6,5)",
            "desc": "Faixa micromolar compatível com potência intermediária.",
        }
    else:
        return {
            "potency_class": "baixa",
            "color_class": "semaphore-red",
            "label": "🔴 Baixa Potência / Inativo (pIC50 < 5,0)",
            "desc": "Predição abaixo do limiar operacional de potência micromolar.",
        }


def classify_quadrant_canonical(x_cur: float, y_cur: float, atol: float = 1e-12) -> Dict[str, str]:
    """Classifica o posicionamento no espaço cartesiano (X: pIC50, Y: 1 - P_hERG).

    Garante que y = 0.30 (P_hERG = 0.70) e y = 0.70 (P_hERG = 0.30) pertençam
    à classe moderada/intermediária (0.30 <= P_hERG <= 0.70), com tolerância IEEE 754.
    """
    if y_cur > 0.70 and not bool(np.isclose(y_cur, 0.70, rtol=0.0, atol=atol)):
        herg_y = "baixo"
    elif y_cur < 0.30 and not bool(np.isclose(y_cur, 0.30, rtol=0.0, atol=atol)):
        herg_y = "alto"
    else:
        herg_y = "intermediario"

    if x_cur >= 6.5 and herg_y == "baixo":
        return {
            "quad_badge_cls": "semaphore-green",
            "quad_title": "🏆 Quadrante 1 — Zona Ideal Operacional",
            "quad_desc": "Alta potência estimada em A2A (pIC50 ≥ 6,5) combinada a baixo escore hERG (P < 0,30).",
        }
    elif x_cur >= 6.5 and herg_y == "alto":
        return {
            "quad_badge_cls": "semaphore-red",
            "quad_title": "⚠️ Quadrante 4 — Potente porém Alto Risco hERG",
            "quad_desc": "Alta potência em A2A, mas escore elevado para bloqueio hERG (P > 0,70).",
        }
    elif x_cur < 5.0 and herg_y == "baixo":
        return {
            "quad_badge_cls": "semaphore-yellow",
            "quad_title": "🌿 Quadrante 2 — Baixa Potência & Baixo Risco hERG",
            "quad_desc": "Baixo escore de risco hERG (P < 0,30), porém sem potência relevante em A2A (pIC50 < 5,0).",
        }
    elif x_cur < 5.0 and herg_y == "alto":
        return {
            "quad_badge_cls": "semaphore-red",
            "quad_title": "❌ Quadrante 3 — Baixa Potência & Alto Risco hERG",
            "quad_desc": "Inativo em A2A associado a alto risco de bloqueio hERG (P > 0,70; descarte imediato).",
        }
    else:
        return {
            "quad_badge_cls": "semaphore-yellow",
            "quad_title": "🟡 Zona de Otimização / Hit-to-Lead",
            "quad_desc": "Perfil intermediário de potência (5,0 ≤ pIC50 < 6,5) e/ou segurança (0,30 ≤ P ≤ 0,70); requer otimização estrutural guiada.",
        }


# -----------------------------------------------------------------------------
# 2. Execução da Análise Molecular Completa (Pipeline Unificado)
# -----------------------------------------------------------------------------
def execute_molecular_pipeline(
    canonical_smi: str,
    mol: Chem.Mol,
    input_method: str,
    raw_input: str,
    models_bundle: Dict[str, Any],
    provenance_info: Optional[Dict[str, Any]] = None,
    pubchem_info: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Executa a featurização 1030-D, as predições dos modelos XGBoost e Random Forest,
    o diagnóstico de Domínio de Aplicabilidade e a decomposição explicativa SHAP.
    Retorna o dicionário completo estruturado para st.session_state["current_analysis"].
    """
    model_a2a = models_bundle["model_a2a"]
    model_herg = models_bundle["model_herg"]
    meta_a2a = models_bundle.get("meta_a2a", {})
    meta_herg = models_bundle.get("meta_herg", {})
    explainer_a2a = models_bundle["explainer_a2a"]
    explainer_herg = models_bundle["explainer_herg"]
    load_ad_ref = models_bundle.get("load_ad_reference", get_ad_reference)

    # 1. Featurização 1030-D (6 descritores RDKit + 1024 bits Morgan ECFP4)
    df_single = pd.DataFrame([{"canonical_smiles": canonical_smi}])
    df_feat = featurize_dataframe(df_single, "canonical_smiles")
    X_sample = df_feat[list(FEATURE_NAMES)]

    # 2. Predição A2A (XGBoost Regressor)
    pred_pIC50 = float(model_a2a.predict(X_sample)[0])
    pred_IC50_nM = float(10 ** (9.0 - pred_pIC50))
    a2a_class = classify_a2a_potency_canonical(pred_pIC50)

    # 3. Predição hERG (Random Forest Classifier)
    prob_herg = float(model_herg.predict_proba(X_sample)[0, 1])
    safety_index = float(1.0 - prob_herg)

    herg_class = classify_herg_risk_canonical(prob_herg)

    # 4. Classificação Multiobjetivo
    quadrant_info = classify_quadrant_canonical(pred_pIC50, safety_index)

    # 5. Domínio de Aplicabilidade (AD)
    try:
        ref_a2a = load_ad_ref("A2A")
        diag_ad_a2a = compute_ad_diagnostics(mol, "A2A", reference=ref_a2a)
    except Exception as exc:
        diag_ad_a2a = {
            "status": "technical_error",
            "endpoint": "A2A",
            "error_message": f"Falha no diagnóstico AD: {exc}",
        }

    try:
        ref_herg = load_ad_ref("hERG")
        diag_ad_herg = compute_ad_diagnostics(mol, "hERG", reference=ref_herg)
    except Exception as exc:
        diag_ad_herg = {
            "status": "technical_error",
            "endpoint": "hERG",
            "error_message": f"Falha no diagnóstico AD: {exc}",
        }

    # 6. Decomposição SHAP (TreeExplainer)
    try:
        shap_raw_a2a = explainer_a2a(X_sample)
        shap_values_a2a = shap_raw_a2a.values[0]
        base_value_a2a = float(shap_raw_a2a.base_values[0])
    except Exception as exc:
        logger.warning(f"Erro ao calcular SHAP A2A: {exc}")
        shap_values_a2a = np.zeros(len(FEATURE_NAMES))
        base_value_a2a = 0.0

    try:
        shap_raw_herg = explainer_herg(X_sample)
        # Random Forest binário no SHAP pode retornar array tridimensional (amostras, features, classes)
        if len(shap_raw_herg.values.shape) == 3:
            shap_values_herg = shap_raw_herg.values[0, :, 1]
            base_value_herg = float(shap_raw_herg.base_values[0, 1])
        else:
            shap_values_herg = shap_raw_herg.values[0]
            base_value_herg = float(shap_raw_herg.base_values[0])
    except Exception as exc:
        logger.warning(f"Erro ao calcular SHAP hERG: {exc}")
        shap_values_herg = np.zeros(len(FEATURE_NAMES))
        base_value_herg = 0.0

    # 7. Descritores físico-químicos e fórmula
    desc_dict = compute_2d_descriptors(mol)
    formula = rdMolDescriptors.CalcMolFormula(mol)
    mol_wt = float(desc_dict.get("MolWt", Descriptors.MolWt(mol)))

    analysis_result = {
        "timestamp": datetime.datetime.now().isoformat(),
        "input_method": input_method,
        "raw_input": raw_input,
        "canonical_smiles": canonical_smi,
        "mol": mol,
        "formula": formula,
        "mol_wt": mol_wt,
        "descriptors": desc_dict,
        "provenance": provenance_info or {},
        "pubchem": pubchem_info or {},
        "features": X_sample,
        "a2a": {
            "pred_pIC50": pred_pIC50,
            "pred_IC50_nM": pred_IC50_nM,
            "classification": a2a_class,
            "ad": diag_ad_a2a,
            "shap_values": shap_values_a2a,
            "base_value": base_value_a2a,
            "model_metadata": meta_a2a,
        },
        "herg": {
            "prob_herg": prob_herg,
            "safety_index": safety_index,
            "complementary_index": safety_index,
            "classification": herg_class,
            "ad": diag_ad_herg,
            "shap_values": shap_values_herg,
            "base_value": base_value_herg,
            "model_metadata": meta_herg,
        },
        "multiobjective": {
            "quadrant": quadrant_info,
            "x": pred_pIC50,
            "y": safety_index,
        },
    }

    return analysis_result


# -----------------------------------------------------------------------------
# 3. Componente de Resumo Executivo Pós-Análise (Feedback Imediato)
# -----------------------------------------------------------------------------
def render_analysis_executive_banner(analysis: Dict[str, Any]) -> None:
    """Renderiza os cartões de resumo da análise recém-concluída e link para Resultados."""
    a2a = analysis["a2a"]
    herg = analysis["herg"]
    quad = analysis["multiobjective"]["quadrant"]
    ad_a2a = a2a["ad"]
    ad_herg = herg["ad"]

    pIC50 = a2a["pred_pIC50"]
    ic50_nm = a2a["pred_IC50_nM"]
    ic50_str = f"{ic50_nm:.1f} nM" if ic50_nm < 1000 else f"{ic50_nm / 1000:.2f} µM"

    prob_herg = herg["prob_herg"]
    safety_idx = herg["safety_index"]

    ad_a2a_smax = ad_a2a.get("s_max_all")
    ad_herg_smax = ad_herg.get("s_max_all")

    ad_a2a_badge = f"S_max {ad_a2a_smax:.1%}" if ad_a2a_smax is not None else "N/A"
    ad_herg_badge = f"S_max {ad_herg_smax:.1%}" if ad_herg_smax is not None else "N/A"

    quad_title_esc = html.escape(str(quad.get("quad_title", "Posicionamento Cartesiano")))
    quad_desc_esc = html.escape(
        str(quad.get("quad_desc", "Avaliação concorrente de afinidade purinérgica e segurança cardíaca."))
    )

    render_html(
        f"""
        <div class="metric-card" style="border-left: 4px solid #00D2BE; margin-top: 1.5rem; margin-bottom: 1.2rem;">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 0.5rem; margin-bottom: 0.8rem;">
                <div style="display: flex; align-items: center; gap: 8px;">
                    <span style="font-size: 1.25rem;">✅</span>
                    <span style="font-size: 1.08rem; font-weight: 700; color: #FFFFFF;">
                        Análise Multiobjetivo Concluída com Sucesso
                    </span>
                </div>
                <span class="{quad.get('quad_badge_cls', 'semaphore-green')}" style="padding: 4px 14px; font-size: 0.82rem; font-weight: 700;">
                    {quad_title_esc}
                </span>
            </div>
            <p style="font-size: 0.84rem; color: #CBD5E1; margin: 0 0 1rem 0; line-height: 1.45;">
                {quad_desc_esc}
            </p>
            <div class="kpi-grid" style="grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));">
                <div class="kpi-box">
                    <div class="kpi-label">Potência A2A (pIC50)</div>
                    <div class="kpi-value kpi-value-accent">{pIC50:.2f}</div>
                    <div style="font-size: 0.78rem; color: #94A3B8; margin-top: 0.2rem;">IC50: {ic50_str}</div>
                    <div style="font-size: 0.74rem; color: #38BDF8; margin-top: 0.25rem;">AD A2A: {ad_a2a_badge}</div>
                </div>
                <div class="kpi-box">
                    <div class="kpi-label">Escore hERG P(Kv11.1)</div>
                    <div class="kpi-value" style="color: {herg['classification'].get('bar_color', '#00D2BE')};">{prob_herg:.1%}</div>
                    <div style="font-size: 0.78rem; color: #94A3B8; margin-top: 0.2rem;" title="Derivação matemática complementar (1.0 - P_hERG) utilizada como coordenada Y no espaço multiobjetivo cartesiano; não representa probabilidade clínica direta nem garantia de segurança.">Índice Complementar (1 - P): {safety_idx:.1%} ℹ️</div>
                    <div style="font-size: 0.74rem; color: #38BDF8; margin-top: 0.25rem;">AD hERG: {ad_herg_badge}</div>
                </div>
                <div class="kpi-box">
                    <div class="kpi-label">Fórmula &amp; Massa Molecular</div>
                    <div class="kpi-value" style="font-size: 1.15rem; color: #FFFFFF;">{html.escape(str(analysis['formula']))}</div>
                    <div style="font-size: 0.78rem; color: #94A3B8; margin-top: 0.2rem;">MolWt: {analysis['mol_wt']:.1f} Da</div>
                    <div style="font-size: 0.74rem; color: #00D2BE; margin-top: 0.25rem;">1030 atributos gerados</div>
                </div>
            </div>
        </div>
        """
    )

    col_btn1, col_btn2 = st.columns([1.8, 1.2])
    with col_btn1:
        st.page_link(
            get_page_target("resultados"),
            label="👉 Abrir Painel Completo de Resultados (Matriz 2D, SHAP e Diagnósticos)",
            icon="📊",
        )
    with col_btn2:
        st.caption(
            "Os dados completos foram salvos na sessão (`st.session_state['current_analysis']`)."
        )


# -----------------------------------------------------------------------------
# 3.1. Componente de Histórico de Moléculas Analisadas na Sessão
# -----------------------------------------------------------------------------
def render_session_history_section() -> None:
    """Renderiza a seção de histórico de moléculas avaliadas na sessão corrente."""
    summaries = get_history_summary_list()
    if not summaries:
        return

    render_html("<div style='margin-top:1.8rem;'></div>")
    st.markdown("---")
    render_html(
        f"""
        <div style="margin-bottom:0.8rem;">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <h3 style="margin:0; font-size:1.15rem; color:#FFFFFF; font-weight:700;">
                    📚 Histórico de Análises na Sessão ({len(summaries)}/{MAX_HISTORY_ITEMS})
                </h3>
            </div>
            <p style="font-size:0.80rem; color:#94A3B8; margin:0.25rem 0 0 0;">
                Moléculas avaliadas durante a sessão atual (política FIFO, limite de {MAX_HISTORY_ITEMS} registros).
                Clique em <b>🔍 Ativar no Painel</b> para recuperar os resultados completos em <i>Resultados</i> e <i>Interpretação</i>.
            </p>
        </div>
        """
    )

    for item in summaries:
        smi = item["canonical_smiles"]
        is_act = item["is_active"]
        idx = item["index"]
        name_esc = html.escape(str(item["display_name"]))
        pIC50 = item["pIC50"]
        prob_h = item["prob_herg"]
        quad_esc = html.escape(str(item["quadrant_title"]))
        a2a_lbl_esc = html.escape(str(item["a2a_class_label"]))
        herg_lbl_esc = html.escape(str(item["herg_class_label"]))
        formula_esc = html.escape(str(item["formula"]))
        smi_esc = html.escape(str(smi))
        active_border = "border: 1px solid #00D2BE; background: rgba(0,210,190,0.04);" if is_act else "border: 1px solid #333B4D;"

        col_info, col_btn = st.columns([4.2, 1.3])
        with col_info:
            active_pill = (
                '<span style="background:rgba(0,210,190,0.2); color:#00D2BE; font-size:0.72rem; '
                'padding:2px 8px; border-radius:10px; font-weight:700; margin-left:8px;">✓ Ativa no Painel</span>'
                if is_act
                else ""
            )
            render_html(
                f"""
                <div class="metric-card" style="{active_border} padding: 0.75rem 1rem; margin-bottom: 0.5rem;">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <span style="color:#FFFFFF; font-weight:700; font-size:0.92rem;">
                            #{idx} — {name_esc} {active_pill}
                        </span>
                        <span style="font-size:0.77rem; color:#CBD5E1;">
                            {quad_esc}
                        </span>
                    </div>
                    <div style="display:flex; gap:16px; font-size:0.78rem; color:#94A3B8; margin-top:0.3rem; flex-wrap:wrap;">
                        <span>A2A pIC₅₀: <strong style="color:#FFFFFF;">{pIC50:.2f}</strong> ({a2a_lbl_esc})</span>
                        <span>hERG Escore: <strong style="color:#FFFFFF;">{prob_h:.2f}</strong> ({herg_lbl_esc})</span>
                        <span>Fórmula: <strong style="color:#CBD5E1;">{formula_esc}</strong> ({item["mol_wt"]:.1f} Da)</span>
                    </div>
                    <div style="font-size:0.72rem; font-family:'JetBrains Mono',monospace; color:#64748B; margin-top:0.25rem; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;">
                        {smi_esc}
                    </div>
                </div>
                """
            )
        with col_btn:
            render_html("<div style='height:8px;'></div>")
            if not is_act:
                btn_key = f"btn_act_hist_{idx}_{abs(hash(smi)) % 100000}"
                if st.button(
                    "🔍 Ativar no Painel",
                    key=btn_key,
                    width="stretch",
                    help="Define esta molécula como ativa no Painel de Resultados e Interpretação.",
                ):
                    set_active_molecule(smi)
                    st.session_state["input_smiles_field"] = smi
                    st.session_state["_pending_smiles_update"] = smi
                    st.rerun()
            else:
                render_html(
                    "<div style='text-align:center; padding-top:14px; font-size:0.80rem; color:#00D2BE; font-weight:600;'>Visualizando</div>"
                )


# -----------------------------------------------------------------------------
# 4. Função Principal de Renderização da Página
# -----------------------------------------------------------------------------
def render_nova_analise(models_bundle: Optional[Dict[str, Any]] = None) -> None:
    """Renderiza a experiência completa de entrada molecular e disparo da predição."""
    # Garante a disponibilidade dos modelos de inferência
    if models_bundle is None:
        try:
            models_bundle = load_models_bundle_fallback()
        except Exception as exc:
            st.error(f"❌ Erro ao inicializar recursos de Machine Learning: {exc}")
            return

    # Cabeçalho padronizado da página
    render_page_header(
        title="Nova Análise Molecular",
        subtitle="Insira uma estrutura química por SMILES ou desenhe sua molécula para iniciar a avaliação multiobjetivo.",
        icon="🧪",
    )

    init_history()

    # Inicialização e sincronização atômica de variáveis de sessão ANTES da instanciação de widgets
    if "input_smiles_field" not in st.session_state:
        st.session_state["input_smiles_field"] = ""
    if "current_analysis" not in st.session_state:
        st.session_state["current_analysis"] = None
    if "current_analysis_smiles" not in st.session_state:
        st.session_state["current_analysis_smiles"] = None
    if not st.session_state.get("input_mode_selector"):
        st.session_state["input_mode_selector"] = "✏️ Inserir SMILES"

    if "_pending_mode_update" in st.session_state:
        st.session_state["input_mode_selector"] = st.session_state.pop("_pending_mode_update")

    if "_pending_smiles_update" in st.session_state:
        pending_smi = st.session_state.pop("_pending_smiles_update")
        st.session_state["input_smiles_field"] = pending_smi
        st.session_state["field_smiles_text"] = pending_smi
        st.session_state.pop("ketcher_molecular_canvas", None)
    elif "field_smiles_text" not in st.session_state:
        st.session_state["field_smiles_text"] = st.session_state.get("input_smiles_field", "")

    # Identificação da molécula atualmente ativa na sessão
    active_analysis = st.session_state.get("current_analysis")
    active_smi = st.session_state.get("current_analysis_smiles")
    if active_analysis and active_smi:
        pubchem_act = active_analysis.get("pubchem") or {}
        prov_act = active_analysis.get("provenance") or {}
        act_name = (
            pubchem_act.get("title")
            or prov_act.get("compound_name")
            or prov_act.get("name")
            or active_smi
        )
        act_name_esc = html.escape(str(act_name))
        pIC50_act = active_analysis.get("a2a", {}).get("pred_pIC50", 0.0)
        prob_h_act = active_analysis.get("herg", {}).get("prob_herg", 0.0)
        render_html(
            f"""
            <div style="background:rgba(0,210,190,0.06); border:1px solid rgba(0,210,190,0.3); border-radius:8px; padding:8px 14px; margin-bottom:12px; display:flex; justify-content:space-between; align-items:center;">
                <span style="font-size:0.83rem; color:#E2E8F0;">
                    🔬 <b>Molécula Ativa no Painel de Resultados:</b> <span style="color:#00D2BE; font-weight:700;">{act_name_esc}</span>
                    <span style="color:#94A3B8; font-size:0.77rem; margin-left:8px;">(pIC₅₀: {pIC50_act:.2f} | Escore hERG: {prob_h_act:.2f})</span>
                </span>
                <span style="font-size:0.74rem; background:rgba(0,210,190,0.2); color:#00D2BE; border-radius:12px; padding:2px 8px; font-weight:600;">
                    ✓ Carregada
                </span>
            </div>
            """
        )

    # -------------------------------------------------------------------------
    # A. Carregamento Rápido de Controles e Moléculas Exemplares (Atalhos Didáticos)
    # -------------------------------------------------------------------------
    with st.expander("💡 Carregar Molécula Modelo ou Controle Farmacológico (Acesso Rápido)", expanded=False):
        render_html(
            "<span style='font-size:0.83rem; color:#94A3B8;'>"
            "Clique em um dos controles de referência abaixo para carregar instantaneamente sua estrutura "
            "e testar o pipeline sem necessidade de copiar códigos SMILES:"
            "</span>"
        )

        preset_cols = st.columns(len(PRESET_REFERENCE_COMPOUNDS))
        for idx, (label, meta) in enumerate(PRESET_REFERENCE_COMPOUNDS.items()):
            with preset_cols[idx]:
                if st.button(label, key=f"btn_preset_{idx}", width="stretch"):
                    select_preset_molecule(meta["smiles"])
                    st.rerun()

    # -------------------------------------------------------------------------
    # B. Seletor de Modalidade de Entrada (Segmented Control)
    # -------------------------------------------------------------------------
    _has_live_ctx = get_script_run_ctx() is not None
    render_html("<div style='margin-top:0.8rem;'></div>")

    _seg_kwargs: Dict[str, Any] = {
        "options": ["✏️ Inserir SMILES", "🎨 Desenhar Molécula (Ketcher)"],
        "key": "input_mode_selector",
    }
    if not (_has_live_ctx and "input_mode_selector" in st.session_state):
        _seg_kwargs["default"] = st.session_state.get("input_mode_selector", "✏️ Inserir SMILES")

    selected_mode = st.segmented_control(
        "Selecione o Método de Entrada:",
        **_seg_kwargs,
    )
    if not selected_mode:
        selected_mode = "✏️ Inserir SMILES"

    raw_input_structure = ""

    # -------------------------------------------------------------------------
    # Modalidade 1: Inserção de SMILES Manual
    # -------------------------------------------------------------------------
    if selected_mode == "✏️ Inserir SMILES":
        if "field_smiles_text" not in st.session_state:
            st.session_state["field_smiles_text"] = st.session_state.get("input_smiles_field", "")

        col_input, col_clear = st.columns([4.2, 0.8])
        with col_input:
            _txt_kwargs: Dict[str, Any] = {
                "placeholder": "Exemplo: Cn1c(=O)c2c(ncn2C)n(C)c1=O",
                "help": "Insira a representação linear SMILES simplificada ou isomérica.",
                "key": "field_smiles_text",
            }
            if not (_has_live_ctx and "field_smiles_text" in st.session_state):
                _txt_kwargs["value"] = st.session_state.get("field_smiles_text", "")

            input_val = st.text_input(
                "Código SMILES da Estrutura:",
                **_txt_kwargs,
            )
            raw_input_structure = (input_val or "").strip()
            if raw_input_structure != st.session_state.get("input_smiles_field", ""):
                st.session_state["input_smiles_field"] = raw_input_structure

        with col_clear:
            render_html("<div style='margin-top:1.75rem;'></div>")
            if st.button("🗑️ Limpar", key="btn_clear_smiles", width="stretch", help="Limpar o campo de entrada"):
                st.session_state["input_smiles_field"] = ""
                st.session_state["_pending_smiles_update"] = ""
                st.rerun()

    # -------------------------------------------------------------------------
    # Modalidade 2: Desenho Molecular no Ketcher
    # -------------------------------------------------------------------------
    else:
        render_html(
            "<span style='font-size:0.84rem; color:#CBD5E1; display:block; margin-bottom:0.4rem;'>"
            "Utilize a barra de ferramentas do Ketcher abaixo para desenhar esqueletos aromáticos, "
            "adicionar heteroátomos e anéis. Ao concluir o desenho, clique em <b>Apply</b> no canto inferior direito do editor para aplicar a estrutura."
            "</span>"
        )

        initial_ketcher_val = st.session_state.get("input_smiles_field", "").strip()
        try:
            ketcher_returned = st_ketcher(
                value=initial_ketcher_val,
                height=480,
                molecule_format="MOLFILE",
                key="ketcher_molecular_canvas",
            )
            # Preserva quebras de linha iniciais de blocos MDL Molfile V2000/V3000 antes do parsing RDKit
            raw_ketcher = initial_ketcher_val if ketcher_returned is None else ketcher_returned
            if raw_ketcher and raw_ketcher.strip():
                canon = canonicalize_smiles(raw_ketcher)
                if canon:
                    st.session_state["input_smiles_field"] = canon
                    st.session_state["field_smiles_text"] = canon
                    raw_input_structure = canon
                elif "M  END" in raw_ketcher or "V2000" in raw_ketcher or "V3000" in raw_ketcher:
                    mol_check = Chem.MolFromMolBlock(raw_ketcher)
                    if mol_check is None or mol_check.GetNumAtoms() == 0:
                        raw_input_structure = ""
                        st.session_state["input_smiles_field"] = ""
                        st.session_state["field_smiles_text"] = ""
                    else:
                        raw_input_structure = raw_ketcher.strip()
                else:
                    raw_input_structure = raw_ketcher.strip()
            else:
                raw_input_structure = ""
                if ketcher_returned == "":
                    st.session_state["input_smiles_field"] = ""
                    st.session_state["field_smiles_text"] = ""
        except Exception as exc:
            st.warning(f"⚠️ Não foi possível carregar o componente Ketcher no navegador: {exc}")
            raw_input_structure = ""

    # -------------------------------------------------------------------------
    # C. Validação Química RDKit e Gerenciamento de Estados de Interface
    # -------------------------------------------------------------------------
    render_html("<div style='margin-top:0.8rem;'></div>")

    # Estado 1: Aguardando estrutura
    if not raw_input_structure:
        render_html(
            """
            <div class="metric-card" style="border-left: 4px solid #64748B; padding: 1rem 1.25rem;">
                <div style="display:flex; align-items:center; gap:8px;">
                    <span style="font-size:1.15rem;">⏳</span>
                    <span style="color:#94A3B8; font-weight:700; font-size:0.95rem;">
                        Estado: Aguardando Estrutura Química
                    </span>
                </div>
                <p style="font-size:0.83rem; color:#94A3B8; margin:0.35rem 0 0 0; line-height:1.45;">
                    Nenhuma molécula fornecida. Digite um código SMILES ou utilize as ferramentas do Ketcher
                    para desenhar sua molécula. Você também pode carregar um controle de referência acima.
                </p>
            </div>
            """
        )
        render_session_history_section()
        return

    # Estado 2: Validação via RDKit
    canonical_smi = canonicalize_smiles(raw_input_structure)

    # Estado 2.1: Estrutura inválida (erro sintático ou químico)
    if canonical_smi is None:
        escaped_raw = html.escape(raw_input_structure)
        render_html(
            f"""
            <div class="metric-card" style="border-left: 4px solid #EF4444; padding: 1.1rem 1.3rem;">
                <div style="display:flex; align-items:center; gap:8px;">
                    <span style="font-size:1.2rem;">❌</span>
                    <span style="color:#EF4444; font-weight:700; font-size:1.02rem;">
                        Estrutura Química Inválida
                    </span>
                </div>
                <p style="font-size:0.84rem; color:#CBD5E1; margin:0.4rem 0 0.5rem 0; line-height:1.45;">
                    O RDKit não pôde interpretar o código molecular submetido. Verifique se há inconsistências químicas:
                </p>
                <ul style="font-size:0.81rem; color:#94A3B8; margin:0.2rem 0 0.6rem 1.2rem; padding:0; line-height:1.4;">
                    <li><strong>Valência excedida:</strong> carbono com mais de 4 ligações ou nitrogênio neutro com 4 ligações;</li>
                    <li><strong>Anéis não fechados:</strong> índices de ciclo não emparelhados (ex: <code>c1ccccc</code> sem o fechamento);</li>
                    <li><strong>Ramificações desbalanceadas:</strong> parênteses de cadeia lateral abertos e não fechados;</li>
                    <li><strong>Símbolos de heteroátomos:</strong> caracteres tipográficos não reconhecidos na convenção Daylight.</li>
                </ul>
                <div style="font-size:0.77rem; color:#CBD5E1; background:rgba(239,68,68,0.1); border-radius:6px; padding:4px 10px; font-family:'JetBrains Mono',monospace;">
                    Entrada com erro: <code>{escaped_raw}</code>
                </div>
            </div>
            """
        )
        render_session_history_section()
        return

    # Estado 3: Estrutura Válida — Preparar Dados e Visualização
    mol_obj = Chem.MolFromSmiles(canonical_smi)
    if mol_obj is None or mol_obj.GetNumAtoms() == 0:
        st.error("Erro interno ao reconstruir objeto molecular a partir do SMILES canônico.")
        return

    # Preserva current_analysis em st.session_state durante edição de nova estrutura

    # Consulta de identificação PubChem (não-bloqueante com cache)
    pubchem_info = identify_compound_pubchem(canonical_smi, timeout=3.5, use_cache=True)

    # Consulta de proveniência em datasets locais e referências conhecidas
    resolve_prov_fn = models_bundle.get("resolve_provenance")
    provenance = resolve_prov_fn(canonical_smi) if resolve_prov_fn else {}

    # Propriedades químicas rápidas para pré-visualização
    formula = rdMolDescriptors.CalcMolFormula(mol_obj)
    mol_wt = Descriptors.MolWt(mol_obj)
    heavy_atoms = mol_obj.GetNumHeavyAtoms()

    # -------------------------------------------------------------------------
    # D. Exibição da Molécula Validada (Preview 2D + PubChem)
    # -------------------------------------------------------------------------
    col_preview_2d, col_pubchem = st.columns([1.1, 1.4])

    with col_preview_2d:
        mol_b64 = render_molecule_dark_base64(mol_obj, size=(420, 250))
        if mol_b64:
            img_tag = (
                f'<div class="mol-canvas-frame">'
                f'<img src="data:image/png;base64,{mol_b64}" style="max-width:100%; height:175px; object-fit:contain; border-radius:8px;"/>'
                f'</div>'
            )
        else:
            img_tag = '<div style="color:#94A3B8; padding:2rem 0; text-align:center;">Visualização gráfica indisponível</div>'

        render_html(
            f"""
            <div class="metric-card">
                <div class="card-header-title">
                    <span>🧪 Estrutura Validada (RDKit)</span>
                    <span style="font-size:0.75rem; background:rgba(0,210,190,0.15); color:#00D2BE; border:1px solid #00D2BE; border-radius:10px; padding:2px 8px; font-weight:700;">
                        ✓ Válida
                    </span>
                </div>
                {img_tag}
                <div style="display:flex; justify-content:space-between; font-size:0.77rem; color:#CBD5E1; margin:0.4rem 0 0.3rem 0;">
                    <span>Fórmula: <strong>{html.escape(formula)}</strong></span>
                    <span>Massa: <strong>{mol_wt:.1f} Da</strong></span>
                    <span>Átomos pesados: <strong>{heavy_atoms}</strong></span>
                </div>
                <div class="smiles-box" title="SMILES Canônico">{html.escape(canonical_smi)}</div>
            </div>
            """
        )

    with col_pubchem:
        pubchem_card_html = build_pubchem_card_html(pubchem_info)
        if pubchem_card_html:
            render_html(pubchem_card_html)
        else:
            st.info(f"InChIKey: {Chem.inchi.MolToInchiKey(mol_obj) if mol_obj else 'N/A'}")

    # -------------------------------------------------------------------------
    # E. Botão de Disparo Explícito da Análise Multiobjetivo
    # -------------------------------------------------------------------------
    render_html("<div style='margin-top:0.6rem;'></div>")

    col_btn_run, col_status_msg = st.columns([1.6, 2.4])
    with col_btn_run:
        btn_exec = st.button(
            "🚀 Analisar Molécula (Predição Multiobjetivo)",
            type="primary",
            width="stretch",
            help="Executa a featurização 1030-D, inferência de afinidade A2A, cardiotoxicidade hERG, AD e SHAP.",
        )

    with col_status_msg:
        if st.session_state.get("current_analysis") is not None and st.session_state.get("current_analysis_smiles") == canonical_smi:
            render_html(
                "<span style='color:#00D2BE; font-size:0.86rem; font-weight:600; display:inline-block; margin-top:0.45rem;'>"
                "✓ Predições atualizadas para a estrutura visualizada."
                "</span>"
            )
        else:
            render_html(
                "<span style='color:#94A3B8; font-size:0.84rem; display:inline-block; margin-top:0.45rem;'>"
                "Pressione o botão para executar os modelos nos 1030 atributos."
                "</span>"
            )

    # -------------------------------------------------------------------------
    # F. Processamento da Análise ao Clicar no Botão
    # -------------------------------------------------------------------------
    if btn_exec:
        with st.spinner("Processando featurização 1030-D, modelos XGBoost/Random Forest e diagnósticos AD..."):
            try:
                analysis_dict = execute_molecular_pipeline(
                    canonical_smi=canonical_smi,
                    mol=mol_obj,
                    input_method="smiles" if selected_mode == "✏️ Inserir SMILES" else "ketcher",
                    raw_input=raw_input_structure,
                    models_bundle=models_bundle,
                    provenance_info=provenance,
                    pubchem_info=pubchem_info,
                )
                add_to_history(analysis_dict)
                st.session_state["current_analysis"] = analysis_dict
                st.session_state["current_analysis_smiles"] = canonical_smi
                st.toast("✅ Análise multiobjetivo concluída e salva no histórico!", icon="🎯")
            except Exception as exc:
                st.error(f"❌ Erro durante o processamento da análise molecular: {exc}")
                logger.exception("Falha ao executar pipeline molecular:")

    # -------------------------------------------------------------------------
    # G. Apresentação do Resumo Executivo caso a Análise esteja Pronta
    # -------------------------------------------------------------------------
    current_analysis = st.session_state.get("current_analysis")
    if (
        current_analysis is not None
        and st.session_state.get("current_analysis_smiles") == canonical_smi
    ):
        render_analysis_executive_banner(current_analysis)

    # Renderiza o histórico de análises da sessão
    render_session_history_section()
