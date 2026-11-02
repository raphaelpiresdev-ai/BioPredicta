"""
Módulo da Página: Resultados da Análise Molecular (Etapa 3 — UX/UI Refactoring).

Apresenta o painel completo, organizado e cientificamente fundamentado da molécula analisada.
Consome estritamente os dados pré-calculados em st.session_state["current_analysis"] sem executar
novamente nenhuma etapa de inferência, featurização ou cálculo explicativo SHAP.
"""

from __future__ import annotations

import html
import logging
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
import streamlit as st

from src.ad_utils import build_ad_section_html
from src.chem_utils import FEATURE_NAMES, smiles_to_mol
from src.ui.catalogs import (
    DESCRIPTOR_INFO_CATALOG,
    PHARMACOPHORE_BIT_CATALOG,
    resolve_translated_feature_name,
)
from src.ui.components import (
    build_multiobjective_figure,
    build_shap_impact_figure,
    get_page_target,
    render_html,
    render_molecule_dark_base64,
    render_page_header,
)

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# 1. Funções Auxiliares de Apresentação e Formatação
# -----------------------------------------------------------------------------
def format_ic50(ic50_nm: float) -> str:
    """Formata valor de concentração inibitória média (IC50) em escala adequada."""
    if ic50_nm < 1.0:
        return f"{ic50_nm * 1000:.1f} pM"
    elif ic50_nm < 1000.0:
        return f"{ic50_nm:.1f} nM"
    elif ic50_nm < 1_000_000.0:
        return f"{ic50_nm / 1000.0:.2f} µM"
    else:
        return f"{ic50_nm / 1_000_000.0:.2f} mM"


def resolve_compound_display_name(analysis: Dict[str, Any]) -> Tuple[str, bool]:
    """
    Retorna (nome_de_exibicao, is_catalogo_comercial).
    Prioriza títulos reais do PubChem ou provenientes de catálogo farmacológico.
    """
    pubchem = analysis.get("pubchem") or {}
    prov = analysis.get("provenance") or {}

    title = pubchem.get("title") or pubchem.get("compound_name")
    if title:
        return str(title).strip(), True

    prov_name = prov.get("compound_name") or prov.get("name")
    if prov_name:
        return str(prov_name).strip(), True

    return "Composto em Avaliação (In Silico)", False


# -----------------------------------------------------------------------------
# 2. Renderização da Página Principal
# -----------------------------------------------------------------------------
def render_resultados(analysis: Optional[Dict[str, Any]] = None) -> None:
    """
    Renderiza o relatório científico completo de resultados da análise ativa.
    Se analysis não for fornecido, consulta st.session_state["current_analysis"].
    """
    if analysis is None:
        analysis = st.session_state.get("current_analysis")

    # -------------------------------------------------------------------------
    # Estado Vazio: Nenhuma análise ativa
    # -------------------------------------------------------------------------
    if not analysis:
        render_page_header(
            title="Painel de Resultados Multiobjetivo",
            subtitle="Consulte o veredito computacional, matriz cartesiana 2D, cobertura do domínio e interpretabilidade SHAP.",
            icon="📊",
        )

        render_html(
            """
            <div class="metric-card" style="border-left: 4px solid #F59E0B; padding: 1.5rem 1.8rem; margin-top: 1rem;">
                <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 0.6rem;">
                    <span style="font-size: 1.6rem;">🧪</span>
                    <span style="font-size: 1.15rem; font-weight: 700; color: #FFFFFF;">
                        Nenhuma Análise Molecular Ativa
                    </span>
                </div>
                <p style="font-size: 0.90rem; color: #CBD5E1; line-height: 1.55; margin: 0 0 1.2rem 0;">
                    Para visualizar os resultados preditivos multiobjetivo (afinidade pelo receptor <b>A2A</b> e escore de bloqueio do canal <b>hERG</b>),
                    é necessário primeiro submeter uma molécula via código linear SMILES ou desenhá-la no editor interativo na página <b>Nova Análise</b>.
                </p>
            </div>
            """
        )

        col_nav1, col_nav2 = st.columns([1.5, 2.5])
        with col_nav1:
            st.page_link(
                get_page_target("nova-analise"),
                label="👉 Ir para Nova Análise Molecular",
                icon="🧪",
            )
        with col_nav2:
            st.caption("Você poderá carregar moléculas modelo com 1 clique para testar a plataforma instantaneamente.")
        return

    # -------------------------------------------------------------------------
    # Extração de Dados da Sessão (Contrato canônico de current_analysis)
    # -------------------------------------------------------------------------
    canonical_smi = analysis.get("canonical_smiles", "")
    mol = analysis.get("mol")
    if mol is None and canonical_smi:
        mol = smiles_to_mol(canonical_smi)

    formula = analysis.get("formula", "")
    if not formula and mol:
        formula = rdMolDescriptors.CalcMolFormula(mol)

    mol_wt = analysis.get("mol_wt", 0.0)
    if (mol_wt is None or mol_wt <= 0) and mol:
        mol_wt = float(Descriptors.MolWt(mol))

    desc = analysis.get("descriptors") or {}
    pubchem = analysis.get("pubchem") or {}
    provenance = analysis.get("provenance") or {}

    a2a = analysis.get("a2a") or {}
    herg = analysis.get("herg") or {}
    multiobj = analysis.get("multiobjective") or {}
    quadrant = multiobj.get("quadrant") or {}

    pIC50 = float(a2a.get("pred_pIC50", 0.0))
    ic50_nm = float(a2a.get("pred_IC50_nM", 10 ** (9.0 - pIC50)))
    a2a_class = a2a.get("classification") or {}
    if not isinstance(a2a_class, dict):
        a2a_class = {}

    prob_herg = float(herg.get("prob_herg", 0.0))
    safety_idx = float(herg.get("safety_index", 1.0 - prob_herg))
    herg_class = herg.get("classification") or {}
    if not isinstance(herg_class, dict):
        herg_class = {}

    ad_a2a = a2a.get("ad") or {}
    ad_herg = herg.get("ad") or {}

    compound_name, has_commercial_name = resolve_compound_display_name(analysis)
    compound_name_esc = html.escape(str(compound_name))
    canonical_smi_esc = html.escape(str(canonical_smi))
    formula_esc = html.escape(str(formula))

    # -------------------------------------------------------------------------
    # A. Cabeçalho Padronizado e Identificação Molecular
    # -------------------------------------------------------------------------
    render_page_header(
        title="Painel de Resultados Multiobjetivo",
        subtitle=f"Avaliação computacional integrada para {compound_name_esc}: afinidade A2A, segurança hERG e explicabilidade SHAP.",
        icon="📊",
    )

    with st.container():
        col_img, col_info = st.columns([1.6, 2.4])

        with col_img:
            render_html(
                """
                <div style="font-size:0.75rem; text-transform:uppercase; color:#94A3B8; font-weight:700; letter-spacing:0.5px; margin-bottom:0.4rem;">
                    Estrutura Molecular Bidimensional (Dark Mode)
                </div>
                """
            )
            b64_mol = render_molecule_dark_base64(mol, size=(380, 240)) if mol else None
            if b64_mol:
                render_html(
                    f"""
                    <div style="background:#1E222D; border:1px solid #333B4D; border-radius:10px; padding:10px; text-align:center;">
                        <img src="data:image/png;base64,{b64_mol}" style="max-width:100%; height:auto;" alt="Estrutura 2D da Molécula" />
                    </div>
                    """
                )
            else:
                st.info("Visualização 2D não gerada pelo renderizador.")

        with col_info:
            pubchem_cid = pubchem.get("cid")

            cid_badge = (
                f'<span style="background:rgba(56,189,248,0.18); color:#38BDF8; padding:3px 9px; border-radius:12px; font-size:0.75rem; font-weight:700;">PubChem CID {html.escape(str(pubchem_cid))}</span>'
                if pubchem_cid
                else '<span style="background:rgba(148,163,184,0.18); color:#94A3B8; padding:3px 9px; border-radius:12px; font-size:0.75rem;">Sem Registro PubChem</span>'
            )

            render_html(
                f"""
                <div class="metric-card" style="margin-bottom:0;">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.6rem; flex-wrap:wrap; gap:6px;">
                        <span style="font-size:1.18rem; font-weight:800; color:#FFFFFF;">
                            {compound_name_esc}
                        </span>
                        {cid_badge}
                    </div>
                    <div style="margin-bottom:0.6rem;">
                        <div style="font-size:0.72rem; text-transform:uppercase; color:#94A3B8; font-weight:600;">SMILES Canônico Isomérico</div>
                        <div class="mono-font" style="font-size:0.80rem; background:#0E1117; padding:6px 8px; border-radius:6px; border:1px solid #333B4D; word-break:break-all; color:#00D2BE; margin-top:2px;">
                            {canonical_smi_esc}
                        </div>
                    </div>
                    <div style="display:grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap:8px; margin-top:0.7rem;">
                        <div style="background:#252A37; padding:7px 10px; border-radius:6px;">
                            <div style="font-size:0.70rem; color:#94A3B8;">Fórmula Molecular</div>
                            <div style="font-size:0.92rem; font-weight:700; color:#FFFFFF;">{formula_esc}</div>
                        </div>
                        <div style="background:#252A37; padding:7px 10px; border-radius:6px;">
                            <div style="font-size:0.70rem; color:#94A3B8;">Massa Molecular</div>
                            <div style="font-size:0.92rem; font-weight:700; color:#FFFFFF;">{mol_wt:.2f} Da</div>
                        </div>
                        <div style="background:#252A37; padding:7px 10px; border-radius:6px;">
                            <div style="font-size:0.70rem; color:#94A3B8;">Lipofilicidade (LogP)</div>
                            <div style="font-size:0.92rem; font-weight:700; color:#FFFFFF;">{desc.get('MolLogP', 0.0):.2f}</div>
                        </div>
                        <div style="background:#252A37; padding:7px 10px; border-radius:6px;">
                            <div style="font-size:0.70rem; color:#94A3B8;">Polaridade (TPSA)</div>
                            <div style="font-size:0.92rem; font-weight:700; color:#FFFFFF;">{desc.get('TPSA', 0.0):.1f} Å²</div>
                        </div>
                    </div>
                </div>
                """
            )

    # -------------------------------------------------------------------------
    # B. Veredito Executivo de Decisão Farmacológica
    # -------------------------------------------------------------------------
    render_html("<div style='margin-top:1.4rem;'></div>")
    st.markdown("### 🎯 Veredito Executivo de Triagem")

    smax_a2a_val = ad_a2a.get("s_max_all")
    smax_herg_val = ad_herg.get("s_max_all")
    badge_ad_a2a = f"S_max {smax_a2a_val:.1%}" if smax_a2a_val is not None else "N/A"
    badge_ad_herg = f"S_max {smax_herg_val:.1%}" if smax_herg_val is not None else "N/A"

    if isinstance(quadrant, dict):
        quad_badge_class = quadrant.get("quad_badge_cls", "semaphore-yellow")
        quad_title_str = html.escape(str(quadrant.get("quad_title", "Posicionamento Multiobjetivo")))
        quad_desc_str = html.escape(
            str(quadrant.get("quad_desc", "Avaliação concorrente de potência em A2A e segurança hERG."))
        )
    else:
        quad_badge_class = "semaphore-yellow"
        quad_title_str = html.escape(str(quadrant))
        quad_desc_str = "Avaliação concorrente de potência em A2A e segurança hERG."

    render_html(
        f"""
        <div class="metric-card" style="border-left: 5px solid #00D2BE;">
            <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:8px; margin-bottom:0.75rem;">
                <div style="display:flex; align-items:center; gap:8px;">
                    <span style="font-size:1.25rem;">⚖️</span>
                    <span style="font-size:1.08rem; font-weight:700; color:#FFFFFF;">Síntese da Avaliação Multiobjetivo</span>
                </div>
                <span class="{quad_badge_class}" style="padding:4px 14px; font-size:0.83rem; font-weight:700;">
                    {quad_title_str}
                </span>
            </div>
            <p style="font-size:0.86rem; color:#CBD5E1; line-height:1.5; margin:0 0 1rem 0;">
                {quad_desc_str}
            </p>
            <div class="kpi-grid" style="grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));">
                <div class="kpi-box">
                    <div class="kpi-label">Potência A2A (pIC50)</div>
                    <div class="kpi-value kpi-value-accent">{pIC50:.2f}</div>
                    <div style="font-size:0.78rem; color:#94A3B8; margin-top:0.25rem;">IC50 Estimado: {format_ic50(ic50_nm)}</div>
                    <div style="font-size:0.74rem; color:#38BDF8; margin-top:0.2rem;">AD A2A: {badge_ad_a2a}</div>
                </div>
                <div class="kpi-box">
                    <div class="kpi-label">Risco Cardíaco hERG P(Kv11.1)</div>
                    <div class="kpi-value" style="color:{herg_class.get('bar_color', '#00D2BE')};">{prob_herg:.1%}</div>
                    <div style="font-size:0.78rem; color:#94A3B8; margin-top:0.25rem;">Classe: {html.escape(str(herg_class.get('risk_class', 'N/A')).title())} Risco</div>
                    <div style="font-size:0.74rem; color:#38BDF8; margin-top:0.2rem;">AD hERG: {badge_ad_herg}</div>
                </div>
                <div class="kpi-box">
                    <div class="kpi-label">Índice Complementar (1 - P)</div>
                    <div class="kpi-value" style="font-size:1.2rem; color:#FFFFFF;">{safety_idx:.1%}</div>
                    <div style="font-size:0.74rem; color:#94A3B8; margin-top:0.25rem;">Coordenada Y cartesiana</div>
                    <div style="font-size:0.72rem; color:#A855F7; margin-top:0.2rem;">Otimização Multiobjetivo</div>
                </div>
                <div class="kpi-box">
                    <div class="kpi-label">Atributos Químicos</div>
                    <div class="kpi-value" style="font-size:1.2rem; color:#FFFFFF;">1.030</div>
                    <div style="font-size:0.78rem; color:#94A3B8; margin-top:0.25rem;">6 Físico-Químicos + 1024 Morgan</div>
                    <div style="font-size:0.72rem; color:#00D2BE; margin-top:0.2rem;">Inferência 100% Concluída</div>
                </div>
            </div>
            <div style="background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.08); border-radius:6px; padding:8px 12px; margin-top:1rem;">
                <span style="font-size:0.76rem; color:#94A3B8; line-height:1.4; display:block;">
                    ⚠️ <b>Aviso de Governança Científica:</b> As predições computacionais são geradas por modelos supervisionados de Machine Learning
                    (XGBoost para afinidade A<sub>2A</sub> e Random Forest para probabilidade de bloqueio hERG) treinados em dados <i>in vitro</i> de ChEMBL e TDC.
                    Elas constituem ferramentas auxiliares de priorização em triagem virtual e <b>não substituem validação experimental biológica nem equivalem a desfechos clínicos</b>.
                </span>
            </div>
        </div>
        """
    )

    # -------------------------------------------------------------------------
    # C & D. Detalhamento dos Alvos: Receptor A2A e Canal hERG
    # -------------------------------------------------------------------------
    render_html("<div style='margin-top:1.4rem;'></div>")
    st.markdown("### 🔬 Resultados Detalhados por Alvo Biológico")

    col_a2a, col_herg = st.columns(2)

    with col_a2a:
        a2a_desc_esc = html.escape(
            str(a2a_class.get("desc") or a2a_class.get("description", "Potência estimada na faixa operacional do receptor purinérgico."))
        )
        render_html(
            f"""
            <div class="metric-card" style="border-top: 4px solid #00D2BE; height: 100%;">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.7rem;">
                    <span style="font-size:1.02rem; font-weight:700; color:#FFFFFF;">Receptor de Adenosina A2A</span>
                    <span class="{a2a_class.get('color_class', 'semaphore-yellow')}" style="padding:2px 10px; font-size:0.76rem; font-weight:700;">
                        {html.escape(str(a2a_class.get('potency_class', 'Moderada')).upper())}
                    </span>
                </div>
                <div style="display:flex; align-items:baseline; gap:8px; margin-bottom:0.4rem;">
                    <span style="font-size:1.85rem; font-weight:800; color:#00D2BE; font-family:'JetBrains Mono',monospace;">
                        {pIC50:.2f}
                    </span>
                    <span style="font-size:0.85rem; color:#94A3B8;">pIC50 predito</span>
                </div>
                <div style="background:#252A37; border-radius:6px; padding:8px 12px; margin-bottom:0.8rem;">
                    <div style="font-size:0.75rem; color:#94A3B8;">Potência Estimada (IC50)</div>
                    <div style="font-size:1.05rem; font-weight:700; color:#FFFFFF; font-family:'JetBrains Mono',monospace;">
                        {format_ic50(ic50_nm)}
                    </div>
                </div>
                <div style="font-size:0.82rem; color:#CBD5E1; line-height:1.45; margin-bottom:0.8rem;">
                    {a2a_desc_esc}
                </div>
                <div style="border-top:1px solid #333B4D; padding-top:0.6rem; font-size:0.74rem; color:#94A3B8; line-height:1.4;">
                    <b>Limiares Operacionais:</b><br/>
                    • Alta Potência: pIC50 ≥ 6,5 (faixa nanomolar sub-300 nM)<br/>
                    • Potência Moderada: 5,0 ≤ pIC50 &lt; 6,5 (faixa micromolar)<br/>
                    • Baixa Potência / Inativo: pIC50 &lt; 5,0 (&gt; 10 µM)<br/>
                    <i>Nota: A predição estima afinidade global de ligação sem inferir seletividade frente a A1, A2B ou A3.</i>
                </div>
            </div>
            """
        )

    with col_herg:
        herg_desc_esc = html.escape(
            str(herg_class.get("desc") or herg_class.get("description", "Classificação de probabilidade estimada de bloqueio."))
        )
        render_html(
            f"""
            <div class="metric-card" style="border-top: 4px solid {herg_class.get('bar_color', '#00D2BE')}; height: 100%;">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.7rem;">
                    <span style="font-size:1.02rem; font-weight:700; color:#FFFFFF;">Canal de Potássio hERG (Kv11.1)</span>
                    <span style="background:rgba(255,255,255,0.08); color:{herg_class.get('bar_color', '#00D2BE')}; border:1px solid {herg_class.get('bar_color', '#00D2BE')}; padding:2px 10px; border-radius:12px; font-size:0.76rem; font-weight:700;">
                        {html.escape(str(herg_class.get('risk_class', 'Baixo')).upper())} RISCO
                    </span>
                </div>
                <div style="display:flex; align-items:baseline; gap:8px; margin-bottom:0.4rem;">
                    <span style="font-size:1.85rem; font-weight:800; color:{herg_class.get('bar_color', '#00D2BE')}; font-family:'JetBrains Mono',monospace;">
                        {prob_herg:.1%}
                    </span>
                    <span style="font-size:0.85rem; color:#94A3B8;">P(bloqueio hERG)</span>
                </div>
                <div style="background:#252A37; border-radius:6px; padding:8px 12px; margin-bottom:0.8rem;">
                    <div style="font-size:0.75rem; color:#94A3B8;">Índice Complementar (1.0 - P_hERG)</div>
                    <div style="font-size:1.05rem; font-weight:700; color:#FFFFFF; font-family:'JetBrains Mono',monospace;">
                        {safety_idx:.1%}
                    </div>
                </div>
                <div style="font-size:0.82rem; color:#CBD5E1; line-height:1.45; margin-bottom:0.8rem;">
                    {herg_desc_esc}
                </div>
                <div style="border-top:1px solid #333B4D; padding-top:0.6rem; font-size:0.74rem; color:#94A3B8; line-height:1.4;">
                    <b>Limiares Canônicos de Decisão:</b><br/>
                    • Baixo Risco: P &lt; 0,30 (coord. Y &gt; 0,70)<br/>
                    • Risco Moderado: 0,30 ≤ P ≤ 0,70 (0,30 ≤ Y ≤ 0,70)<br/>
                    • Alto Risco: P &gt; 0,70 (coord. Y &lt; 0,30)<br/>
                    <i>Nota: Escore de classificação binária in vitro; não equivale a risco clínico direto de prolongamento de QT.</i>
                </div>
            </div>
            """
        )

    # -------------------------------------------------------------------------
    # E. Domínio de Aplicabilidade (AD)
    # -------------------------------------------------------------------------
    render_html("<div style='margin-top:1.4rem;'></div>")
    st.markdown("### 🌐 Cobertura do Domínio de Aplicabilidade (AD)")
    st.caption(
        "Avaliação da similaridade estrutural (Tanimoto ECFP4) e dos descritores físico-químicos frente aos compostos de calibração do modelo."
    )

    try:
        ad_html = build_ad_section_html(ad_a2a, ad_herg)
        render_html(ad_html)
    except Exception as exc:
        logger.warning(f"Erro ao renderizar seção de AD: {exc}")
        st.info("ℹ️ Diagnóstico de Domínio de Aplicabilidade incompleto ou em formato preliminar.")

    # -------------------------------------------------------------------------
    # F. Matriz Multiobjetivo Cartesiana (A2A × hERG)
    # -------------------------------------------------------------------------
    render_html("<div style='margin-top:1.4rem;'></div>")
    st.markdown("### 🗺️ Matriz Cartesiana Multiobjetivo (A2A × hERG)")
    st.caption(
        "Posicionamento da molécula no espaço bidimensional de decisão: Potência predita em A2A (Eixo X) vs. "
        "Índice Complementar Derivado 1 - P_hERG (Eixo Y)."
    )

    try:
        fig_mat, ax_mat = build_multiobjective_figure(
            x_cur=pIC50,
            y_cur=safety_idx,
            active_name=compound_name,
        )
        st.pyplot(fig_mat, width="stretch")
        plt.close(fig_mat)
    except Exception as exc:
        logger.warning(f"Erro ao renderizar gráfico multiobjetivo: {exc}")
        st.info("Gráfico multiobjetivo indisponível com as coordenadas fornecidas.")

    with st.expander("📖 Guia de Leitura e Critérios dos Quadrantes Cartesianos", expanded=False):
        st.markdown(
            r"""
            | Quadrante | Critério Operacional | Interpretação Farmacológica |
            | :--- | :--- | :--- |
            | **🏆 Quadrante 1 (Ideal)** | $pIC_{50} \ge 6,5$ e $P_{\text{hERG}} < 0,30$ | Alta afinidade nanomolar aliada a baixo escore de cardiotoxicidade. Molécula de alta prioridade. |
            | **🌿 Quadrante 2 (Baixo Interesse)** | $pIC_{50} < 5,0$ e $P_{\text{hERG}} < 0,30$ | Perfil cardíaco favorável, porém sem afinidade relevante pelo receptor A2A. |
            | **❌ Quadrante 3 (Descarte)** | $pIC_{50} < 5,0$ e $P_{\text{hERG}} > 0,70$ | Inativo no alvo terapêutico e alto risco predito de bloqueio de canal cardíaco. Descarte imediato. |
            | **⚠️ Quadrante 4 (Alerta Crítico)** | $pIC_{50} \ge 6,5$ e $P_{\text{hERG}} > 0,70$ | Alta afinidade biológica, mas associada a alto escore cardiotóxico. Exige desderivação guiada. |
            | **🟡 Zona de Otimização** | Faixas intermediárias | $5,0 \le pIC_{50} < 6,5$ e/ou $0,30 \le P_{\text{hERG}} \le 0,70$. Hit com potencial de otimização estrutural. |
            
            *Aviso: O eixo Y (1 - P) é uma construção matemática auxiliar de otimização cartesiana multiobjetivo; não representa probabilidade clínica nem margem de segurança clínica in vivo.*
            """
        )

    # -------------------------------------------------------------------------
    # G. Interpretabilidade Explicativa SHAP (A2A e hERG)
    # -------------------------------------------------------------------------
    render_html("<div style='margin-top:1.6rem;'></div>")
    st.markdown("### 🧩 Interpretabilidade Explicativa Local (SHAP)")
    st.caption(
        "Decomposição aditiva das contribuições locais dos descritores moleculares e fragmentos ECFP4 "
        "para as saídas dos modelos preditivos, recuperadas diretamente do pipeline sem novo cálculo."
    )

    shap_a2a_vals = a2a.get("shap_values")
    base_a2a = float(a2a.get("base_value", 0.0))

    shap_herg_vals = herg.get("shap_values")
    base_herg = float(herg.get("base_value", 0.0))

    tab_shap_a2a, tab_shap_herg = st.tabs([
        "🎯 Decomposição SHAP: Receptor A2A (Afinidade)",
        "🫀 Decomposição SHAP: Canal hERG (Risco de Bloqueio)",
    ])

    with tab_shap_a2a:
        if shap_a2a_vals is not None and len(shap_a2a_vals) > 0 and not np.all(shap_a2a_vals == 0):
            fig_sa, ax_sa, top_items_a2a = build_shap_impact_figure(
                shap_values=np.asarray(shap_a2a_vals),
                base_value=base_a2a,
                pred_value=pIC50,
                target="A2A",
                top_n=10,
                mol=mol,
            )
            if fig_sa:
                st.pyplot(fig_sa, width="stretch")
                plt.close(fig_sa)

            st.markdown("#### 🔍 Detalhamento dos Atributos Mais Influentes para $A_{2A}$")
            cols_grid_a = st.columns(2)
            for idx_card, item in enumerate(top_items_a2a[:6]):
                feat_key = item["feature_name"]
                impact_val = item["impact"]
                tr_name = html.escape(str(item["translated_name"]))
                color_sgn = "#00D2BE" if impact_val >= 0 else "#FF4B4B"
                sign_str = f"+{impact_val:.3f}" if impact_val >= 0 else f"{impact_val:.3f}"

                role_txt = "Atributo com contribuição aditiva para o modelo de afinidade A2A."
                if feat_key in DESCRIPTOR_INFO_CATALOG:
                    role_txt = DESCRIPTOR_INFO_CATALOG[feat_key].get("role_a2a", role_txt)
                elif feat_key.startswith("fp_"):
                    try:
                        b_id = int(feat_key.split("_")[1])
                        if b_id in PHARMACOPHORE_BIT_CATALOG:
                            role_txt = PHARMACOPHORE_BIT_CATALOG[b_id].get("role_a2a", role_txt)
                    except Exception:
                        pass

                with cols_grid_a[idx_card % 2]:
                    render_html(
                        f"""
                        <div style="background:#1E222D; border:1px solid #333B4D; border-left:4px solid {color_sgn}; border-radius:8px; padding:10px 12px; margin-bottom:10px;">
                            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;">
                                <span style="font-size:0.85rem; font-weight:700; color:#FFFFFF;">{tr_name}</span>
                                <span style="font-family:'JetBrains Mono',monospace; font-size:0.85rem; font-weight:700; color:{color_sgn};">{sign_str}</span>
                            </div>
                            <div style="font-size:0.73rem; color:#94A3B8; margin-bottom:4px;">Código: <code>{html.escape(str(feat_key))}</code></div>
                            <div style="font-size:0.77rem; color:#CBD5E1; line-height:1.4;">{html.escape(str(role_txt))}</div>
                        </div>
                        """
                    )
        else:
            st.info("Decomposição SHAP para A2A não disponível ou com valores neutros para esta estrutura.")

    with tab_shap_herg:
        if shap_herg_vals is not None and len(shap_herg_vals) > 0 and not np.all(shap_herg_vals == 0):
            fig_sh, ax_sh, top_items_herg = build_shap_impact_figure(
                shap_values=np.asarray(shap_herg_vals),
                base_value=base_herg,
                pred_value=prob_herg,
                target="hERG",
                top_n=10,
                mol=mol,
            )
            if fig_sh:
                st.pyplot(fig_sh, width="stretch")
                plt.close(fig_sh)

            st.markdown("#### 🔍 Detalhamento dos Alertas e Atributos Mais Influentes para hERG")
            cols_grid_h = st.columns(2)
            for idx_card, item in enumerate(top_items_herg[:6]):
                feat_key = item["feature_name"]
                impact_val = item["impact"]
                tr_name = html.escape(str(item["translated_name"]))
                color_sgn = "#FF4B4B" if impact_val >= 0 else "#00D2BE"
                sign_str = f"+{impact_val:.3f}" if impact_val >= 0 else f"{impact_val:.3f}"

                role_txt = "Atributo com impacto na probabilidade de bloqueio do canal hERG."
                if feat_key in DESCRIPTOR_INFO_CATALOG:
                    role_txt = DESCRIPTOR_INFO_CATALOG[feat_key].get("role_herg", role_txt)
                elif feat_key.startswith("fp_"):
                    try:
                        b_id = int(feat_key.split("_")[1])
                        if b_id in PHARMACOPHORE_BIT_CATALOG:
                            role_txt = PHARMACOPHORE_BIT_CATALOG[b_id].get("role_herg", role_txt)
                    except Exception:
                        pass

                with cols_grid_h[idx_card % 2]:
                    render_html(
                        f"""
                        <div style="background:#1E222D; border:1px solid #333B4D; border-left:4px solid {color_sgn}; border-radius:8px; padding:10px 12px; margin-bottom:10px;">
                            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;">
                                <span style="font-size:0.85rem; font-weight:700; color:#FFFFFF;">{tr_name}</span>
                                <span style="font-family:'JetBrains Mono',monospace; font-size:0.85rem; font-weight:700; color:{color_sgn};">{sign_str}</span>
                            </div>
                            <div style="font-size:0.73rem; color:#94A3B8; margin-bottom:4px;">Código: <code>{html.escape(str(feat_key))}</code></div>
                            <div style="font-size:0.77rem; color:#CBD5E1; line-height:1.4;">{html.escape(str(role_txt))}</div>
                        </div>
                        """
                    )
        else:
            st.info("Decomposição SHAP para hERG não disponível ou com valores neutros para esta estrutura.")

    render_html(
        """
        <div style="background:rgba(255,255,255,0.02); border:1px solid rgba(255,255,255,0.06); border-radius:6px; padding:8px 12px; margin-top:0.8rem;">
            <span style="font-size:0.75rem; color:#94A3B8; line-height:1.4; display:block;">
                ℹ️ <b>Ressalva Metodológica sobre SHAP:</b> Os valores de impacto refletem estritamente a importância matemática aditiva
                de cada variável dentro do espaço vetorial de entrada do modelo de Machine Learning em relação à média do conjunto de treinamento.
                Eles apontam correlações aprendidas pelo algoritmo e <b>não constituem mecanismo molecular comprovado ou causalidade biológica definitiva</b>.
            </span>
        </div>
        """
    )

    render_html("<div style='margin-top:1.2rem;'></div>")
    st.page_link(
        get_page_target("interpretacao"),
        label="🔬 Aprofundar Interpretação Química e Computacional (Etapa 4)",
        icon="🧩",
    )

    # -------------------------------------------------------------------------
    # H. Ações de Continuidade e Navegação
    # -------------------------------------------------------------------------
    render_html("<div style='margin-top:2rem;'></div>")
    st.markdown("---")
    col_nav_a, col_nav_b, col_nav_c = st.columns([1.8, 1.8, 1.8])

    with col_nav_a:
        st.page_link(
            get_page_target("interpretacao"),
            label="🔬 Interpretação Detalhada",
            icon="🧩",
        )

    with col_nav_b:
        st.page_link(
            get_page_target("nova-analise"),
            label="🧪 Analisar Outra Molécula",
            icon="🔄",
        )

    with col_nav_c:
        st.page_link(
            get_page_target("fundamentos"),
            label="📚 Fundamentos Científicos",
            icon="📖",
        )
