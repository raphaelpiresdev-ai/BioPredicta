"""
Módulo da Página: Interpretação Química e Computacional (Etapa 4 — UX/UI Refactoring).

MultiTarget BioPredict — Arquitetura Multipágina Streamlit.
Apresenta uma experiência didática e cientificamente rigorosa para interpretação
das contribuições SHAP, dos subambientes moleculares Morgan (ECFP4) e do
Domínio de Aplicabilidade (AD).

Consome estritamente os resultados armazenados em st.session_state["current_analysis"]
sem repetir inferências preditivas.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem
import streamlit as st

from src.chem_utils import DESCRIPTOR_NAMES, FEATURE_NAMES, smiles_to_mol
from src.ui.components import (
    get_page_target,
    render_html,
    render_molecule_dark_base64,
    render_page_header,
)
from src.ui.interpretation_engine import (
    build_attribute_explanation_card,
    explain_ad_coverage,
    explain_physicochemical_descriptors,
    extract_morgan_bit_info,
    get_ranked_explanations,
    inspect_bit_environments,
    render_mol_highlight_png,
    render_morgan_subenv_png,
)

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# 1. Funções Auxiliares de Exibição
# -----------------------------------------------------------------------------
def _resolve_compound_display_name(analysis: Dict[str, Any]) -> Tuple[str, str]:
    """Retorna (nome_de_exibicao, subtitulo_de_apoio)."""
    pubchem = analysis.get("pubchem") or {}
    prov = analysis.get("provenance") or {}

    title = pubchem.get("title") or pubchem.get("compound_name")
    if title:
        return str(title).strip(), "Identificação via PubChem"

    prov_name = prov.get("compound_name") or prov.get("name")
    if prov_name:
        return str(prov_name).strip(), "Controle Farmacológico de Referência"

    return "Composto em Avaliação", "Estrutura Fornecida In Silico"


def _render_5_field_card(
    card: Dict[str, Any],
    mol: Optional[Chem.Mol],
    bit_info: Optional[Dict[int, Any]],
    card_index: int,
    target: str,
) -> None:
    """Renderiza visualmente o cartão estruturado de 5 campos obrigatórios."""
    feat_name = card["feature_name"]
    comp = card["computational_attribute"]
    shap = card["shap_contribution"]
    struct = card["structural_subenvironment"]
    obs = card["observational_description"]
    bib = card["bibliographic_hypothesis"]

    # Cores e estilos conforme sinal e modelo
    is_a2a = (target.upper() == "A2A")
    if is_a2a:
        # A2A: Verde/Ciano favorece afinidade (+), Vermelho reduz afinidade (-)
        accent_color = "#00D2BE" if shap["is_positive"] else "#FF4B4B"
        badge_bg = "rgba(0, 210, 190, 0.12)" if shap["is_positive"] else "rgba(255, 75, 75, 0.12)"
    else:
        # hERG: Vermelho eleva risco de bloqueio (+), Verde reduz risco de bloqueio (-)
        accent_color = "#FF4B4B" if shap["is_positive"] else "#00D2BE"
        badge_bg = "rgba(255, 75, 75, 0.12)" if shap["is_positive"] else "rgba(0, 210, 190, 0.12)"

    render_html(f"""
        <div style="background:#1E222D; border:1px solid #333B4D; border-left:5px solid {accent_color};
                    border-radius:10px; padding:1.1rem 1.3rem; margin-bottom:0.8rem;">
            <!-- Cabeçalho do Cartão: Campo 1 (Atributo) e Campo 2 (SHAP) -->
            <div style="display:flex; justify-content:space-between; align-items:flex-start; flex-wrap:wrap; gap:10px; margin-bottom:0.9rem;">
                <div>
                    <span style="font-size:0.75rem; text-transform:uppercase; letter-spacing:0.05em; color:#94A3B8; font-weight:600;">
                        1. ATRIBUTO COMPUTACIONAL • {comp['type_label']}
                    </span>
                    <div style="font-size:1.1rem; font-weight:700; color:#FFFFFF; margin-top:2px;">
                        {comp['translated_name']}
                        <code style="font-size:0.85rem; color:#38BDF8; background:#181B22; padding:2px 6px; border-radius:4px; margin-left:6px;">{comp['feature_id']}</code>
                    </div>
                </div>
                <div style="text-align:right;">
                    <span style="font-size:0.75rem; text-transform:uppercase; letter-spacing:0.05em; color:#94A3B8; font-weight:600;">
                        2. CONTRIBUIÇÃO SHAP (ϕ)
                    </span>
                    <div style="margin-top:2px;">
                        <span style="font-family:'JetBrains Mono',monospace; font-size:1.15rem; font-weight:700; color:{accent_color};
                                     background:{badge_bg}; padding:3px 10px; border-radius:6px; border:1px solid {accent_color}40;">
                            {shap['sign_str']}
                        </span>
                    </div>
                    <div style="font-size:0.75rem; color:#CBD5E1; margin-top:4px;">
                        {shap['direction_desc']}
                    </div>
                </div>
            </div>
            <div style="height:1px; background:#333B4D; margin:0.4rem 0 0 0;"></div>
        </div>
        """)

    # Corpo em duas colunas: Campo 3 (Subambiente) e Campos 4/5 (Química e Literatura)
    col_vis, col_txt = st.columns([1.1, 1.4])

    with col_vis:
        render_html("""
            <span style="font-size:0.72rem; text-transform:uppercase; letter-spacing:0.04em; color:#94A3B8; font-weight:600; display:block; margin-bottom:6px;">
                3. SUBAMBIENTE ESTRUTURAL
            </span>
            """)

        status = struct["status"]

        if status == "scalar_descriptor":
            render_html(f"""
                <div style="background:#181B22; border:1px dashed #3E4759; border-radius:8px; padding:1.2rem; text-align:center;">
                    <div style="font-size:1.8rem; margin-bottom:6px;">📐</div>
                    <div style="font-size:0.85rem; font-weight:600; color:#F8FAFC;">Propriedade Molecular Global</div>
                    <div style="font-size:0.76rem; color:#94A3B8; margin-top:4px;">{struct['message']}</div>
                </div>
                """)

        elif status == "absent":
            render_html(f"""
                <div style="background:#181B22; border:1px dashed #3E4759; border-radius:8px; padding:1.2rem; text-align:center;">
                    <div style="font-size:1.8rem; margin-bottom:6px;">⭕</div>
                    <div style="font-size:0.85rem; font-weight:600; color:#F8FAFC;">Subestrutura Ausente na Molécula (Valor = 0)</div>
                    <div style="font-size:0.76rem; color:#CBD5E1; margin-top:6px; line-height:1.4;">{struct['message']}</div>
                    <div style="font-size:0.7rem; color:#94A3B8; margin-top:8px; font-style:italic;">
                        Nenhuma imagem de fragmento é renderizada pois a característica não está presente na estrutura avaliada.
                    </div>
                </div>
                """)

        elif status == "present":
            envs = struct["environments"]
            num_envs = len(envs)

            # Notificação de colisão de hashing ou multiplicidade
            if struct.get("collision_warning"):
                warn_color = "#F59E0B" if struct["has_collision"] else "#38BDF8"
                render_html(f"""
                    <div style="background:{warn_color}15; border-left:3px solid {warn_color}; border-radius:4px; padding:6px 8px; margin-bottom:8px;">
                        <span style="font-size:0.73rem; color:#F8FAFC; line-height:1.3; display:block;">
                            {struct['collision_warning']}
                        </span>
                    </div>
                    """)

            # Seletor de ocorrência se houver mais de um ambiente
            selected_idx = 0
            if num_envs > 1:
                env_labels = [
                    f"Ambiente #{i+1} (Centro {e['symbol']}{e['center_atom_idx']}, r={e['radius']})"
                    for i, e in enumerate(envs)
                ]
                sel_lbl = st.selectbox(
                    "Selecione o ambiente para visualização:",
                    options=env_labels,
                    index=0,
                    key=f"sel_env_{target}_{feat_name}_{card_index}",
                )
                selected_idx = env_labels.index(sel_lbl)

            active_env = envs[selected_idx]
            bit_id = int(feat_name.split("_")[1]) if "_" in feat_name else -1

            # Renderiza imagens do fragmento isolado e no contexto molecular
            b64_sub = render_morgan_subenv_png(mol, bit_id, bit_info, which_example=selected_idx, size=(160, 160))
            b64_high = render_mol_highlight_png(mol, active_env["atom_indices"], active_env["bond_indices"], size=(280, 180))

            sub_tabs = st.tabs(["🧩 Fragmento 2D", "🔬 Destaque na Molécula"])

            with sub_tabs[0]:
                if b64_sub:
                    render_html(f"""
                        <div style="background:#181B22; border:1px solid #333B4D; border-radius:6px; padding:8px; text-align:center;">
                            <img src="data:image/png;base64,{b64_sub}" style="max-height:150px; width:auto;" alt="Fragmento Morgan" />
                            <div style="font-size:0.7rem; color:#94A3B8; margin-top:4px;">
                                Subambiente isolado (raio {active_env['radius']})
                            </div>
                        </div>
                        """)
                else:
                    st.caption("Visualização 2D do fragmento indisponível.")

            with sub_tabs[1]:
                if b64_high:
                    render_html(f"""
                        <div style="background:#181B22; border:1px solid #333B4D; border-radius:6px; padding:8px; text-align:center;">
                            <img src="data:image/png;base64,{b64_high}" style="max-height:150px; width:auto;" alt="Destaque Molecular" />
                            <div style="font-size:0.7rem; color:#94A3B8; margin-top:4px;">
                                {len(active_env['atom_indices'])} átomos destacados na estrutura completa
                            </div>
                        </div>
                        """)
                else:
                    st.caption("Destaque na estrutura completa indisponível.")

    with col_txt:
        # Campo 4: Descrição Observacional
        render_html(f"""
            <div style="margin-bottom:12px;">
                <span style="font-size:0.72rem; text-transform:uppercase; letter-spacing:0.04em; color:#94A3B8; font-weight:600; display:block; margin-bottom:4px;">
                    4. DESCRIÇÃO QUÍMICA OBSERVACIONAL
                </span>
                <div style="font-size:0.83rem; color:#E2E8F0; line-height:1.45; background:#181B22; border:1px solid #333B4D; border-radius:6px; padding:10px 12px;">
                    {obs}
                </div>
            </div>
            """)

        # Campo 5: Hipótese Bibliográfica
        if bib["has_reference"]:
            render_html(f"""
                <div>
                    <span style="font-size:0.72rem; text-transform:uppercase; letter-spacing:0.04em; color:#00D2BE; font-weight:600; display:block; margin-bottom:4px;">
                        5. HIPÓTESE DA LITERATURA CIENTÍFICA
                    </span>
                    <div style="font-size:0.81rem; color:#CBD5E1; line-height:1.45; background:#181B22; border:1px solid rgba(0, 210, 190, 0.25); border-radius:6px; padding:10px 12px;">
                        {bib['text']}
                    </div>
                    <div style="font-size:0.69rem; color:#94A3B8; margin-top:4px; font-style:italic;">
                        ⚠️ {bib['disclaimer']}
                    </div>
                </div>
                """)
        else:
            render_html("""
                <div>
                    <span style="font-size:0.72rem; text-transform:uppercase; letter-spacing:0.04em; color:#94A3B8; font-weight:600; display:block; margin-bottom:4px;">
                        5. HIPÓTESE DA LITERATURA CIENTÍFICA
                    </span>
                    <div style="font-size:0.78rem; color:#94A3B8; font-style:italic; background:#181B22; border:1px dashed #333B4D; border-radius:6px; padding:8px 12px;">
                        Sem hipótese bibliográfica documentada para este atributo específico.
                    </div>
                </div>
                """)

    render_html("<div style='margin-bottom:1.2rem;'></div>")


# -----------------------------------------------------------------------------
# 2. Renderização da Página Principal
# -----------------------------------------------------------------------------
def render_interpretacao(analysis: Optional[Dict[str, Any]] = None) -> None:
    """Renderiza a experiência completa de interpretação química e computacional."""
    if analysis is None:
        analysis = st.session_state.get("current_analysis")

    # -------------------------------------------------------------------------
    # Estado Vazio: Nenhuma análise ativa
    # -------------------------------------------------------------------------
    if not analysis:
        render_page_header(
            title="Interpretação Química e Computacional",
            subtitle="Decomposição didática das contribuições SHAP, identificação visual de subambientes moleculares e cobertura do domínio de aplicabilidade.",
            icon="🔬",
        )

        render_html("""
            <div class="metric-card" style="border-left: 4px solid #F59E0B; padding: 1.5rem 1.8rem; margin-top: 1rem;">
                <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 0.6rem;">
                    <span style="font-size: 1.6rem;">🧪</span>
                    <span style="font-size: 1.15rem; font-weight: 700; color: #FFFFFF;">
                        Nenhuma Análise Molecular Ativa
                    </span>
                </div>
                <p style="color: #CBD5E1; font-size: 0.92rem; line-height: 1.5; margin-bottom: 1rem;">
                    A interpretação detalhada requer uma molécula processada no pipeline preditivo.
                    Acesse a página <b>Nova Análise</b> para inserir um SMILES, selecionar um controle de referência
                    ou desenhar uma estrutura no editor bidimensional Ketcher.
                </p>
            </div>
            """)

        st.page_link(
            get_page_target("nova-analise"),
            label="🧪 Ir para Nova Análise",
            icon="✨",
        )
        return

    # -------------------------------------------------------------------------
    # Extração de Dados da Análise em Sessão
    # -------------------------------------------------------------------------
    canon_smi = analysis.get("canonical_smiles", "")
    mol = analysis.get("mol")
    if mol is None and canon_smi:
        mol = smiles_to_mol(canon_smi)

    # Extrai bitInfo de Morgan (radius=2, nBits=1024)
    bit_info = extract_morgan_bit_info(mol) if mol is not None else {}

    display_name, name_source = _resolve_compound_display_name(analysis)
    formula = analysis.get("formula", "")
    mol_wt = analysis.get("mol_wt", 0.0)
    desc_dict = analysis.get("descriptors") or {}
    prov = analysis.get("provenance") or {}

    # Dados dos modelos
    a2a = analysis.get("a2a") or {}
    herg = analysis.get("herg") or {}

    pIC50 = a2a.get("pred_pIC50", 0.0)
    ic50_nm = a2a.get("pred_IC50_nM", 0.0)
    prob_herg = herg.get("prob_herg", 0.0)
    safety_idx = herg.get("safety_index", 1.0 - prob_herg)

    shap_a2a = a2a.get("shap_values")
    base_a2a = float(a2a.get("base_value", 0.0))
    shap_herg = herg.get("shap_values")
    base_herg = float(herg.get("base_value", 0.0))

    ad_a2a = a2a.get("ad")
    ad_herg = herg.get("ad")

    # Dicionário com valores numéricos das features para os descritores
    features_df = analysis.get("features")
    feature_vals_map: Dict[str, float] = {}
    if isinstance(features_df, pd.DataFrame) and not features_df.empty:
        feature_vals_map = features_df.iloc[0].to_dict()

    # -------------------------------------------------------------------------
    # Cabeçalho da Página
    # -------------------------------------------------------------------------
    render_page_header(
        title="Interpretação Química e Computacional",
        subtitle=f"Decomposição didática das contribuições SHAP, identificação visual de subambientes e análise de AD para {display_name}.",
        icon="🔬",
    )

    # -------------------------------------------------------------------------
    # Banner de Contexto e Separação Epistemológica
    # -------------------------------------------------------------------------
    col_info_a, col_info_b = st.columns([1.6, 1.4])

    with col_info_a:
        render_html(f"""
            <div style="background:#1E222D; border:1px solid #333B4D; border-radius:10px; padding:14px 18px; margin-bottom:14px;">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
                    <span style="font-size:1.15rem; font-weight:700; color:#FFFFFF;">🧬 {display_name}</span>
                    <span style="font-size:0.75rem; color:#94A3B8; background:#181B22; padding:3px 8px; border-radius:4px;">{name_source}</span>
                </div>
                <div style="font-family:'JetBrains Mono',monospace; font-size:0.8rem; color:#38BDF8; word-break:break-all; margin-bottom:6px;">
                    {canon_smi}
                </div>
                <div style="display:flex; gap:16px; font-size:0.78rem; color:#CBD5E1;">
                    <span><b>Fórmula:</b> {formula}</span>
                    <span><b>Massa:</b> {mol_wt:.2f} Da</span>
                    <span><b>Bits Morgan Ativos:</b> {len(bit_info)}/1024</span>
                </div>
            </div>
            """)

    with col_info_b:
        render_html("""
            <div style="background:rgba(56, 189, 248, 0.06); border:1px solid rgba(56, 189, 248, 0.25); border-radius:10px; padding:14px 18px; margin-bottom:14px;">
                <div style="font-size:0.82rem; font-weight:700; color:#38BDF8; margin-bottom:4px;">
                    💡 Diferenciação Epistemológica Obrigatória
                </div>
                <div style="font-size:0.75rem; color:#CBD5E1; line-height:1.4;">
                    Esta experiência distingue estritamente a <b>Interpretação Computacional</b> (valores SHAP, decomposição aditiva de pesos das árvores de decisão e métricas Tanimoto) da <b>Interpretação Química</b> (reconhecimento visual de subambientes moleculares, características atômicas observáveis e hipóteses bibliográficas registradas). 
                    <br/><span style="color:#94A3B8; font-style:italic;">SHAP reflete estritamente a função de decisão do algoritmo, não comprovando mecanismo biológico in vitro.</span>
                </div>
            </div>
            """)

    # -------------------------------------------------------------------------
    # Abas Principais da Experiência de Interpretação
    # -------------------------------------------------------------------------
    tab_a2a, tab_herg, tab_ad, tab_desc = st.tabs([
        "🎯 Receptor A2A (Potência)",
        "🫀 Canal hERG (Cardiotoxicidade)",
        "🌐 Domínio de Aplicabilidade (AD)",
        "⚖️ Quadro de Descritores Físico-Químicos",
    ])

    # =========================================================================
    # ABA 1: RECEPTOR A2A
    # =========================================================================
    with tab_a2a:
        st.markdown("### 🎯 Interpretação do Modelo $A_{2A}$ (Potência Antagonista)")
        st.caption(
            "Modelo XGBoost de Regressão. Saída explicada: pIC₅₀ predito. "
            "Contribuição positiva (+) eleva o pIC₅₀ (favorece afinidade); contribuição negativa (-) reduz o pIC₅₀."
        )

        # Métricas do modelo A2A
        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            st.metric("pIC₅₀ Predito", f"{pIC50:.3f}", help="Saída final do modelo supervisionado XGBoost.")
        with col_m2:
            st.metric("Valor Base E[f(X)]", f"{base_a2a:.3f}", help="Média esperada de predição sobre o conjunto de calibração.")
        with col_m3:
            diff_a2a = pIC50 - base_a2a
            st.metric("Deslocamento SHAP Total", f"{diff_a2a:+.3f}", help="Soma de todas as contribuições locais: ∑ ϕ_i.")

        render_html("<div style='margin-top:1rem;'></div>")

        if shap_a2a is not None and len(shap_a2a) == len(FEATURE_NAMES) and not np.all(shap_a2a == 0):
            ranked_a2a = get_ranked_explanations(
                shap_values=np.asarray(shap_a2a),
                target="A2A",
                mol=mol,
                bit_info=bit_info,
                feature_values=feature_vals_map,
                top_n=6,
            )

            filter_mode_a2a = st.radio(
                "Filtrar atributos por tipo de contribuição:",
                options=[
                    "⭐ Maior Magnitude (|SHAP|)",
                    "🟢 Contribuições Positivas (Elevam pIC₅₀)",
                    "🔴 Contribuições Negativas (Reduzem pIC₅₀)",
                ],
                horizontal=True,
                key="filter_radio_a2a",
            )

            if "Maior Magnitude" in filter_mode_a2a:
                cards_to_show = ranked_a2a["top_absolute"]
            elif "Positivas" in filter_mode_a2a:
                cards_to_show = ranked_a2a["top_positive"]
            else:
                cards_to_show = ranked_a2a["top_negative"]

            if not cards_to_show:
                st.info("Nenhum atributo encontrado para o filtro selecionado.")
            else:
                for idx_c, card in enumerate(cards_to_show):
                    _render_5_field_card(card, mol, bit_info, card_index=idx_c, target="A2A")

        else:
            st.info("ℹ️ Dados explicativos SHAP não disponíveis para o modelo A2A nesta análise.")

    # =========================================================================
    # ABA 2: CANAL hERG
    # =========================================================================
    with tab_herg:
        st.markdown("### 🫀 Interpretação do Modelo hERG (Escore do Classificador)")
        st.caption(
            "Modelo Random Forest de Classificação. Saída explicada: Escore do classificador para a Classe 1 (fração de estimadores para hERG positivo). Este valor reflete o escore do modelo in silico e não uma probabilidade clínica ou experimental calibrada de cardiotoxicidade. "
            "Contribuição positiva (+) eleva a saída do modelo associada à classe hERG positiva; "
            "contribuição negativa (-) reduz essa saída."
        )

        render_html("""
            <div style="background:rgba(255,255,255,0.02); border:1px solid rgba(255,255,255,0.08); border-radius:6px; padding:8px 12px; margin-bottom:10px;">
                <span style="font-size:0.75rem; color:#94A3B8; line-height:1.4; display:block;">
                    ⚠️ <b>Ressalva Metodológica e de Calibração:</b> O escore de saída do classificador representa a frequência de votos das árvores de decisão no espaço de treino e <b>não uma probabilidade clínica ou experimental calibrada de cardiotoxicidade</b>. Contribuições SHAP negativas representam reduções matemáticas na saída do classificador in silico. Elas <b>não constituem fatores biologicamente protetivos</b>, não garantem segurança cardíaca e não comprovam ausência de risco in vitro ou in vivo.
                </span>
            </div>
            """)

        # Métricas do modelo hERG
        col_h1, col_h2, col_h3 = st.columns(3)
        with col_h1:
            st.metric("Escore do Classificador (hERG=1)", f"{prob_herg:.3f}", help="Escore bruto de saída do modelo Random Forest (não calibrado clínica ou experimentalmente).")
        with col_h2:
            st.metric("Valor Base E[f(X)]", f"{base_herg:.3f}", help="Média esperada da classe positiva nos dados de treino.")
        with col_h3:
            diff_herg = prob_herg - base_herg
            st.metric("Deslocamento SHAP Total", f"{diff_herg:+.3f}", help="Concordância aditiva da soma das contribuições locais SHAP frente ao valor base.")

        render_html("<div style='margin-top:1rem;'></div>")

        if shap_herg is not None and len(shap_herg) == len(FEATURE_NAMES) and not np.all(shap_herg == 0):
            ranked_herg = get_ranked_explanations(
                shap_values=np.asarray(shap_herg),
                target="hERG",
                mol=mol,
                bit_info=bit_info,
                feature_values=feature_vals_map,
                top_n=6,
            )

            filter_mode_herg = st.radio(
                "Filtrar atributos por tipo de contribuição:",
                options=[
                    "⭐ Maior Magnitude (|SHAP|)",
                    "🔴 Contribuições Positivas (Elevam Saída hERG Positiva)",
                    "🟢 Contribuições Negativas (Reduzem Saída hERG Positiva)",
                ],
                horizontal=True,
                key="filter_radio_herg",
            )

            if "Maior Magnitude" in filter_mode_herg:
                cards_to_show_h = ranked_herg["top_absolute"]
            elif "Positivas" in filter_mode_herg:
                cards_to_show_h = ranked_herg["top_positive"]
            else:
                cards_to_show_h = ranked_herg["top_negative"]

            if not cards_to_show_h:
                st.info("Nenhum atributo encontrado para o filtro selecionado.")
            else:
                for idx_ch, card_h in enumerate(cards_to_show_h):
                    _render_5_field_card(card_h, mol, bit_info, card_index=idx_ch, target="hERG")

        else:
            st.info("ℹ️ Dados explicativos SHAP não disponíveis para o modelo hERG nesta análise.")

    # =========================================================================
    # ABA 3: DOMÍNIO DE APLICABILIDADE (AD)
    # =========================================================================
    with tab_ad:
        st.markdown("### 🌐 Diagnóstico de Cobertura do Domínio de Aplicabilidade (AD)")
        st.caption(
            "Análise didática da proximidade estrutural frente aos dados de calibração, "
            "sem recalcular diagnósticos e sem promessa de confiança estatística."
        )

        render_html("""
            <div style="background:#1E222D; border:1px solid #333B4D; border-radius:8px; padding:12px 16px; margin-bottom:1.2rem;">
                <div style="font-size:0.85rem; font-weight:700; color:#F8FAFC; margin-bottom:6px;">
                    📚 Guia Epistemológico dos Conceitos de Domínio
                </div>
                <div style="font-size:0.77rem; color:#CBD5E1; line-height:1.45;">
                    • <b>Correspondência Estrutural:</b> Identidade molecular exata (SMILES idêntico ou vetor de 1.030 atributos coincidente). Ocorre quando a molécula foi avaliada in vitro no conjunto de treino.<br/>
                    • <b>Similaridade Química (Tanimoto ECFP4):</b> Medida matemática de sobreposição de fragmentos circulares de raio 2. Não expressa probabilidade clínica.<br/>
                    • <b>Cobertura do Domínio:</b> Avaliação se a estrutura reside no espaço químico amostrado pelos dados experimentais do modelo.<br/>
                    • <b>Confiança Estatística:</b> A proximidade de Tanimoto <b>não constitui intervalo de confiança ou garantia de precisão</b>. Modelos QSAR são estimadores locais pontuais.<br/>
                    • <b>Ausência de Invalidação Automática:</b> Uma molécula em potencial extrapolação não é considerada inválida; o resultado sinaliza que a predição deve ser interpretada com cautela qualitativa.
                </div>
            </div>
            """)

        col_ad_a, col_ad_h = st.columns(2)

        exp_ad_a2a = explain_ad_coverage(ad_a2a, "A2A")
        exp_ad_herg = explain_ad_coverage(ad_herg, "hERG")

        with col_ad_a:
            st.markdown("#### 🎯 Cobertura do Modelo $A_{2A}$")
            if exp_ad_a2a["status"] == "ok":
                render_html(f"""
                    <div style="background:#181B22; border:1px solid #333B4D; border-radius:8px; padding:12px 14px; margin-bottom:10px;">
                        <div style="display:flex; justify-content:space-between; margin-bottom:6px;">
                            <span style="font-size:0.76rem; color:#94A3B8;">Proveniência:</span>
                            <span style="font-size:0.8rem; font-weight:700; color:#00D2BE;">Nível {exp_ad_a2a['provenance_level']}</span>
                        </div>
                        <div style="font-size:0.82rem; color:#FFFFFF; font-weight:600; margin-bottom:8px;">
                            {exp_ad_a2a['provenance_desc']}
                        </div>
                        <div style="font-size:0.76rem; color:#CBD5E1; line-height:1.4; margin-bottom:8px;">
                            • Similaridade Máxima Geral: <code>{exp_ad_a2a['s_max_all']:.4f}</code><br/>
                            • Similaridade Excluindo Auto-Match: <code>{f"{exp_ad_a2a['s_max_excluding_match']:.4f}" if exp_ad_a2a['s_max_excluding_match'] is not None else "Não aplicável"}</code><br/>
                            • Mediana de Referência LOO Treino: <code>{exp_ad_a2a['loo_median']:.4f}</code>
                        </div>
                        <div style="font-size:0.75rem; color:#94A3B8; border-top:1px solid #333B4D; padding-top:6px; margin-top:6px;">
                            {exp_ad_a2a['domain_interpretation']}
                        </div>
                    </div>
                    """)

                nn_a = exp_ad_a2a.get("nearest_neighbor") or {}
                if nn_a.get("canonical_smiles"):
                    st.markdown("**Vizinho Mais Próximo no Treino A2A:**")
                    st.caption(f"ID: {nn_a.get('id')} | Similaridade: {nn_a.get('similarity', 0.0):.4f}")
                    mol_nn_a = smiles_to_mol(nn_a["canonical_smiles"])
                    b64_nn_a = render_molecule_dark_base64(mol_nn_a, size=(260, 160)) if mol_nn_a else None
                    if b64_nn_a:
                        render_html(f"""
                            <div style="background:#181B22; border:1px solid #333B4D; border-radius:6px; padding:6px; text-align:center;">
                                <img src="data:image/png;base64,{b64_nn_a}" style="max-height:130px; width:auto;" alt="Vizinho A2A" />
                                <div style="font-size:0.7rem; color:#CBD5E1; margin-top:2px;">
                                    {nn_a.get('experimental_label_name')}: <b>{nn_a.get('experimental_value')}</b>
                                </div>
                            </div>
                            """)
            else:
                st.info(exp_ad_a2a["domain_interpretation"])

        with col_ad_h:
            st.markdown("#### 🫀 Cobertura do Modelo hERG")
            if exp_ad_herg["status"] == "ok":
                render_html(f"""
                    <div style="background:#181B22; border:1px solid #333B4D; border-radius:8px; padding:12px 14px; margin-bottom:10px;">
                        <div style="display:flex; justify-content:space-between; margin-bottom:6px;">
                            <span style="font-size:0.76rem; color:#94A3B8;">Proveniência:</span>
                            <span style="font-size:0.8rem; font-weight:700; color:#38BDF8;">Nível {exp_ad_herg['provenance_level']}</span>
                        </div>
                        <div style="font-size:0.82rem; color:#FFFFFF; font-weight:600; margin-bottom:8px;">
                            {exp_ad_herg['provenance_desc']}
                        </div>
                        <div style="font-size:0.76rem; color:#CBD5E1; line-height:1.4; margin-bottom:8px;">
                            • Similaridade Máxima Geral: <code>{exp_ad_herg['s_max_all']:.4f}</code><br/>
                            • Similaridade Excluindo Auto-Match: <code>{f"{exp_ad_herg['s_max_excluding_match']:.4f}" if exp_ad_herg['s_max_excluding_match'] is not None else "Não aplicável"}</code><br/>
                            • Mediana de Referência LOO Treino: <code>{exp_ad_herg['loo_median']:.4f}</code>
                        </div>
                        <div style="font-size:0.75rem; color:#94A3B8; border-top:1px solid #333B4D; padding-top:6px; margin-top:6px;">
                            {exp_ad_herg['domain_interpretation']}
                        </div>
                    </div>
                    """)

                nn_h = exp_ad_herg.get("nearest_neighbor") or {}
                if nn_h.get("canonical_smiles"):
                    st.markdown("**Vizinho Mais Próximo no Treino hERG:**")
                    st.caption(f"ID: {nn_h.get('id')} | Similaridade: {nn_h.get('similarity', 0.0):.4f}")
                    mol_nn_h = smiles_to_mol(nn_h["canonical_smiles"])
                    b64_nn_h = render_molecule_dark_base64(mol_nn_h, size=(260, 160)) if mol_nn_h else None
                    if b64_nn_h:
                        render_html(f"""
                            <div style="background:#181B22; border:1px solid #333B4D; border-radius:6px; padding:6px; text-align:center;">
                                <img src="data:image/png;base64,{b64_nn_h}" style="max-height:130px; width:auto;" alt="Vizinho hERG" />
                                <div style="font-size:0.7rem; color:#CBD5E1; margin-top:2px;">
                                    {nn_h.get('experimental_label_name')}: <b>{nn_h.get('experimental_value')}</b>
                                </div>
                            </div>
                            """)
            else:
                st.info(exp_ad_herg["domain_interpretation"])

    # =========================================================================
    # ABA 4: DESCRITORES FÍSICO-QUÍMICOS
    # =========================================================================
    with tab_desc:
        st.markdown("### ⚖️ Quadro Comparativo dos 6 Descritores Físico-Químicos")
        st.caption(
            "Análise multidimensional das propriedades físico-químicas globais, "
            "com valores calculados via RDKit e contribuições SHAP individuais para cada modelo."
        )

        desc_explanations = explain_physicochemical_descriptors(
            descriptors_dict=desc_dict,
            shap_a2a=np.asarray(shap_a2a) if shap_a2a is not None else None,
            shap_herg=np.asarray(shap_herg) if shap_herg is not None else None,
        )

        # Tabela consolidada
        table_rows = []
        for item in desc_explanations:
            table_rows.append({
                "Descritor": item["label"],
                "Código": item["name"],
                "Valor": f"{item['value']:.2f} {item['unit']}".strip(),
                "SHAP A2A (Potência)": f"{item['shap_a2a_str']} ({item['direction_a2a']})",
                "SHAP hERG (Cardiotox)": f"{item['shap_herg_str']} ({item['direction_herg']})",
            })

        df_desc = pd.DataFrame(table_rows)
        st.dataframe(df_desc, width="stretch", hide_index=True)

        render_html("<div style='margin-top:1rem;'></div>")
        st.markdown("#### 🔍 Detalhamento das Propriedades Físico-Químicas Globais")

        for d_item in desc_explanations:
            with st.expander(f"📌 {d_item['label']} ({d_item['name']}) = {d_item['value']:.2f} {d_item['unit']}", expanded=False):
                c_d1, c_d2 = st.columns(2)
                with c_d1:
                    st.markdown("**🎯 Impacto no Modelo A2A:**")
                    render_html(f"""
                        • Contribuição SHAP: <code>{d_item['shap_a2a_str']}</code> ({d_item['direction_a2a']})<br/>
                        • Racional Bibliográfico: {d_item['role_a2a']}
                        """)
                with c_d2:
                    st.markdown("**🫀 Impacto no Modelo hERG:**")
                    render_html(f"""
                        • Contribuição SHAP: <code>{d_item['shap_herg_str']}</code> ({d_item['direction_herg']})<br/>
                        • Racional Bibliográfico: {d_item['role_herg']}
                        """)
                st.caption(
                    "⚠️ Ressalva: Descritores físico-químicos são propriedades contínuas de toda a molécula. "
                    "A contribuição SHAP reflete a sensibilidade local do modelo e não constitui regra farmacológica universal."
                )

    # -------------------------------------------------------------------------
    # Rodapé de Continuidade e Navegação
    # -------------------------------------------------------------------------
    render_html("<div style='margin-top:2rem;'></div>")
    st.markdown("---")
    col_nav_1, col_nav_2, col_nav_3 = st.columns([1.5, 1.5, 2.0])

    with col_nav_1:
        st.page_link(
            get_page_target("resultados"),
            label="📊 Painel de Resultados",
            icon="🔙",
        )

    with col_nav_2:
        st.page_link(
            get_page_target("nova-analise"),
            label="🧪 Analisar Outra Molécula",
            icon="🔄",
        )

    with col_nav_3:
        st.page_link(
            get_page_target("fundamentos"),
            label="📚 Fundamentos Científicos",
            icon="🔬",
        )
