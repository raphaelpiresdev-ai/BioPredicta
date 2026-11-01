"""
Módulo da Página: Comparação Multimolecular (Etapa 5.3 — UX/UI Refactoring).

Permite ao usuário selecionar de 2 a 5 moléculas registradas no histórico da sessão,
definir uma molécula de referência para o cálculo de deltas farmacológicos e físico-químicos,
visualizar o posicionamento conjunto na matriz cartesiana 2D e comparar seus perfis
farmacológicos, de domínio de aplicabilidade e estruturais de maneira sinóptica.
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
from streamlit.runtime.scriptrunner_utils.script_run_context import get_script_run_ctx

from src.ad_utils import (
    PROVENANCE_LEVEL_1_STRUCTURE,
    PROVENANCE_LEVEL_2_FEATURES,
    PROVENANCE_LEVEL_3_FINGERPRINT,
    PROVENANCE_LEVEL_4_NO_EXACT_MATCH,
)
from src.chem_utils import smiles_to_mol
from src.ui.components import (
    build_comparative_multiobjective_figure,
    compute_comparative_deltas,
    compute_pairwise_tanimoto_matrix,
    get_page_target,
    reconstruct_mol_from_smiles,
    render_html,
    render_molecule_dark_base64,
    render_page_header,
)
from src.ui.history_manager import get_history, init_history, set_active_molecule
from src.ui.pages.p3_resultados import format_ic50, resolve_compound_display_name

logger = logging.getLogger(__name__)


def _render_ad_badge(ad_dict: Optional[Dict[str, Any]], endpoint: str) -> str:
    """Gera o HTML do badge independente de Domínio de Aplicabilidade para o endpoint."""
    if not ad_dict or ad_dict.get("status") != "ok":
        return f"<div style='font-size:0.78rem; color:#94A3B8;'><strong>{endpoint} AD:</strong> Não disponível</div>"

    level = ad_dict.get("provenance_level")
    s_max = ad_dict.get("s_max_all")
    s_max_str = f"S_max={s_max:.2f}" if s_max is not None else "S_max=N/A"

    if level == 1:
        color = "#00D2BE"
        desc = f"Nível 1 — {PROVENANCE_LEVEL_1_STRUCTURE}"
    elif level == 2:
        color = "#38BDF8"
        desc = f"Nível 2 — {PROVENANCE_LEVEL_2_FEATURES}"
    elif level == 3:
        color = "#F59E0B"
        desc = f"Nível 3 — {PROVENANCE_LEVEL_3_FINGERPRINT}"
    elif level == 4:
        color = "#CBD5E1"
        desc = f"Nível 4 — {PROVENANCE_LEVEL_4_NO_EXACT_MATCH}"
    else:
        color = "#94A3B8"
        desc = f"Nível {level}"

    return (
        f"<div style='font-size:0.78rem; color:#CBD5E1; margin-bottom:2px;'>"
        f"<strong style='color:#FFFFFF;'>{endpoint} AD:</strong> "
        f"<span style='background:rgba(255,255,255,0.06); border:1px solid {color}; color:{color}; "
        f"padding:1px 6px; border-radius:10px; font-weight:600;' title='{desc}'>{desc} ({s_max_str})</span>"
        f"</div>"
    )


def render_comparacao() -> None:
    """Renderiza a página principal de comparação multimolecular."""
    init_history()
    history = get_history()

    render_page_header(
        title="Comparação Multimolecular",
        subtitle="Avaliação visual, quantitativa e estrutural de 2 a 5 candidatos avaliados durante a sessão.",
        icon="⚖️",
    )

    # -------------------------------------------------------------------------
    # 1. Estado Vazio (< 2 moléculas no histórico)
    # -------------------------------------------------------------------------
    if len(history) < 2:
        st.session_state.pop("comp_multiselect_molecules", None)
        st.session_state.pop("comp_select_ref_molecule", None)
        st.session_state.pop("_comp_known_history_smiles", None)

        if len(history) == 0:
            msg_detail = (
                "Nenhuma análise molecular foi concluída nesta sessão. Para comparar candidatos, "
                "é necessário analisar pelo menos duas moléculas na página <b>Nova Análise</b>."
            )
        else:
            single_mol = history[0]
            name, _ = resolve_compound_display_name(single_mol)
            msg_detail = (
                f"Apenas uma análise molecular foi concluída nesta sessão (<b>{html.escape(str(name))}</b>). "
                "Para realizar comparações pareadas ou multiobjetivo, avalie pelo menos mais um composto "
                "na página <b>Nova Análise</b>."
            )

        render_html(
            f"""
            <div class="metric-card" style="border-left: 4px solid #F59E0B; padding: 1.5rem 1.8rem; margin-top: 1rem;">
                <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 0.6rem;">
                    <span style="font-size: 1.6rem;">⚖️</span>
                    <span style="font-size: 1.15rem; font-weight: 700; color: #FFFFFF;">
                        Comparação Multimolecular Indisponível (Mínimo de 2 Moléculas)
                    </span>
                </div>
                <p style="font-size: 0.90rem; color: #CBD5E1; line-height: 1.55; margin: 0 0 1.2rem 0;">
                    {msg_detail}
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
            st.caption("Você pode carregar controles como Cafeína e Istradefilina para testar a comparação imediatamente.")
        return

    # -------------------------------------------------------------------------
    # 2. Painel de Controle: Seleção de Candidatos e Molécula de Referência
    # -------------------------------------------------------------------------
    st.markdown("### 🎛️ Painel de Seleção e Referência")

    # Mapeamento para opções amigáveis
    smiles_to_mol_dict = {item["canonical_smiles"]: item for item in history}
    all_smiles = [item["canonical_smiles"] for item in history]
    _has_live_ctx = get_script_run_ctx() is not None

    def format_mol_option(smi: str) -> str:
        item = smiles_to_mol_dict.get(smi)
        if not item:
            return smi
        name, _ = resolve_compound_display_name(item)
        pIC50 = item.get("a2a", {}).get("pred_pIC50", 0.0)
        prob_herg = item.get("herg", {}).get("prob_herg", 0.0)
        return f"{name} (pIC50: {pIC50:.2f} | P(hERG=1): {prob_herg:.1%}) [{smi[:10]}...]"

    col_sel, col_ref = st.columns([3.2, 1.8])

    # Seleção padrão: até as primeiras 5 moléculas
    default_selected = all_smiles[: min(len(all_smiles), 5)]

    # Sincronização segura do estado de seleção com o histórico atual
    prev_known_smiles = st.session_state.get("_comp_known_history_smiles", [])
    if "comp_multiselect_molecules" not in st.session_state:
        st.session_state["comp_multiselect_molecules"] = list(default_selected)
    else:
        valid_current = [
            s for s in st.session_state["comp_multiselect_molecules"]
            if s in smiles_to_mol_dict
        ]
        newly_added = [s for s in all_smiles if s not in prev_known_smiles]
        for new_s in newly_added:
            if new_s not in valid_current and len(valid_current) < 5:
                valid_current.append(new_s)
        if not valid_current:
            valid_current = list(default_selected)
        st.session_state["comp_multiselect_molecules"] = valid_current[:5]
    st.session_state["_comp_known_history_smiles"] = list(all_smiles)

    _multi_kwargs: Dict[str, Any] = {
        "options": all_smiles,
        "format_func": format_mol_option,
        "max_selections": 5,
        "help": "Escolha entre 2 e 5 moléculas já inferidas para comparação simultânea.",
        "key": "comp_multiselect_molecules",
    }
    if not (_has_live_ctx and "comp_multiselect_molecules" in st.session_state):
        _multi_kwargs["default"] = st.session_state["comp_multiselect_molecules"]

    with col_sel:
        selected_smiles = st.multiselect(
            "Selecione de 2 a 5 moléculas para comparar:",
            **_multi_kwargs,
        )

    if len(selected_smiles) < 2:
        st.warning("⚠️ Selecione pelo menos 2 moléculas acima para visualizar a comparação multiobjetivo.")
        return

    # Moléculas selecionadas
    selected_mols = [smiles_to_mol_dict[smi] for smi in selected_smiles if smi in smiles_to_mol_dict]

    # Molécula de referência para deltas (garantindo validade em session_state)
    active_smi = st.session_state.get("current_analysis_smiles")
    default_ref_idx = 0
    if active_smi and active_smi in selected_smiles:
        default_ref_idx = selected_smiles.index(active_smi)

    if (
        "comp_select_ref_molecule" not in st.session_state
        or st.session_state.get("comp_select_ref_molecule") not in selected_smiles
    ):
        st.session_state["comp_select_ref_molecule"] = selected_smiles[default_ref_idx]

    _ref_kwargs: Dict[str, Any] = {
        "options": selected_smiles,
        "format_func": format_mol_option,
        "help": "Define o composto base frente ao qual serão calculadas as diferenças (Δ = Molécula - Referência).",
        "key": "comp_select_ref_molecule",
    }
    if not (_has_live_ctx and "comp_select_ref_molecule" in st.session_state):
        _ref_kwargs["index"] = selected_smiles.index(st.session_state["comp_select_ref_molecule"])

    with col_ref:
        ref_smiles = st.selectbox(
            "Molécula de Referência (Base para Deltas Δ):",
            **_ref_kwargs,
        )

    # -------------------------------------------------------------------------
    # 3. Gráfico Cartesiano Multiobjetivo Multiponto (A2A × hERG)
    # -------------------------------------------------------------------------
    st.markdown("---")
    st.markdown("### 🎯 Posicionamento Multiobjetivo Conjunto")
    st.caption(
        "Dispersão espacial das moléculas selecionadas sobre as 4 zonas de decisão canônicas. "
        "A molécula de referência é realçada com anel dourado e rótulo [REF]."
    )

    fig_mat, _ = build_comparative_multiobjective_figure(selected_mols, ref_smiles=ref_smiles)
    st.pyplot(fig_mat, width="stretch")
    plt.close(fig_mat)

    render_html(
        """
        <div style="background:#1E222D; border:1px solid #333B4D; border-radius:8px; padding:10px 14px; font-size:0.83rem; color:#94A3B8; margin-top:6px;">
            <strong style="color:#00D2BE;">Nota Científica:</strong> O eixo vertical representa o <em>Índice Complementar</em> <code>1 - P(hERG)</code>,
            derivado da saída probabilística do classificador para a classe positiva (<code>P(hERG=1)</code>).
            Esse escore expressa a propensão estimada pelo modelo supervisionado a partir dos descritores e fingerprints moleculares e
            <strong>não</strong> constitui probabilidade clínica calibrada de cardiotoxicidade nem estimativa direta de incidência de eventos adversos cardíacos.
        </div>
        """
    )

    # -------------------------------------------------------------------------
    # 4. Quadro Sinóptico Estrutural e Farmacológico (Grade Lado a Lado)
    # -------------------------------------------------------------------------
    st.markdown("---")
    st.markdown("### 🧬 Comparação Estrutural e Farmacológica Individual")

    cols = st.columns(len(selected_mols))
    palette = ["#00D2BE", "#F59E0B", "#A855F7", "#38BDF8", "#EC4899"]

    for idx, item in enumerate(selected_mols):
        smi = item.get("canonical_smiles", "")
        name, _ = resolve_compound_display_name(item)
        short_name = name[:16] + "..." if len(name) > 18 else name
        short_name_esc = html.escape(str(short_name))
        name_esc = html.escape(str(name))
        is_ref = (smi == ref_smiles)
        marker_color = palette[idx % len(palette)]

        # Dados estruturais e moleculares
        mol = item.get("mol")
        if mol is None and smi:
            mol = reconstruct_mol_from_smiles(smi)

        formula = item.get("formula", "")
        if not formula and mol:
            formula = rdMolDescriptors.CalcMolFormula(mol)
        formula_esc = html.escape(str(formula))

        mol_wt = item.get("mol_wt", 0.0)
        if (mol_wt is None or mol_wt <= 0) and mol:
            mol_wt = float(Descriptors.MolWt(mol))

        img_b64 = render_molecule_dark_base64(mol, size=(280, 160)) if mol else None

        # Dados A2A
        a2a = item.get("a2a") or {}
        pIC50 = float(a2a.get("pred_pIC50", item.get("x", 0.0)))
        ic50_nm = float(a2a.get("pred_IC50_nM", 10.0 ** (9.0 - pIC50) if pIC50 > 0 else 0.0))
        ic50_str = format_ic50(ic50_nm)
        a2a_cls = a2a.get("classification") if isinstance(a2a.get("classification"), dict) else {}
        a2a_color = a2a_cls.get("color", "#00D2BE" if pIC50 >= 6.5 else ("#F59E0B" if pIC50 >= 5.0 else "#94A3B8"))

        # Dados hERG
        herg = item.get("herg") or {}
        prob_herg = float(herg.get("prob_herg", 1.0 - float(item.get("y", 0.0)) if "y" in item else 0.0))
        safety_idx = float(herg.get("safety_index", 1.0 - prob_herg))
        herg_cls = herg.get("classification") if isinstance(herg.get("classification"), dict) else {}
        herg_color = (
            herg_cls.get("bar_color")
            or herg_cls.get("color")
            or ("#00D2BE" if prob_herg < 0.30 else ("#F59E0B" if prob_herg <= 0.70 else "#FF4B4B"))
        )

        # Quadrante Multiobjetivo
        quad_info = item.get("multiobjective", {}).get("quadrant") or {}
        quad_title = quad_info.get("quad_title", "Multiobjetivo") if isinstance(quad_info, dict) else str(quad_info)
        quad_title_esc = html.escape(str(quad_title))

        # AD
        a2a_ad = a2a.get("ad")
        herg_ad = herg.get("ad")

        with cols[idx]:
            ref_badge = (
                "<span style='background:#FBBF24; color:#0E1117; font-weight:700; font-size:0.68rem; "
                "padding:2px 6px; border-radius:4px;'>REFERÊNCIA</span>"
                if is_ref
                else ""
            )
            img_block = (
                f'<div style="background:#131720; border-radius:8px; padding:6px; text-align:center; margin-bottom:8px; border:1px solid #2A303F;">'
                f'<img src="data:image/png;base64,{img_b64}" style="max-width:100%; height:auto; max-height:130px; object-fit:contain;" alt="{name_esc}" />'
                f'</div>'
                if img_b64
                else ""
            )

            render_html(
                f"""
                <div class="surface-n2" style="border: 1px solid {'#FBBF24' if is_ref else '#3E4759'}; border-radius: 10px; padding: 12px; margin-bottom: 10px;">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;">
                        <span style="font-weight:700; color:{marker_color}; font-size:0.95rem;">#{idx+1} {short_name_esc}</span>
                        {ref_badge}
                    </div>
                    <div style="font-size:0.75rem; color:#94A3B8; margin-bottom:8px;">
                        {formula_esc} • {mol_wt:.1f} Da
                    </div>
                    {img_block}
                    <div style="display:flex; flex-direction:column; gap:5px; font-size:0.80rem;">
                        <div style="background:#1E222D; padding:5px 8px; border-radius:6px; border-left:3px solid {a2a_color};">
                            <div style="display:flex; justify-content:space-between;">
                                <span style="color:#CBD5E1;">A2A pIC50:</span>
                                <strong style="color:#FFFFFF;">{pIC50:.2f}</strong>
                            </div>
                            <div style="display:flex; justify-content:space-between; font-size:0.75rem; color:#94A3B8;">
                                <span>IC50 est.:</span>
                                <span>{ic50_str}</span>
                            </div>
                        </div>
                        <div style="background:#1E222D; padding:5px 8px; border-radius:6px; border-left:3px solid {herg_color};">
                            <div style="display:flex; justify-content:space-between;">
                                <span style="color:#CBD5E1;" title="Saída probabilística do classificador para a classe hERG positiva">Saída P(hERG=1):</span>
                                <strong style="color:#FFFFFF;">{prob_herg:.1%}</strong>
                            </div>
                            <div style="display:flex; justify-content:space-between; font-size:0.75rem; color:#94A3B8;">
                                <span>Índice 1 - P:</span>
                                <span>{safety_idx:.1%}</span>
                            </div>
                        </div>
                        <div style="background:#1E222D; padding:5px 8px; border-radius:6px; border-left:3px solid #CBD5E1;">
                            <span style="color:#94A3B8; font-size:0.72rem;">Posicionamento:</span>
                            <div style="font-weight:600; color:#FFFFFF; font-size:0.78rem;">{quad_title_esc}</div>
                        </div>
                    </div>
                    <div style="margin-top:8px; padding-top:6px; border-top:1px solid #333B4D;">
                        {_render_ad_badge(a2a_ad, 'A2A')}
                        {_render_ad_badge(herg_ad, 'hERG')}
                    </div>
                </div>
                """
            )

            # Ação voluntária de ativar no painel (não altera current_analysis sem clique explícito)
            if active_smi != smi:
                if st.button(
                    "📌 Tornar Ativa",
                    key=f"btn_activate_comp_{idx}",
                    width="stretch",
                    help="Define como molécula ativa para inspeção de SHAP e abas individuais",
                ):
                    set_active_molecule(smi)
                    st.session_state["input_smiles_field"] = smi
                    st.session_state["_pending_smiles_update"] = smi
                    st.rerun()

    # -------------------------------------------------------------------------
    # 5. Matriz de Similaridade Estrutural (Tanimoto ECFP4)
    # -------------------------------------------------------------------------
    st.markdown("---")
    st.markdown("### 📐 Matriz de Similaridade Estrutural (Tanimoto Morgan ECFP4)")
    st.caption(
        "Coeficientes de similaridade par-a-par calculados via Tanimoto sobre fingerprints Morgan (ECFP4, raio 2, 1024 bits). "
        "A diagonal é estritamente 1,000 (auto-similaridade). Valores superiores a 0,70 indicam alta similaridade estrutural "
        "baseada na sobreposição de ambientes circulares locais, sem implicar necessariamente conservação ou identidade de scaffold central."
    )

    labels, sim_matrix = compute_pairwise_tanimoto_matrix(selected_mols)
    col_labels = [f"#{i+1} {labels[i]}" for i in range(len(labels))]
    df_sim = pd.DataFrame(sim_matrix, index=col_labels, columns=col_labels)

    st.dataframe(
        df_sim.style.format("{:.3f}").background_gradient(cmap="Blues", vmin=0.0, vmax=1.0),
        width="stretch",
    )

    # -------------------------------------------------------------------------
    # 6. Perfil Físico-Químico e Tabela de Deltas (Δ)
    # -------------------------------------------------------------------------
    st.markdown("---")
    st.markdown("### 📊 Propriedades Físico-Químicas e Deltas Relativos à Referência")
    st.caption(
        f"Comparações quantitativas (Δ = Composto - Referência) calculadas frente a: "
        f"<b>{html.escape(format_mol_option(ref_smiles))}</b>. "
        "O escore <code>P(hERG=1)</code> expressa a saída probabilística do classificador para a classe positiva e não representa probabilidade clínica de cardiotoxicidade.",
        unsafe_allow_html=True,
    )

    deltas_data = compute_comparative_deltas(selected_mols, ref_smiles=ref_smiles)
    table_rows = []

    for d in deltas_data:
        name_display = f"#{d['index']+1} {d['compound_name']}" + (" [REF]" if d["is_reference"] else "")
        v = d["values"]
        dt = d["deltas"]

        dp_str = f"{dt['delta_pIC50']:+.2f}" if not d["is_reference"] else "— (Ref)"
        ric50_str = f"{dt['ratio_IC50']:.2f}x" if not d["is_reference"] else "1.00x"
        dprob_str = f"{dt['delta_prob_herg']:+.1%}" if not d["is_reference"] else "— (Ref)"
        dsafety_str = f"{dt['delta_safety_index']:+.1%}" if not d["is_reference"] else "— (Ref)"
        dwt_str = f"{dt['delta_MolWt']:+.1f}" if not d["is_reference"] else "— (Ref)"
        dlogp_str = f"{dt['delta_MolLogP']:+.2f}" if not d["is_reference"] else "— (Ref)"
        dhbd_str = f"{dt['delta_NumHDonors']:+d}" if not d["is_reference"] else "— (Ref)"
        dhba_str = f"{dt['delta_NumHAcceptors']:+d}" if not d["is_reference"] else "— (Ref)"
        dtpsa_str = f"{dt['delta_TPSA']:+.1f}" if not d["is_reference"] else "— (Ref)"
        drot_str = f"{dt['delta_NumRotatableBonds']:+d}" if not d["is_reference"] else "— (Ref)"

        table_rows.append({
            "Candidato": name_display,
            "pIC50 A2A": f"{v['pIC50']:.2f}",
            "Δ pIC50": dp_str,
            "IC50 (nM)": f"{v['pred_IC50_nM']:.1f}",
            "Razão IC50": ric50_str,
            "P(hERG=1)": f"{v['prob_herg']:.1%}",
            "Δ P(hERG)": dprob_str,
            "1 - P(hERG)": f"{v['safety_index']:.1%}",
            "Δ (1 - P)": dsafety_str,
            "MolWt (Da)": f"{v['MolWt']:.1f}",
            "Δ MolWt": dwt_str,
            "LogP": f"{v['MolLogP']:.2f}",
            "Δ LogP": dlogp_str,
            "HBD": f"{v['NumHDonors']}",
            "Δ HBD": dhbd_str,
            "HBA": f"{v['NumHAcceptors']}",
            "Δ HBA": dhba_str,
            "TPSA (Å²)": f"{v['TPSA']:.1f}",
            "Δ TPSA": dtpsa_str,
            "RotBonds": f"{v['NumRotatableBonds']}",
            "Δ Rot": drot_str,
        })

    df_deltas = pd.DataFrame(table_rows)
    st.dataframe(df_deltas, width="stretch")

    render_html(
        """
        <div style="background:#1E222D; border:1px solid #333B4D; border-radius:8px; padding:10px 14px; font-size:0.82rem; color:#94A3B8; margin-top:8px;">
            <strong style="color:#FFFFFF;">Diretrizes de Druglikeness (Regra dos 5 de Lipinski &amp; Veber):</strong>
            Massa Molecular &le; 500 Da • LogP &le; 5,0 • Doadores de H (HBD) &le; 5 • Aceptores de H (HBA) &le; 10 •
            Área Polar Topológica (TPSA) &le; 140 Å² • Ligações Rotacionáveis &le; 10.
        </div>
        """
    )
