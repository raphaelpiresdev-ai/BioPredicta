"""
Componentes Reutilizáveis de Interface — MultiTarget BioPredict.

Fornece elementos compartilhados de UI (cabeçalhos, cartões e marcadores visuais)
utilizando o Design System centralizado (src.ui.theme).
"""

from __future__ import annotations

import base64
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Draw import rdMolDraw2D
import streamlit as st

from src.chem_utils import (
    DESCRIPTOR_NAMES,
    FEATURE_NAMES,
    canonicalize_smiles,
    compute_2d_descriptors,
    smiles_to_mol,
)
from src.ui.catalogs import resolve_translated_feature_name

__all__ = [
    "render_html",
    "render_molecule_dark_base64",
    "render_page_header",
    "render_provisional_page",
    "build_multiobjective_figure",
    "build_comparative_multiobjective_figure",
    "build_shap_impact_figure",
    "reconstruct_mol_from_smiles",
    "get_morgan_fp_bitvect",
    "compute_pairwise_tanimoto_matrix",
    "compute_comparative_deltas",
    "get_page_target",
]


def get_page_target(key: str) -> Any:
    """Recupera o objeto st.Page registrado em st.session_state['pages_registry'].

    Caso a página seja renderizada fora do contexto de app.py (ex.: testes unitários isolados),
    retorna a chave de string como fallback seguro.
    """
    registry = st.session_state.get("pages_registry", {})
    return registry.get(key, key)


def render_html(html_code: str) -> None:
    """Renderiza snippet HTML direto com suporte a CSS customizado, removendo indentação espúria."""
    clean_html = "\n".join(line.strip() for line in str(html_code).strip().splitlines() if line.strip())
    st.markdown(clean_html, unsafe_allow_html=True)


def render_molecule_dark_base64(mol: Chem.Mol, size: Tuple[int, int] = (420, 260)) -> Optional[str]:
    """Renderiza a molécula 2D com fundo transparente e paleta Dark Mode via MolDraw2DCairo."""
    try:
        drawer = rdMolDraw2D.MolDraw2DCairo(size[0], size[1])
        opts = drawer.drawOptions()
        rdMolDraw2D.SetDarkMode(opts)
        opts.clearBackground = False
        opts.bondLineWidth = 2.2
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        return base64.b64encode(drawer.GetDrawingText()).decode("utf-8")
    except Exception:
        return None


def render_page_header(title: str, subtitle: str, icon: str = "🧪") -> None:
    """Renderiza o cabeçalho padronizado da página no estilo Hero Banner."""
    render_html(
        f"""
        <div class="hero-banner">
            <div class="main-title">{icon} {title}</div>
            <p class="sub-title">{subtitle}</p>
        </div>
        """
    )


def render_provisional_page(
    page_name: str,
    scheduled_stage: str,
    description: str,
    icon: str = "🚧",
) -> None:
    """Renderiza uma página provisória informativa para navegação estrutural preliminar."""
    render_html(
        f"""
        <div class="hero-banner">
            <span class="provisional-badge">Módulo Provisório • {scheduled_stage}</span>
            <div class="main-title">{icon} {page_name}</div>
            <p class="sub-title">{description}</p>
        </div>
        """
    )

    st.info(
        f"**Arquitetura de Navegação Ativa:** A rota para **{page_name}** foi configurada com sucesso "
        f"na estrutura multipágina. O conteúdo funcional completo desta página será implementado na "
        f"**{scheduled_stage}**, conforme definido no cronograma técnico de UX/UI (`PLANEJAMENTO_UX_UI_FASE_1.md`)."
    )


def build_multiobjective_figure(
    x_cur: float,
    y_cur: float,
    active_name: str,
) -> Tuple[plt.Figure, plt.Axes]:
    """Constrói o gráfico multiobjetivo cartesiano (X: pIC50, Y: 1 - P_hERG) em coordenadas estritas de dados."""
    fig_mat, ax_mat = plt.subplots(figsize=(10.5, 5.2), facecolor="#0E1117")
    ax_mat.set_facecolor("#1E222D")

    # Linhas de referência delimitando as faixas operacionais canônicas
    ax_mat.axvline(
        5.0,
        color="#94A3B8",
        linestyle=":",
        linewidth=1.1,
        alpha=0.80,
        label="Limiar Potência Moderada A2A (pIC50 = 5,0)",
    )
    ax_mat.axvline(
        6.5,
        color="#00D2BE",
        linestyle="--",
        linewidth=1.3,
        alpha=0.85,
        label="Limiar Alta Potência A2A (pIC50 = 6,5)",
    )
    ax_mat.axhline(
        0.70,
        color="#00D2BE",
        linestyle=":",
        linewidth=1.3,
        alpha=0.85,
        label="Limiar Operacional Baixo Risco hERG (1 - P = 0,70 | P = 0,30)",
    )
    ax_mat.axhline(
        0.30,
        color="#FF4B4B",
        linestyle=":",
        linewidth=1.1,
        alpha=0.80,
        label="Limiar Operacional Alto Risco hERG (1 - P = 0,30 | P = 0,70)",
    )

    # Sombreamentos dos quadrantes operacionais
    ax_mat.fill_between([6.5, 11.5], 0.70, 1.00, color="#00D2BE", alpha=0.14)
    ax_mat.fill_between([6.5, 11.5], 0.00, 0.30, color="#FF4B4B", alpha=0.14)
    ax_mat.fill_between([3.5, 5.0], 0.00, 0.30, color="#FF4B4B", alpha=0.08)
    ax_mat.fill_between([3.5, 5.0], 0.70, 1.00, color="#94A3B8", alpha=0.08)

    ax_mat.text(9.0, 0.95, "QUADRANTE 1: ZONA IDEAL\nAlta Potência & Baixo Risco", ha="center", va="top", fontsize=9.2, fontweight="bold", color="#00F5D4")
    ax_mat.text(9.0, 0.05, "QUADRANTE 4: ALERTA CRÍTICO\nPotente porém Alto Risco hERG", ha="center", va="bottom", fontsize=9.2, fontweight="bold", color="#FF7676")
    ax_mat.text(4.25, 0.95, "QUADRANTE 2: BAIXO INTERESSE\nInativo & Baixo Risco hERG", ha="center", va="top", fontsize=8.5, fontweight="bold", color="#CBD5E1")
    ax_mat.text(4.25, 0.05, "QUADRANTE 3: DESCARTE\nInativo & Alto Risco hERG", ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#94A3B8")

    # Marcador do composto analisado
    ax_mat.scatter(x_cur, y_cur, s=320, color="#00D2BE", edgecolors="#FFFFFF", linewidths=2.5, zorder=6)

    offset_x = -2.2 if x_cur > 9.2 else 0.22
    offset_y = -0.14 if y_cur > 0.88 else 0.04
    ax_mat.annotate(
        f"{active_name}\n(pIC50={x_cur:.2f} | 1-P={y_cur:.1%})",
        (x_cur + offset_x, y_cur + offset_y),
        fontsize=9.2,
        fontweight="bold",
        color="#FFFFFF",
        bbox=dict(boxstyle="round,pad=0.35", fc="#0E1117", ec="#00D2BE", lw=1.5, alpha=0.95),
    )

    ax_mat.set_xlabel("Potência Predita A2A: pIC50 (Maior = Maior Potência Estimada)", fontsize=10, fontweight="bold", color="#FFFFFF")
    ax_mat.set_ylabel("Índice Complementar Derivado: 1 - P(hERG) [Escore do Modelo]", fontsize=10, fontweight="bold", color="#FFFFFF")
    ax_mat.set_title("Posicionamento Cartesiano Multiobjetivo (A2A × hERG)", fontsize=11.5, fontweight="bold", color="#FFFFFF")
    ax_mat.set_xlim(3.5, 11.5)
    ax_mat.set_ylim(-0.05, 1.05)
    ax_mat.tick_params(colors="#E2E8F0", labelsize=9)
    for spine in ax_mat.spines.values():
        spine.set_color("#475569")
    ax_mat.grid(True, linestyle=":", alpha=0.35, color="#64748B")

    plt.tight_layout()
    return fig_mat, ax_mat


def build_shap_impact_figure(
    shap_values: np.ndarray,
    base_value: float,
    pred_value: float,
    target: str = "A2A",
    top_n: int = 10,
    mol: Optional[Chem.Mol] = None,
) -> Tuple[Optional[plt.Figure], Optional[plt.Axes], List[Dict[str, Any]]]:
    """
    Constrói o gráfico de barras horizontais em Dark Mode para os atributos de maior impacto SHAP.
    Retorna (fig, ax, top_items) ou (None, None, []) caso não haja valores disponíveis.
    """
    if shap_values is None or len(shap_values) == 0 or np.all(shap_values == 0):
        return None, None, []

    top_idx = np.argsort(np.abs(shap_values))[::-1][:top_n]
    top_names = [FEATURE_NAMES[i] for i in top_idx]
    top_impacts = [float(shap_values[i]) for i in top_idx]
    translated_names = [resolve_translated_feature_name(fn, mol=mol) for fn in top_names]

    fig, ax = plt.subplots(figsize=(10.5, 4.6), facecolor="#0E1117")
    ax.set_facecolor("#1E222D")

    if target == "A2A":
        bar_colors = ["#00D2BE" if imp >= 0 else "#FF4B4B" for imp in top_impacts]
        xlabel = "Impacto Aditivo no pIC50 (Ciano = Eleva Afinidade A2A | Vermelho = Atenua)"
        title = f"Decomposição SHAP — Top {top_n} Atributos A2A (Linha de Base = {base_value:.2f} → Predito = {pred_value:.2f})"
    else:
        bar_colors = ["#FF4B4B" if imp >= 0 else "#00D2BE" for imp in top_impacts]
        xlabel = "Impacto Aditivo em P(hERG) (Vermelho = Eleva Risco de Bloqueio | Ciano = Efeito Protetor)"
        title = f"Decomposição SHAP — Top {top_n} Atributos hERG (Linha de Base = {base_value:.1%} → Predito = {pred_value:.1%})"

    y_pos = np.arange(len(top_names))
    ax.barh(y_pos, top_impacts, color=bar_colors, edgecolor="#FFFFFF", linewidth=0.6, alpha=0.92)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(translated_names, fontsize=9.2, fontweight="bold", color="#FFFFFF")
    ax.invert_yaxis()
    ax.axvline(0, color="#CBD5E1", linestyle="-", linewidth=1.1)
    ax.set_xlabel(xlabel, fontsize=9.5, fontweight="bold", color="#FFFFFF")
    ax.set_title(title, fontsize=11, fontweight="bold", color="#FFFFFF")
    ax.tick_params(colors="#E2E8F0", labelsize=9)
    for spine in ax.spines.values():
        spine.set_color("#475569")
    ax.grid(axis="x", linestyle=":", alpha=0.35, color="#64748B")
    plt.tight_layout()

    top_items = []
    for fn, tr, imp, idx in zip(top_names, translated_names, top_impacts, top_idx):
        top_items.append({
            "feature_name": fn,
            "translated_name": tr,
            "impact": imp,
            "feature_index": int(idx),
        })

    return fig, ax, top_items


# -----------------------------------------------------------------------------
# Funções Auxiliares de Comparação Multimolecular e Quimioinformática
# -----------------------------------------------------------------------------
def reconstruct_mol_from_smiles(smiles: str) -> Optional[Chem.Mol]:
    """Reconstrói uma molécula RDKit padronizada a partir de uma string SMILES.

    Retorna None caso o SMILES seja nulo, vazio ou quimicamente inválido.
    """
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    try:
        mol = smiles_to_mol(smiles.strip(), standardize=True)
        if mol is not None and mol.GetNumAtoms() > 0:
            return mol
        canon = canonicalize_smiles(smiles.strip())
        if canon:
            mol = Chem.MolFromSmiles(canon)
            if mol is not None and mol.GetNumAtoms() > 0:
                return mol
    except Exception:
        return None
    return None


@lru_cache(maxsize=128)
def _get_cached_morgan_generator(radius: int, n_bits: int):
    """Gerador Morgan ECFP4 reutilizável (API moderna rdFingerprintGenerator)."""
    return rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)


def get_morgan_fp_bitvect(
    mol: Union[Chem.Mol, str],
    radius: int = 2,
    n_bits: int = 1024,
) -> Optional[DataStructs.ExplicitBitVect]:
    """Gera o fingerprint circular Morgan ECFP4 (ExplicitBitVect) para cálculo de Tanimoto.

    Args:
        mol: Chem.Mol ou string SMILES.
        radius: Raio circular Morgan (padrão 2 para ECFP4).
        n_bits: Comprimento do vetor binário (padrão 1024 bits).

    Returns:
        ExplicitBitVect do RDKit ou None se a molécula for inválida.
    """
    if mol is None:
        return None
    mol_obj = mol if isinstance(mol, Chem.Mol) else reconstruct_mol_from_smiles(mol)
    if mol_obj is None:
        return None
    try:
        gen = _get_cached_morgan_generator(radius, n_bits)
        return gen.GetFingerprint(mol_obj)
    except Exception:
        return None


def compute_pairwise_tanimoto_matrix(
    items: List[Union[str, Chem.Mol, Dict[str, Any]]],
    radius: int = 2,
    n_bits: int = 1024,
) -> Tuple[List[str], np.ndarray]:
    """Calcula a matriz quadrada simétrica de similaridade Tanimoto (ECFP4) entre os compostos.

    Garante:
    - Simetria estrita: M[i, j] == M[j, i] para qualquer par (i, j).
    - Diagonal unitária: M[i, i] == 1.0 para todo composto válido.
    - Valores delimitados no intervalo contínuo [0.0, 1.0].

    Args:
        items: Lista de SMILES (str), Chem.Mol ou registros de análise (dict).
        radius: Raio Morgan (2 para ECFP4).
        n_bits: Número de bits (1024).

    Returns:
        Tuple contendo (labels_list, similarity_matrix_ndarray de shape (N, N)).
    """
    if not items:
        return [], np.zeros((0, 0), dtype=float)

    labels: List[str] = []
    fps: List[Optional[DataStructs.ExplicitBitVect]] = []

    for idx, it in enumerate(items):
        if isinstance(it, dict):
            smi = it.get("canonical_smiles") or it.get("raw_input") or ""
            name = (
                it.get("provenance", {}).get("compound_name")
                or it.get("pubchem", {}).get("title")
                or it.get("provenance", {}).get("name")
                or it.get("compound_name")
                or (smi[:14] + "..." if len(smi) > 14 else smi)
                or f"Molécula #{idx + 1}"
            )
            labels.append(name)
            mol = it.get("mol")
            if mol is None and smi:
                mol = reconstruct_mol_from_smiles(smi)
            fp = get_morgan_fp_bitvect(mol, radius=radius, n_bits=n_bits) if mol else None
            fps.append(fp)
        elif isinstance(it, Chem.Mol):
            labels.append(f"Molécula #{idx + 1}")
            fps.append(get_morgan_fp_bitvect(it, radius=radius, n_bits=n_bits))
        elif isinstance(it, str):
            labels.append(it[:16] + "..." if len(it) > 16 else it)
            fps.append(get_morgan_fp_bitvect(it, radius=radius, n_bits=n_bits))
        else:
            labels.append(f"Molécula #{idx + 1}")
            fps.append(None)

    n = len(items)
    matrix = np.zeros((n, n), dtype=float)

    for i in range(n):
        fp_i = fps[i]
        for j in range(i, n):
            if i == j:
                matrix[i, j] = 1.0 if fp_i is not None else 0.0
            else:
                fp_j = fps[j]
                if fp_i is not None and fp_j is not None:
                    sim = float(DataStructs.TanimotoSimilarity(fp_i, fp_j))
                else:
                    sim = 0.0
                matrix[i, j] = sim
                matrix[j, i] = sim

    return labels, matrix


def _extract_mol_properties(item: Dict[str, Any]) -> Dict[str, Any]:
    """Extrai com segurança e resiliência os valores farmacológicos e descritores de um registro."""
    smi = item.get("canonical_smiles") or item.get("raw_input") or ""
    name = (
        item.get("provenance", {}).get("compound_name")
        or item.get("pubchem", {}).get("title")
        or item.get("provenance", {}).get("name")
        or item.get("compound_name")
        or (smi[:14] + "..." if len(smi) > 14 else smi)
        or "Composto"
    )

    # 1. Farmacologia A2A
    a2a_dict = item.get("a2a") or {}
    pIC50 = a2a_dict.get("pred_pIC50")
    if pIC50 is None:
        pIC50 = item.get("pred_pIC50") or item.get("x") or 0.0
    pIC50 = float(pIC50)

    ic50_nM = a2a_dict.get("pred_IC50_nM")
    if ic50_nM is None:
        ic50_nM = item.get("pred_IC50_nM")
        if ic50_nM is None and pIC50 > 0:
            ic50_nM = 10.0 ** (9.0 - pIC50)
        else:
            ic50_nM = float(ic50_nM or 0.0)
    ic50_nM = float(ic50_nM)

    # 2. Toxicidade hERG
    herg_dict = item.get("herg") or {}
    prob_herg = herg_dict.get("prob_herg")
    if prob_herg is None:
        prob_herg = item.get("prob_herg")
        if prob_herg is None and "y" in item:
            prob_herg = 1.0 - float(item["y"])
        else:
            prob_herg = float(prob_herg or 0.0)
    prob_herg = float(prob_herg)

    safety_idx = herg_dict.get("safety_index")
    if safety_idx is None:
        safety_idx = item.get("safety_index") or (1.0 - prob_herg)
    safety_idx = float(safety_idx)

    # 3. Descritores 2D
    raw_desc = item.get("descriptors") or {}
    descriptors: Dict[str, Any] = {}
    standard_descriptors = ["MolWt", "MolLogP", "NumHDonors", "NumHAcceptors", "TPSA", "NumRotatableBonds"]

    missing_desc = any(k not in raw_desc for k in standard_descriptors)
    if missing_desc and smi:
        try:
            mol_temp = reconstruct_mol_from_smiles(smi)
            if mol_temp is not None:
                calc_desc = compute_2d_descriptors(mol_temp)
                raw_desc = {**calc_desc, **raw_desc}
        except Exception:
            pass

    descriptors["MolWt"] = float(raw_desc.get("MolWt", item.get("mol_wt", 0.0)))
    descriptors["MolLogP"] = float(raw_desc.get("MolLogP", 0.0))
    descriptors["NumHDonors"] = int(raw_desc.get("NumHDonors", 0))
    descriptors["NumHAcceptors"] = int(raw_desc.get("NumHAcceptors", 0))
    descriptors["TPSA"] = float(raw_desc.get("TPSA", 0.0))
    descriptors["NumRotatableBonds"] = int(raw_desc.get("NumRotatableBonds", 0))

    return {
        "canonical_smiles": smi,
        "compound_name": name,
        "pIC50": pIC50,
        "pred_IC50_nM": ic50_nM,
        "prob_herg": prob_herg,
        "safety_index": safety_idx,
        "descriptors": descriptors,
    }


def compute_comparative_deltas(
    molecules_list: List[Dict[str, Any]],
    ref_index: int = 0,
    ref_smiles: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Calcula os deltas farmacológicos e físico-químicos das moléculas frente à molécula de referência.

    Preserva integralmente as métricas canônicas:
    - A2A: pIC50 contínuo e IC50 molar estimado (nM)
    - hERG: P(hERG=1) [escore do classificador] e 1 - P [índice complementar/segurança]
    - Descritores 2D RDKit: MolWt, MolLogP, NumHDonors, NumHAcceptors, TPSA, NumRotatableBonds

    Args:
        molecules_list: Lista de dicionários de análise (2 a 5 compostos).
        ref_index: Índice da molécula de referência caso ref_smiles não seja fornecido (padrão 0).
        ref_smiles: SMILES canônico da molécula a ser usada como referência.

    Returns:
        Lista de dicionários contendo valores absolutos e deltas relativos (Delta = Mol_i - Mol_ref).
    """
    if not molecules_list:
        return []

    chosen_ref_idx = ref_index
    if ref_smiles:
        for idx, m in enumerate(molecules_list):
            if m.get("canonical_smiles") == ref_smiles:
                chosen_ref_idx = idx
                break

    if chosen_ref_idx < 0 or chosen_ref_idx >= len(molecules_list):
        chosen_ref_idx = 0

    ref_mol_data = _extract_mol_properties(molecules_list[chosen_ref_idx])

    results: List[Dict[str, Any]] = []
    for idx, item in enumerate(molecules_list):
        is_ref = (idx == chosen_ref_idx)
        cur_data = _extract_mol_properties(item)

        # 1. Deltas farmacológicos
        delta_pIC50 = cur_data["pIC50"] - ref_mol_data["pIC50"]
        delta_prob_herg = cur_data["prob_herg"] - ref_mol_data["prob_herg"]
        delta_safety_index = cur_data["safety_index"] - ref_mol_data["safety_index"]

        ratio_ic50 = (
            cur_data["pred_IC50_nM"] / ref_mol_data["pred_IC50_nM"]
            if ref_mol_data["pred_IC50_nM"] > 0
            else 1.0
        )

        # 2. Deltas físico-químicos
        delta_descriptors: Dict[str, Any] = {}
        for desc_name in ["MolWt", "MolLogP", "NumHDonors", "NumHAcceptors", "TPSA", "NumRotatableBonds"]:
            cur_val = cur_data["descriptors"].get(desc_name, 0.0)
            ref_val = ref_mol_data["descriptors"].get(desc_name, 0.0)
            diff = cur_val - ref_val
            if desc_name in ("NumHDonors", "NumHAcceptors", "NumRotatableBonds"):
                delta_descriptors[f"delta_{desc_name}"] = int(round(diff))
            else:
                delta_descriptors[f"delta_{desc_name}"] = float(diff)

        entry = {
            "index": idx,
            "canonical_smiles": cur_data["canonical_smiles"],
            "compound_name": cur_data["compound_name"],
            "is_reference": is_ref,
            "values": {
                "pIC50": cur_data["pIC50"],
                "pred_IC50_nM": cur_data["pred_IC50_nM"],
                "prob_herg": cur_data["prob_herg"],
                "safety_index": cur_data["safety_index"],
                "complementary_index": cur_data["safety_index"],
                **cur_data["descriptors"],
            },
            "deltas": {
                "delta_pIC50": float(delta_pIC50),
                "ratio_IC50": float(ratio_ic50),
                "delta_prob_herg": float(delta_prob_herg),
                "delta_safety_index": float(delta_safety_index),
                "delta_complementary_index": float(delta_safety_index),
                **delta_descriptors,
            },
        }
        results.append(entry)

    return results


def build_comparative_multiobjective_figure(
    molecules_list: List[Dict[str, Any]],
    ref_smiles: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Constrói o gráfico cartesiano multiobjetivo comparativo para 2 a 5 moléculas simultâneas.

    Preserva integralmente:
    - Mesmos limites de eixos: X in [3.5, 11.5], Y in [-0.05, 1.05]
    - Mesmos limiares canônicos: A2A (5.0, 6.5) e hERG (0.30, 0.70)
    - Mesmos 4 quadrantes operacionais sombreados com idêntica paleta
    - Diferenciação cromática e morfológica dos marcadores para acessibilidade
    - Destaque específico e legenda identificando a molécula de referência
    - Posicionamento inteligente de rótulos para atenuar sobreposição

    Args:
        molecules_list: Lista contendo de 2 a 5 dicionários de moléculas do histórico.
        ref_smiles: SMILES canônico da molécula de referência (se None, assume a primeira).

    Returns:
        Tupla (fig, ax) contendo a figura Matplotlib renderizada.
    """
    fig_mat, ax_mat = plt.subplots(figsize=(10.5, 5.8), facecolor="#0E1117")
    ax_mat.set_facecolor("#1E222D")

    # Linhas de referência delimitando as faixas operacionais canônicas
    ax_mat.axvline(
        5.0,
        color="#94A3B8",
        linestyle=":",
        linewidth=1.1,
        alpha=0.80,
        label="Limiar Potência Moderada A2A (pIC50 = 5,0)",
    )
    ax_mat.axvline(
        6.5,
        color="#00D2BE",
        linestyle="--",
        linewidth=1.3,
        alpha=0.85,
        label="Limiar Alta Potência A2A (pIC50 = 6,5)",
    )
    ax_mat.axhline(
        0.70,
        color="#00D2BE",
        linestyle=":",
        linewidth=1.3,
        alpha=0.85,
        label="Limiar Operacional Baixo Risco hERG (1 - P = 0,70 | P = 0,30)",
    )
    ax_mat.axhline(
        0.30,
        color="#FF4B4B",
        linestyle=":",
        linewidth=1.1,
        alpha=0.80,
        label="Limiar Operacional Alto Risco hERG (1 - P = 0,30 | P = 0,70)",
    )

    # Sombreamentos dos quadrantes operacionais
    ax_mat.fill_between([6.5, 11.5], 0.70, 1.00, color="#00D2BE", alpha=0.14)
    ax_mat.fill_between([6.5, 11.5], 0.00, 0.30, color="#FF4B4B", alpha=0.14)
    ax_mat.fill_between([3.5, 5.0], 0.00, 0.30, color="#FF4B4B", alpha=0.08)
    ax_mat.fill_between([3.5, 5.0], 0.70, 1.00, color="#94A3B8", alpha=0.08)

    ax_mat.text(9.0, 0.95, "QUADRANTE 1: ZONA IDEAL\nAlta Potência & Baixo Risco", ha="center", va="top", fontsize=9.2, fontweight="bold", color="#00F5D4")
    ax_mat.text(9.0, 0.05, "QUADRANTE 4: ALERTA CRÍTICO\nPotente porém Alto Risco hERG", ha="center", va="bottom", fontsize=9.2, fontweight="bold", color="#FF7676")
    ax_mat.text(4.25, 0.95, "QUADRANTE 2: BAIXO INTERESSE\nInativo & Baixo Risco hERG", ha="center", va="top", fontsize=8.5, fontweight="bold", color="#CBD5E1")
    ax_mat.text(4.25, 0.05, "QUADRANTE 3: DESCARTE\nInativo & Alto Risco hERG", ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#94A3B8")

    # Paleta de 5 cores com alto contraste em Dark Mode
    palette = ["#00D2BE", "#F59E0B", "#A855F7", "#38BDF8", "#EC4899"]
    markers = ["o", "s", "^", "D", "v"]

    # Determina o índice de referência
    ref_idx = 0
    if ref_smiles:
        for idx, m in enumerate(molecules_list):
            if m.get("canonical_smiles") == ref_smiles:
                ref_idx = idx
                break

    # Posições de offset padrão para atenuar sobreposição
    default_offsets = [
        (0.22, 0.04),
        (-1.90, 0.04),
        (0.22, -0.11),
        (-1.90, -0.11),
        (0.00, 0.08),
    ]

    for i, item in enumerate(molecules_list):
        is_ref = (i == ref_idx) or (ref_smiles and item.get("canonical_smiles") == ref_smiles)
        color = palette[i % len(palette)]
        marker = markers[i % len(markers)]

        props = _extract_mol_properties(item)
        x_val = props["pIC50"]
        y_val = props["safety_index"]
        name = props["compound_name"]
        short_name = name[:12] + ".." if len(name) > 14 else name

        # Tamanho e borda diferenciados para a referência
        pt_size = 360 if is_ref else 260
        edge_color = "#FBBF24" if is_ref else "#FFFFFF"
        lw = 2.8 if is_ref else 1.8
        z_order = 8 if is_ref else 6

        lbl = f"#{i+1} {short_name}{' [REF]' if is_ref else ''}"
        ax_mat.scatter(
            x_val,
            y_val,
            s=pt_size,
            color=color,
            marker=marker,
            edgecolors=edge_color,
            linewidths=lw,
            zorder=z_order,
            label=f"#{i+1} {short_name}{' (Ref)' if is_ref else ''}: pIC50={x_val:.2f}, 1-P={y_val:.1%}",
        )

        # Cálculo dinâmico do deslocamento da caixa de anotação
        dx, dy = default_offsets[i % len(default_offsets)]
        if x_val > 9.5:
            dx = -1.85
        elif x_val < 5.2:
            dx = 0.22
        if y_val > 0.88:
            dy = -0.12
        elif y_val < 0.15:
            dy = 0.05

        ax_mat.annotate(
            f"#{i+1} {short_name}{' [REF]' if is_ref else ''}\n(pIC50={x_val:.2f} | 1-P={y_val:.1%})",
            (x_val + dx, y_val + dy),
            fontsize=8.8,
            fontweight="bold",
            color="#FFFFFF",
            bbox=dict(
                boxstyle="round,pad=0.32",
                fc="#0E1117",
                ec=edge_color if is_ref else color,
                lw=1.6 if is_ref else 1.2,
                alpha=0.95,
            ),
            zorder=9,
        )

    ax_mat.set_xlabel("Potência Predita A2A: pIC50 (Maior = Maior Potência Estimada)", fontsize=10, fontweight="bold", color="#FFFFFF")
    ax_mat.set_ylabel("Índice Complementar Derivado: 1 - P(hERG) [Escore do Modelo]", fontsize=10, fontweight="bold", color="#FFFFFF")
    ax_mat.set_title("Posicionamento Cartesiano Multiobjetivo Comparativo (A2A × hERG)", fontsize=11.5, fontweight="bold", color="#FFFFFF")
    ax_mat.set_xlim(3.5, 11.5)
    ax_mat.set_ylim(-0.05, 1.05)
    ax_mat.tick_params(colors="#E2E8F0", labelsize=9)
    for spine in ax_mat.spines.values():
        spine.set_color("#475569")
    ax_mat.grid(True, linestyle=":", alpha=0.35, color="#64748B")

    # Legenda estruturada inferior externa (não invade os quadrantes de dados)
    if molecules_list:
        n_cols = min(len(molecules_list), 3) if len(molecules_list) > 3 else len(molecules_list)
        ax_mat.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.15),
            ncol=n_cols,
            facecolor="#0E1117",
            edgecolor="#475569",
            fontsize=8.5,
            labelcolor="#FFFFFF",
            framealpha=0.95,
        )

    plt.tight_layout()
    return fig_mat, ax_mat
