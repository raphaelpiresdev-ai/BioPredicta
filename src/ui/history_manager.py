"""
MultiTarget BioPredict — Gerenciador de Histórico de Análises Moleculares (Etapa 5.2).

Implementa armazenamento em memória de sessão (st.session_state) não destrutivo,
isolando as análises concluídas da molécula em edição no canvas/formulário.

Características:
1. Limite FIFO estrito de até 10 análises concluídas por sessão.
2. Deduplicação por SMILES canônico (atualiza o registro e reposiciona como mais recente).
3. Armazenamento independente (cópias profundas de tensores/dataframes e exclusão de imagens pesadas).
4. Reconstrução determinística do objeto RDKit Mol via smiles_to_mol ao ativar um registro.
5. Compatibilidade integral com st.session_state["current_analysis"] em p3_resultados e p4_interpretacao.
"""

from __future__ import annotations

import copy
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem

from src.chem_utils import smiles_to_mol

logger = logging.getLogger(__name__)

MAX_HISTORY_ITEMS: int = 10
SESSION_HISTORY_KEY: str = "analysis_history"
SESSION_ACTIVE_KEY: str = "current_analysis"
SESSION_ACTIVE_SMILES_KEY: str = "current_analysis_smiles"


def _get_session_state(session_state: Optional[Any] = None) -> Any:
    """Retorna o estado de sessão fornecido ou o singleton st.session_state."""
    if session_state is not None:
        return session_state
    try:
        import streamlit as st
        return st.session_state
    except Exception:
        return {}


def init_history(session_state: Optional[Any] = None) -> List[Dict[str, Any]]:
    """Inicializa de forma segura o histórico de análises na sessão se ainda não existir.

    Retorna a lista de registros de histórico.
    """
    state = _get_session_state(session_state)
    if SESSION_HISTORY_KEY not in state or not isinstance(state[SESSION_HISTORY_KEY], list):
        state[SESSION_HISTORY_KEY] = []
    return state[SESSION_HISTORY_KEY]


def _deep_copy_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    """Copia um dicionário recursivamente preservando arrays numpy e dataframes pandas.

    Omite campos de imagens em base64 e referências C++ brutas a objetos RDKit para economizar RAM.
    """
    out: Dict[str, Any] = {}
    for k, v in d.items():
        if k in ("mol", "mol_b64", "b64_img", "image_base64", "b64_subenv", "b64_highlight"):
            # Omitidos no armazenamento para não onerar o heap da sessão
            continue
        elif isinstance(v, np.ndarray):
            out[k] = np.copy(v)
        elif isinstance(v, pd.DataFrame):
            out[k] = v.copy()
        elif isinstance(v, dict):
            out[k] = _deep_copy_dict(v)
        else:
            try:
                out[k] = copy.deepcopy(v)
            except Exception:
                out[k] = v
    return out


def _sanitize_record_for_storage(analysis_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Prepara um registro sanitizado para armazenamento seguro e leve no histórico."""
    record = _deep_copy_dict(analysis_dict)
    # Garante que o objeto mol seja explicitamente None na persistência
    record["mol"] = None
    return record


def add_to_history(
    analysis_dict: Optional[Dict[str, Any]],
    session_state: Optional[Any] = None,
    max_size: int = MAX_HISTORY_ITEMS,
) -> Optional[Dict[str, Any]]:
    """Adiciona uma análise molecular concluída ao histórico da sessão.

    Regras:
    1. Validação estrita: requer canonical_smiles válido e não-vazio.
    2. Deduplicação: se a molécula já estiver presente, remove a entrada antiga
       e posiciona a nova versão no final (mais recente).
    3. Política FIFO: se a quantidade de registros ultrapassar max_size,
       remove os registros mais antigos da cabeça da lista.
    4. Independência de memória: armazena cópias isoladas para evitar efeitos colaterais.

    Retorna o registro armazenado ou None se a análise for inválida.
    """
    if not analysis_dict or not isinstance(analysis_dict, dict):
        return None

    canon_smi = analysis_dict.get("canonical_smiles")
    if not canon_smi or not isinstance(canon_smi, str) or not canon_smi.strip():
        logger.warning("add_to_history: análise ignorada por ausência de canonical_smiles válido.")
        return None

    history = init_history(session_state)
    stored_record = _sanitize_record_for_storage(analysis_dict)

    # Deduplicação por SMILES canônico
    existing_idx: Optional[int] = None
    for i, item in enumerate(history):
        if item.get("canonical_smiles") == canon_smi:
            existing_idx = i
            break

    if existing_idx is not None:
        history.pop(existing_idx)

    history.append(stored_record)

    # Política FIFO
    while len(history) > max_size:
        history.pop(0)

    return stored_record


def get_history(session_state: Optional[Any] = None) -> List[Dict[str, Any]]:
    """Retorna a lista ordenada (FIFO) de análises armazenadas no histórico da sessão."""
    return init_history(session_state)


def set_active_molecule(
    canonical_smi: str,
    session_state: Optional[Any] = None,
) -> Optional[Dict[str, Any]]:
    """Recupera uma análise do histórico pelo SMILES canônico e a torna ativa na sessão.

    Ações:
    1. Localiza o registro correspondente no histórico.
    2. Cria uma cópia independente para a sessão ativa.
    3. Reconstrói o objeto RDKit Mol via smiles_to_mol(canonical_smi).
    4. Atualiza st.session_state["current_analysis"] e st.session_state["current_analysis_smiles"].

    Retorna o dicionário da análise ativa restaurada ou None se não encontrado.
    """
    if not canonical_smi or not isinstance(canonical_smi, str):
        return None

    history = get_history(session_state)
    match_record: Optional[Dict[str, Any]] = None
    for item in history:
        if item.get("canonical_smiles") == canonical_smi:
            match_record = item
            break

    if match_record is None:
        logger.warning(f"set_active_molecule: molécula {canonical_smi} não encontrada no histórico.")
        return None

    # Restaura estrutura completa
    active_dict = _deep_copy_dict(match_record)
    active_dict["canonical_smiles"] = canonical_smi
    active_dict["mol"] = smiles_to_mol(canonical_smi)

    state = _get_session_state(session_state)
    state[SESSION_ACTIVE_KEY] = active_dict
    state[SESSION_ACTIVE_SMILES_KEY] = canonical_smi

    return active_dict


def clear_history(session_state: Optional[Any] = None) -> None:
    """Limpa todos os registros de histórico da sessão (não remove current_analysis)."""
    state = _get_session_state(session_state)
    state[SESSION_HISTORY_KEY] = []


def get_history_summary_list(session_state: Optional[Any] = None) -> List[Dict[str, Any]]:
    """Retorna lista simplificada e pronta para exibição em tabelas/cartões de UI."""
    history = get_history(session_state)
    state = _get_session_state(session_state)
    active_smiles = state.get(SESSION_ACTIVE_SMILES_KEY)

    summary_list: List[Dict[str, Any]] = []
    # Itera em ordem reversa para apresentar as análises mais recentes no topo
    for idx, rec in enumerate(reversed(history)):
        smi = rec.get("canonical_smiles", "")
        pubchem = rec.get("pubchem") or {}
        prov = rec.get("provenance") or {}
        name = (
            pubchem.get("title")
            or prov.get("compound_name")
            or prov.get("name")
            or f"Composto #{len(history) - idx}"
        )

        a2a = rec.get("a2a") or {}
        herg = rec.get("herg") or {}
        quad = (rec.get("multiobjective") or {}).get("quadrant") or {}

        summary_list.append({
            "index": len(history) - idx,
            "display_name": str(name).strip(),
            "canonical_smiles": smi,
            "formula": rec.get("formula", ""),
            "mol_wt": rec.get("mol_wt", 0.0),
            "pIC50": a2a.get("pred_pIC50", 0.0),
            "a2a_class_label": (a2a.get("classification") or {}).get("label", "N/A"),
            "prob_herg": herg.get("prob_herg", 0.0),
            "safety_index": herg.get("safety_index", 1.0 - herg.get("prob_herg", 0.0)),
            "herg_class_label": (herg.get("classification") or {}).get("label", "N/A"),
            "quadrant_title": quad.get("quad_title", "N/A"),
            "is_active": (smi == active_smiles),
            "timestamp": rec.get("timestamp", ""),
        })

    return summary_list

