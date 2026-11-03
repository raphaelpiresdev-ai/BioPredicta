"""
Motor de Interpretação Química e Computacional — MultiTarget BioPredict.

Implementa a lógica auditada da Etapa 4 de UX/UI:
1. Mapeamento fiel de bits Morgan (ECFP4, radius=2, nBits=1024) via bitInfo do RDKit.
2. Identificação estrutural completa de subambientes (átomos, ligações, aromaticidade, raio e submol canônica).
3. Diferenciação rigorosa entre multiplicidade equivalente, colisão de hashing e ambiguidade não resolvida.
4. Renderização 2D de subambientes e destaque estrutural em Dark Mode.
5. Estruturação rigorosa das explicações em 5 campos independentes.
6. Decomposição explicativa SHAP com estrita observância do sinal por modelo (A2A e hERG) e sem extrapolações biológicas.
7. Interpretação didática de Domínio de Aplicabilidade e descritores físico-químicos.
"""

from __future__ import annotations

import base64
import logging
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit.Chem.Draw import rdMolDraw2D

from src.chem_utils import DESCRIPTOR_NAMES, FEATURE_NAMES
from src.ui.catalogs import (
    DESCRIPTOR_INFO_CATALOG,
    PHARMACOPHORE_BIT_CATALOG,
    resolve_translated_feature_name,
)

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# 1. Extração e Assinatura Estrutural de Ambientes Morgan (ECFP4)
# -----------------------------------------------------------------------------
def extract_morgan_bit_info(
    mol: Chem.Mol,
    radius: int = 2,
    n_bits: int = 1024,
) -> Dict[int, Tuple[Tuple[int, int], ...]]:
    """Calcula o bitInfo do Morgan Fingerprint preservando radius=2 e nBits=1024.

    Retorna um dicionário {bit_id: ((atom_idx, radius), ...)}.
    """
    if mol is None:
        return {}
    bit_info: Dict[int, Tuple[Tuple[int, int], ...]] = {}
    try:
        AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits, bitInfo=bit_info)
    except Exception as exc:
        logger.warning(f"Erro ao extrair bitInfo Morgan: {exc}")
        return {}
    return bit_info


def compute_subenvironment_signature(
    mol: Chem.Mol,
    atom_idx: int,
    radius: int,
) -> Dict[str, Any]:
    """Calcula a assinatura estrutural detalhada e a subestrutura canônica de uma ocorrência.

    Compara: símbolo atômico central, aromaticidade, hibridização, carga formal,
    hidrogênios totais, presença em anel, raio e a submol canônica com suas ligações.
    """
    if mol is None or atom_idx >= mol.GetNumAtoms():
        return {
            "signature": "invalid",
            "submol_smi": None,
            "desc": "Ocorrência inválida ou índice fora da molécula.",
            "center_atom_idx": atom_idx,
            "radius": radius,
            "symbol": "?",
            "is_aromatic": False,
            "atom_indices": [],
            "bond_indices": [],
            "is_valid": False,
        }

    atom = mol.GetAtomWithIdx(atom_idx)
    sym = atom.GetSymbol()
    arom = atom.GetIsAromatic()
    hyb = str(atom.GetHybridization())
    fc = atom.GetFormalCharge()
    num_hs = atom.GetTotalNumHs()
    in_ring = atom.IsInRing()

    # Raio 0: átomo central isolado
    if radius == 0:
        rooted_smi = f"[{sym}:1]"
        sig = f"Atom(sym={sym},arom={arom},hyb={hyb},fc={fc},h={num_hs},ring={in_ring},rooted_smi={rooted_smi})"
        desc = (
            f"Átomo de {sym} central isolado (raio 0, hibridização {hyb}, "
            f"{'aromático' if arom else 'alifático'}, carga {fc})."
        )
        return {
            "signature": sig,
            "submol_smi": rooted_smi,
            "desc": desc,
            "center_atom_idx": atom_idx,
            "radius": 0,
            "symbol": sym,
            "is_aromatic": arom,
            "atom_indices": [atom_idx],
            "bond_indices": [],
            "is_valid": True,
        }

    # Raio > 0: subambiente circular
    env_bonds = list(Chem.FindAtomEnvironmentOfRadiusN(mol, radius, atom_idx))
    if not env_bonds:
        rooted_smi = f"[{sym}:1]"
        sig = f"AtomOnly(r={radius},sym={sym},arom={arom},hyb={hyb},fc={fc},rooted_smi={rooted_smi})"
        return {
            "signature": sig,
            "submol_smi": rooted_smi,
            "desc": f"Átomo {sym} sem ligações detectadas no raio {radius}.",
            "center_atom_idx": atom_idx,
            "radius": radius,
            "symbol": sym,
            "is_aromatic": arom,
            "atom_indices": [atom_idx],
            "bond_indices": [],
            "is_valid": True,
        }

    submol_smi: Optional[str] = None
    try:
        atom_map: Dict[int, int] = {}
        submol = Chem.PathToSubmol(mol, env_bonds, atomMap=atom_map)
        root_submol_idx = atom_map.get(atom_idx)
        if root_submol_idx is not None and root_submol_idx < submol.GetNumAtoms():
            # Mapeamento do átomo central com atom map :1 para preservar a raiz e a topologia original
            submol.GetAtomWithIdx(root_submol_idx).SetAtomMapNum(1)
        submol_smi = Chem.MolToSmiles(submol, canonical=True)
    except Exception as exc:
        logger.debug(f"Falha ao gerar submol SMILES enraizado para átomo {atom_idx}, raio {radius}: {exc}")
        submol_smi = "unresolved_submol"

    atom_set = {atom_idx}
    for b_idx in env_bonds:
        b = mol.GetBondWithIdx(b_idx)
        atom_set.add(b.GetBeginAtomIdx())
        atom_set.add(b.GetEndAtomIdx())
    atom_indices = sorted(list(atom_set))

    # Assinatura canônica enraizada: preserva centro atômico, estado eletrônico e conectividade local
    sig = f"Env(r={radius},root_sym={sym},root_arom={arom},root_hyb={hyb},fc={fc},h={num_hs},ring={in_ring},rooted_smi={submol_smi})"
    desc = (
        f"Centro {sym} (átomo #{atom_idx}) com vizinhança circular de raio {radius} "
        f"({len(atom_indices)} átomos, {len(env_bonds)} ligações, subestrutura enraizada: {submol_smi or 'indisponível'})."
    )

    return {
        "signature": sig,
        "submol_smi": submol_smi,
        "desc": desc,
        "center_atom_idx": atom_idx,
        "radius": radius,
        "symbol": sym,
        "is_aromatic": arom,
        "atom_indices": atom_indices,
        "bond_indices": env_bonds,
        "is_valid": (submol_smi != "unresolved_submol"),
    }


def inspect_bit_environments(
    mol: Optional[Chem.Mol],
    bit_id: int,
    bit_info: Optional[Dict[int, Any]] = None,
) -> Dict[str, Any]:
    """Inspeciona os ambientes estruturais associados a um bit Morgan na molécula avaliada.

    Classifica rigorosamente em:
    1. 'single_environment': ocorrência única.
    2. 'multiple_occurrences': multiplicidade de ocorrências estruturalmente equivalentes.
    3. 'hash_collision': colisão de hashing entre ambientes estruturalmente distintos.
    4. 'unresolved_ambiguity': ambiguidade não resolvida.
    """
    if mol is None or bit_info is None or bit_id not in bit_info:
        return {
            "status": "absent",
            "bit_id": bit_id,
            "count": 0,
            "has_collision": False,
            "ambiguity_type": "none",
            "environments": [],
        }

    raw_envs = bit_info[bit_id]
    result_envs: List[Dict[str, Any]] = []

    for idx, (atom_idx, radius) in enumerate(raw_envs):
        sig_data = compute_subenvironment_signature(mol, atom_idx, radius)
        sig_data["example_index"] = idx
        sig_data["observational_desc"] = sig_data["desc"]
        result_envs.append(sig_data)

    count = len(result_envs)

    # Análise de assinaturas estruturais distintas
    signatures = [e["signature"] for e in result_envs]
    distinct_sigs = set(signatures)
    has_invalid = any(not e.get("is_valid", True) for e in result_envs)

    if has_invalid:
        ambiguity_type = "unresolved_ambiguity"
        has_collision = True
    elif count == 1:
        ambiguity_type = "single_environment"
        has_collision = False
    elif len(distinct_sigs) > 1:
        ambiguity_type = "hash_collision"
        has_collision = True
    else:
        ambiguity_type = "multiple_occurrences"
        has_collision = False

    return {
        "status": "present",
        "bit_id": bit_id,
        "count": count,
        "has_collision": has_collision,
        "ambiguity_type": ambiguity_type,
        "environments": result_envs,
    }


# -----------------------------------------------------------------------------
# 2. Renderização 2D de Subambientes em Dark Mode
# -----------------------------------------------------------------------------
def render_morgan_subenv_png(
    mol: Optional[Chem.Mol],
    bit_id: int,
    bit_info: Optional[Dict[int, Any]] = None,
    which_example: int = 0,
    size: Tuple[int, int] = (160, 160),
) -> Optional[str]:
    """Renderiza a imagem 2D isolada do subambiente correspondente ao bit via Draw.DrawMorganBit.

    Retorna string base64 PNG com paleta Dark Mode e fundo transparente.
    """
    if mol is None or bit_info is None or bit_id not in bit_info:
        return None

    envs = bit_info[bit_id]
    if which_example >= len(envs):
        which_example = 0

    opts = rdMolDraw2D.MolDrawOptions()
    rdMolDraw2D.SetDarkMode(opts)
    opts.clearBackground = False

    try:
        png_bytes = Draw.DrawMorganBit(
            mol,
            bit_id,
            bit_info,
            whichExample=which_example,
            molSize=size,
            useSVG=False,
            drawOptions=opts,
        )
        return base64.b64encode(png_bytes).decode("utf-8")
    except Exception as exc:
        logger.debug(f"Falha ao desenhar subambiente Morgan para bit {bit_id}: {exc}")
        return None


def render_mol_highlight_png(
    mol: Optional[Chem.Mol],
    highlight_atoms: Optional[List[int]] = None,
    highlight_bonds: Optional[List[int]] = None,
    size: Tuple[int, int] = (300, 200),
) -> Optional[str]:
    """Renderiza a estrutura completa da molécula com o subambiente destacado em Dark Mode."""
    if mol is None:
        return None

    atoms = highlight_atoms or []
    bonds = highlight_bonds or []

    try:
        drawer = rdMolDraw2D.MolDraw2DCairo(size[0], size[1])
        opts = drawer.drawOptions()
        rdMolDraw2D.SetDarkMode(opts)
        opts.clearBackground = False
        opts.bondLineWidth = 2.0
        drawer.DrawMolecule(
            mol,
            highlightAtoms=atoms,
            highlightBonds=bonds,
        )
        drawer.FinishDrawing()
        return base64.b64encode(drawer.GetDrawingText()).decode("utf-8")
    except Exception as exc:
        logger.debug(f"Falha ao desenhar molécula com destaques: {exc}")
        return None


# -----------------------------------------------------------------------------
# 3. Estruturação dos 5 Campos Obrigatórios da Explicação
# -----------------------------------------------------------------------------
def build_attribute_explanation_card(
    feature_name: str,
    shap_value: float,
    target: str,
    mol: Optional[Chem.Mol] = None,
    bit_info: Optional[Dict[int, Any]] = None,
    feature_value: Optional[float] = None,
) -> Dict[str, Any]:
    """Constrói a estrutura rigorosa de 5 campos independentes para qualquer atributo SHAP.

    Campos:
    1. Atributo computacional (id, tipo, nome didático).
    2. Contribuição SHAP (valor, sinal, direção da contribuição para a saída do modelo).
    3. Subambiente estrutural (fragmento 2D, átomos, múltiplos ambientes ou ausência).
    4. Descrição observacional (caracterização química neutra).
    5. Hipótese bibliográfica (somente se documentada).
    """
    target_clean = "A2A" if "A2A" in target.upper() else "hERG"
    is_descriptor = feature_name in DESCRIPTOR_NAMES
    translated_name = resolve_translated_feature_name(feature_name, mol=mol, bit_info=bit_info)

    # 1. Atributo Computacional
    attr_info = {
        "feature_id": feature_name,
        "is_descriptor": is_descriptor,
        "type_label": "Descritor Físico-Químico 2D" if is_descriptor else "Fingerprint de Morgan (ECFP4)",
        "translated_name": translated_name,
        "feature_value": feature_value,
    }

    # 2. Contribuição SHAP e Direção
    sign_str = f"+{shap_value:.4f}" if shap_value >= 0 else f"{shap_value:.4f}"
    is_positive = shap_value >= 0

    if target_clean == "A2A":
        # Modelo XGBoost de regressão pIC50
        if is_positive:
            direction_desc = "Eleva a saída predita de afinidade (aumenta o pIC₅₀ estimado)."
            impact_class = "a2a_potency_elevated"
        else:
            direction_desc = "Reduz a saída predita de afinidade (diminui o pIC₅₀ estimado)."
            impact_class = "a2a_potency_reduced"
    else:
        # Modelo Random Forest de classificação hERG (probabilidade classe 1: saída positiva)
        # Terminologia estritamente neutra: não usar "fatores protetivos" ou "cardiac_safe"
        if is_positive:
            direction_desc = "Eleva a saída do modelo associada à classe hERG positiva (aumenta o escore predito pelo classificador)."
            impact_class = "herg_positive_elevated"
        else:
            direction_desc = "Reduz a saída do modelo associada à classe hERG positiva (diminui o escore predito pelo classificador)."
            impact_class = "herg_positive_reduced"

    shap_info = {
        "value": float(shap_value),
        "sign_str": sign_str,
        "is_positive": is_positive,
        "target": target_clean,
        "direction_desc": direction_desc,
        "impact_class": impact_class,
    }

    # 3. Subambiente Estrutural e 4. Descrição Observacional e 5. Hipótese Bibliográfica
    if is_descriptor:
        desc_catalog = DESCRIPTOR_INFO_CATALOG.get(feature_name, {})
        val_str = f"{feature_value:.2f}" if feature_value is not None else "Calculado na análise"
        unit_str = desc_catalog.get("unit", "")

        structural_info = {
            "status": "scalar_descriptor",
            "message": f"Propriedade físico-química global contínua (não fragmentária). Valor: {val_str} {unit_str}.",
            "environments": [],
            "b64_subenv": None,
            "b64_highlight": None,
            "has_collision": False,
        }

        observational_desc = (
            f"Descritor escalar molecular: {desc_catalog.get('label', feature_name)}. "
            f"Mede a propriedade global da estrutura como um todo, sem localização atômica única."
        )

        role_key = "role_a2a" if target_clean == "A2A" else "role_herg"
        hypo_text = desc_catalog.get(role_key)
        if hypo_text:
            bibliographic_info = {
                "has_reference": True,
                "text": hypo_text,
                "disclaimer": (
                    "Contexto bibliográfico da propriedade molecular. "
                    "A contribuição SHAP local não constitui regra farmacológica universal."
                ),
            }
        else:
            bibliographic_info = {
                "has_reference": False,
                "text": "Sem hipótese bibliográfica documentada para este descritor específico.",
                "disclaimer": None,
            }

    else:
        # É um bit Morgan
        bit_id = int(feature_name.split("_")[1]) if "_" in feature_name else -1
        env_diag = inspect_bit_environments(mol, bit_id, bit_info)

        # Regra mandatória: bit fp_926
        is_fp_926 = (bit_id == 926 or feature_name == "fp_926")

        if env_diag["status"] == "absent":
            structural_info = {
                "status": "absent",
                "message": (
                    f"Atributo {feature_name} ausente na molécula avaliada (valor = 0 no vetor de 1.024 bits). "
                    "O valor SHAP é não-nulo porque o ensemble de árvores computa o impacto da ausência dessa subestrutura "
                    "em seus nós de decisão, influenciando aditivamente a predição. Nenhuma imagem de fragmento é exibida "
                    "pois o padrão não existe na molécula."
                ),
                "environments": [],
                "b64_subenv": None,
                "b64_highlight": None,
                "has_collision": False,
            }
            if is_fp_926:
                observational_desc = "Carbono alifático (raio 0) ausente na estrutura avaliada."
            else:
                observational_desc = (
                    f"Subambiente correspondente ao bit #{bit_id} não detectado nesta molécula (valor 0)."
                )

        else:
            # Bit presente
            envs = env_diag["environments"]
            first_env = envs[0]
            b64_sub = render_morgan_subenv_png(mol, bit_id, bit_info, which_example=0)
            b64_high = render_mol_highlight_png(mol, first_env["atom_indices"], first_env["bond_indices"])

            collision_msg = None
            if env_diag["ambiguity_type"] == "hash_collision":
                collision_msg = (
                    f"Colisão de Hashing: {env_diag['count']} ambientes estruturalmente distintos mapeados "
                    f"para o mesmo bit {feature_name} no vetor dobrado de 1.024 bits."
                )
            elif env_diag["ambiguity_type"] == "multiple_occurrences":
                collision_msg = (
                    f"Multiplicidade: {env_diag['count']} ocorrências estruturalmente equivalentes "
                    f"deste fragmento na molécula."
                )
            elif env_diag["ambiguity_type"] == "unresolved_ambiguity":
                collision_msg = (
                    f"Ambiguidade não resolvida: falha na reconstrução canônica de subambientes para {feature_name}."
                )

            structural_info = {
                "status": "present",
                "message": collision_msg or "Subambiente estrutural único identificado com sucesso.",
                "environments": envs,
                "b64_subenv": b64_sub,
                "b64_highlight": b64_high,
                "has_collision": env_diag["has_collision"],
                "collision_warning": collision_msg,
                "ambiguity_type": env_diag["ambiguity_type"],
            }

            if is_fp_926:
                observational_desc = "Carbono alifático (raio 0)."
            else:
                observational_desc = first_env["observational_desc"]

        # Hipótese bibliográfica do catálogo
        bit_cat = PHARMACOPHORE_BIT_CATALOG.get(bit_id, {})
        role_key = "role_a2a" if target_clean == "A2A" else "role_herg"
        hypo_text = bit_cat.get(role_key)

        if hypo_text:
            bibliographic_info = {
                "has_reference": True,
                "text": hypo_text,
                "disclaimer": (
                    "Hipótese fundamentada na literatura de quimioinformática médica; "
                    "não constitui comprovação de mecanismo experimental in vitro para este composto."
                ),
            }
        else:
            bibliographic_info = {
                "has_reference": False,
                "text": "Sem hipótese bibliográfica documentada para este atributo específico.",
                "disclaimer": None,
            }

    return {
        "feature_name": feature_name,
        "target": target_clean,
        "computational_attribute": attr_info,
        "shap_contribution": shap_info,
        "structural_subenvironment": structural_info,
        "observational_description": observational_desc,
        "bibliographic_hypothesis": bibliographic_info,
    }


# -----------------------------------------------------------------------------
# 4. Seleção e Ranqueamento dos Atributos Relevantes
# -----------------------------------------------------------------------------
def get_ranked_explanations(
    shap_values: np.ndarray,
    target: str,
    mol: Optional[Chem.Mol] = None,
    bit_info: Optional[Dict[int, Any]] = None,
    feature_values: Optional[Dict[str, float]] = None,
    top_n: int = 6,
) -> Dict[str, List[Dict[str, Any]]]:
    """Retorna explicações completas para os atributos de maior impacto positivo, negativo e absoluto."""
    if shap_values is None or len(shap_values) != len(FEATURE_NAMES):
        return {"top_positive": [], "top_negative": [], "top_absolute": []}

    feat_vals = feature_values or {}
    indexed_items: List[Tuple[int, str, float]] = []

    for idx, feat_name in enumerate(FEATURE_NAMES):
        s_val = float(shap_values[idx])
        indexed_items.append((idx, feat_name, s_val))

    # Top positivos (ordenados do maior para o menor)
    positives = sorted([item for item in indexed_items if item[2] > 0], key=lambda x: x[2], reverse=True)
    # Top negativos (ordenados do mais negativo para o zero)
    negatives = sorted([item for item in indexed_items if item[2] < 0], key=lambda x: x[2])
    # Top por magnitude absoluta
    absolutes = sorted(indexed_items, key=lambda x: abs(x[2]), reverse=True)

    def _build_list(subset: List[Tuple[int, str, float]]) -> List[Dict[str, Any]]:
        result = []
        for _, f_name, s_val in subset[:top_n]:
            f_val = feat_vals.get(f_name)
            card = build_attribute_explanation_card(
                feature_name=f_name,
                shap_value=s_val,
                target=target,
                mol=mol,
                bit_info=bit_info,
                feature_value=f_val,
            )
            result.append(card)
        return result

    return {
        "top_positive": _build_list(positives),
        "top_negative": _build_list(negatives),
        "top_absolute": _build_list(absolutes),
    }


# -----------------------------------------------------------------------------
# 5. Interpretação Didática dos Descritores Físico-Químicos
# -----------------------------------------------------------------------------
def explain_physicochemical_descriptors(
    descriptors_dict: Dict[str, float],
    shap_a2a: Optional[np.ndarray],
    shap_herg: Optional[np.ndarray],
) -> List[Dict[str, Any]]:
    """Gera o quadro comparativo didático dos 6 descritores físico-químicos com valores SHAP."""
    results: List[Dict[str, Any]] = []

    for desc_name in DESCRIPTOR_NAMES:
        val = float(descriptors_dict.get(desc_name, 0.0))
        cat = DESCRIPTOR_INFO_CATALOG.get(desc_name, {})

        idx = FEATURE_NAMES.index(desc_name) if desc_name in FEATURE_NAMES else -1

        s_a2a = float(shap_a2a[idx]) if (shap_a2a is not None and idx >= 0 and idx < len(shap_a2a)) else 0.0
        s_herg = float(shap_herg[idx]) if (shap_herg is not None and idx >= 0 and idx < len(shap_herg)) else 0.0

        results.append({
            "name": desc_name,
            "label": cat.get("label", desc_name),
            "unit": cat.get("unit", ""),
            "value": val,
            "shap_a2a": s_a2a,
            "shap_a2a_str": f"+{s_a2a:.4f}" if s_a2a >= 0 else f"{s_a2a:.4f}",
            "direction_a2a": "Eleva pIC50" if s_a2a >= 0 else "Reduz pIC50",
            "shap_herg": s_herg,
            "shap_herg_str": f"+{s_herg:.4f}" if s_herg >= 0 else f"{s_herg:.4f}",
            "direction_herg": "Eleva saída hERG positiva" if s_herg >= 0 else "Reduz saída hERG positiva",
            "role_a2a": cat.get("role_a2a", "Sem descrição específica."),
            "role_herg": cat.get("role_herg", "Sem descrição específica."),
        })

    return results


# -----------------------------------------------------------------------------
# 6. Interpretação Didática do Domínio de Aplicabilidade (AD)
# -----------------------------------------------------------------------------
def explain_ad_coverage(
    ad_diag: Optional[Dict[str, Any]],
    endpoint: str,
) -> Dict[str, Any]:
    """Extrai e formata os diagnósticos de AD existentes sem recalcular nenhuma métrica."""
    ep = "A2A" if "A2A" in endpoint.upper() else "hERG"
    if not ad_diag or ad_diag.get("status") != "ok":
        return {
            "status": "unavailable",
            "endpoint": ep,
            "provenance_level": None,
            "provenance_desc": "Diagnóstico não disponível ou incompleto.",
            "s_max_all": None,
            "s_max_excluding_match": None,
            "has_exclusion": False,
            "nearest_neighbor": None,
            "loo_median": None,
            "is_above_loo": False,
            "domain_interpretation": "Impossível avaliar a cobertura do domínio com os dados fornecidos.",
        }

    prov_level = ad_diag.get("provenance_level")
    prov_desc = ad_diag.get("provenance", "Sem correspondência exata conhecida")
    s_max_all = ad_diag.get("s_max_all")
    s_max_excl = ad_diag.get("s_max_excluding_match")
    has_excl = ad_diag.get("has_exclusion", False)
    nn = ad_diag.get("nearest_neighbor") or {}
    loo_med = ad_diag.get("loo_median_similarity")

    effective_sim = s_max_excl if (has_excl and s_max_excl is not None) else s_max_all
    is_above_loo = (effective_sim is not None and loo_med is not None and effective_sim >= loo_med)

    if prov_level in (1, 2):
        interp = (
            f"Molécula com correspondência direta no conjunto de calibração ({prov_desc}). "
            f"Operando em regime de alta densidade amostral in-sample. "
            f"Similaridade com outros compostos de treino (auto-exclusão aplicada): S_max = {s_max_excl:.4f}."
            if s_max_excl is not None else
            f"Molécula com correspondência direta no conjunto de calibração ({prov_desc})."
        )
    elif prov_level == 3:
        interp = (
            "Mesmo padrão circular de 1.024 bits que uma amostra de treino, porém com descritores físico-químicos "
            "distintos (colisão de fingerprint ou isômero estrutural). Não sofreu auto-exclusão."
        )
    else:
        if is_above_loo:
            interp = (
                f"Molécula out-of-sample com vizinhança química densa (S_max = {s_max_all:.4f} ≥ mediana LOO {loo_med:.4f}). "
                "Predição operando dentro da cobertura estrutural típica observada nos dados de calibração."
            )
        else:
            interp = (
                f"Molécula out-of-sample em região de menor densidade química (S_max = {s_max_all:.4f} < mediana LOO {loo_med:.4f}). "
                "Predição em potencial extrapolação estrutural em relação à distribuição central do conjunto de treino."
            )

    return {
        "status": "ok",
        "endpoint": ep,
        "provenance_level": prov_level,
        "provenance_desc": prov_desc,
        "s_max_all": s_max_all,
        "s_max_excluding_match": s_max_excl,
        "has_exclusion": has_excl,
        "nearest_neighbor": nn,
        "loo_median": loo_med,
        "is_above_loo": is_above_loo,
        "domain_interpretation": interp,
    }
