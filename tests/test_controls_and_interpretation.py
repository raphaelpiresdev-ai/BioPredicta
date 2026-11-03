"""
Suíte de testes automatizados da Fase 3C — Controles Farmacológicos e Interpretação dos Resultados.

Verifica:
1. Correspondência exata entre nome, SMILES canônico, InChIKey e identificador (CHEMBL431770 para Istradefilina,
   CHEMBL113 para Cafeína, CHEMBL22 para Terfenadina) nos controles e ausência de referências legadas incorretas.
2. Separação clara entre predição do modelo, medição experimental presente no dataset (com registro do tipo de
   ensaio ChEMBL) e referência bibliográfica externa, incluindo a marcação visual explícita
   'Amostra de Treinamento (In-Sample)' para Cafeína e Istradefilina.
3. Terminologia de hERG (escore de probabilidade estimado pelo modelo via predict_proba não calibrado,
   índice complementar derivado 1 - P_hERG e classes operacionais) e aviso sobre reajuste (refit 100%) pós-holdout.
4. Preservação dos limiares operacionais (5.0 e 6.5 para A2A; 0.30 e 0.70 para hERG) e das predições de referência.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any, Callable, Dict, Tuple

import numpy as np
import pandas as pd
import pytest
from rdkit import Chem
from rdkit.Chem import inchi

from src.chem_utils import canonicalize_smiles

ROOT_DIR = Path(__file__).resolve().parent.parent
APP_PATH = ROOT_DIR / "app.py"
DEV_MD_PATH = ROOT_DIR / "DEVELOPMENT.md"
PRD_MD_PATH = ROOT_DIR / "context" / "PRD.md"
HERG_META_PATH = ROOT_DIR / "models" / "herg_model_metadata.json"
NB06_PATH = ROOT_DIR / "notebooks" / "06_herg_modeling.ipynb"
NB07_PATH = ROOT_DIR / "notebooks" / "07_multitarget_shap_evaluation.ipynb"
A2A_CURATED_PATH = ROOT_DIR / "data" / "processed" / "a2a_curated.csv"
HERG_CURATED_PATH = ROOT_DIR / "data" / "processed" / "herg_curated.csv"


def _load_app_herg_classifiers() -> Tuple[Callable, Callable, Callable]:
    """Carrega as funções canônicas de classificação hERG e de quadrantes definidas em app.py."""
    source = APP_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(APP_PATH))
    target_funcs = {
        "classify_herg_operational_risk",
        "classify_herg_safety_axis",
        "classify_multiobjective_quadrant",
    }
    selected_nodes = [
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in target_funcs
    ]
    mod = ast.Module(body=selected_nodes, type_ignores=[])
    namespace: Dict[str, Any] = {"Dict": Dict, "np": np}
    exec(compile(mod, filename=str(APP_PATH), mode="exec"), namespace)
    return (
        namespace["classify_herg_operational_risk"],
        namespace["classify_herg_safety_axis"],
        namespace["classify_multiobjective_quadrant"],
    )


def _load_app_control_symbols() -> Tuple[Dict[str, Dict[str, Any]], Callable, Callable]:
    """
    Carrega REFERENCE_CONTROLS, load_training_lookup, resolve_compound_provenance e
    build_control_provenance_html diretamente da AST de app.py, sem inicializar o servidor Streamlit.
    """
    source = APP_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(APP_PATH))

    target_funcs = {
        "load_training_lookup",
        "resolve_compound_provenance",
        "build_control_provenance_html",
    }
    selected_nodes = []

    for node in tree.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.target.id == "REFERENCE_CONTROLS":
                selected_nodes.append(node)
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == "REFERENCE_CONTROLS":
                    selected_nodes.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in target_funcs:
            # Remove decoradores do Streamlit (@st.cache_data) para execução direta em teste
            node_copy = ast.FunctionDef(
                name=node.name,
                args=node.args,
                body=node.body,
                decorator_list=[],
                returns=node.returns,
                type_comment=node.type_comment,
            )
            ast.copy_location(node_copy, node)
            ast.fix_missing_locations(node_copy)
            selected_nodes.append(node_copy)

    mod = ast.Module(body=selected_nodes, type_ignores=[])
    namespace: Dict[str, Any] = {
        "Any": Any,
        "Dict": Dict,
        "Tuple": Tuple,
        "Optional": Any,
        "Path": Path,
        "pd": pd,
        "canonicalize_smiles": canonicalize_smiles,
        "__file__": str(APP_PATH),
    }
    exec(compile(mod, filename=str(APP_PATH), mode="exec"), namespace)
    return (
        namespace["REFERENCE_CONTROLS"],
        namespace["resolve_compound_provenance"],
        namespace["build_control_provenance_html"],
    )


def test_controls_identity_smiles_and_chembl_ids() -> None:
    """
    Verifica a identidade química dos 5 controles, garantindo que a Istradefilina esteja
    associada ao seu ID real nos dados (CHEMBL431770) e que nenhuma referência incorreta
    (CHEMBL223405 ou SMILES inválido com valência incorreta) permaneça no código ou documentação.
    """
    ref_controls, resolve_prov, _ = _load_app_control_symbols()

    expected_controls = {
        "☕ Cafeína (CHEMBL113)": {
            "name": "Cafeína",
            "chembl_id": "CHEMBL113",
            "canonical_smiles": "Cn1c(=O)c2c(ncn2C)n(C)c1=O",
            "inchikey": "RYYVLZVUVIJVGH-UHFFFAOYSA-N",
            "a2a_in_sample": True,
        },
        "💊 Istradefilina (CHEMBL431770)": {
            "name": "Istradefilina",
            "chembl_id": "CHEMBL431770",
            "canonical_smiles": "CCn1c(=O)c2c(nc(/C=C/c3ccc(OC)c(OC)c3)n2C)n(CC)c1=O",
            "inchikey": "IQVRBWUUXZMOPW-PKNBQFBNSA-N",
            "a2a_in_sample": True,
        },
        "⚠️ Terfenadina (CHEMBL22)": {
            "name": "Terfenadina",
            "chembl_id": "CHEMBL22",
            "canonical_smiles": "CC(C)(C)c1ccc(C(O)CCCN2CCC(C(O)(c3ccccc3)c3ccccc3)CC2)cc1",
            "inchikey": "GUGOEEXESWIERI-UHFFFAOYSA-N",
            "a2a_in_sample": False,
        },
        "🌿 Taurina (Endógeno)": {
            "name": "Taurina",
            "chembl_id": "Endógeno",
            "canonical_smiles": "NCCS(=O)(=O)O",
            "inchikey": "XOAAWQZATWQOTB-UHFFFAOYSA-N",
            "a2a_in_sample": False,
        },
        "🍵 L-Teanina (Nutracêutico)": {
            "name": "L-Teanina",
            "chembl_id": "Nutracêutico",
            "canonical_smiles": "CCNC(=O)CCC(N)C(=O)O",
            "inchikey": "DATAGRPVKZEWHA-UHFFFAOYSA-N",
            "a2a_in_sample": False,
        },
    }

    assert set(ref_controls.keys()) == set(expected_controls.keys())

    df_a2a = pd.read_csv(A2A_CURATED_PATH)
    for key, exp in expected_controls.items():
        ctrl = ref_controls[key]
        assert ctrl["name"] == exp["name"]
        assert ctrl["chembl_id"] == exp["chembl_id"]

        canon = canonicalize_smiles(ctrl["smiles"])
        assert canon == exp["canonical_smiles"]
        mol = Chem.MolFromSmiles(canon)
        assert mol is not None
        assert inchi.MolToInchiKey(mol) == exp["inchikey"]

        prov = resolve_prov(canon)
        assert prov["name"] == exp["name"]
        assert prov["chembl_id"] == exp["chembl_id"]
        assert prov["a2a_in_sample"] is exp["a2a_in_sample"]

        if exp["a2a_in_sample"]:
            row = df_a2a[df_a2a["canonical_smiles"] == canon]
            assert len(row) == 1
            assert row.iloc[0]["molecule_chembl_id"] == exp["chembl_id"]
            assert abs(float(row.iloc[0]["pIC50"]) - float(prov["a2a_exp_pic50"])) < 1e-3

    # Garante ausência de CHEMBL223405 e do SMILES inválido nos arquivos rastreados
    checked_files = [
        APP_PATH,
        DEV_MD_PATH,
        ROOT_DIR / "notebooks" / "05_a2a_modeling.ipynb",
        ROOT_DIR / "notebooks" / "06_herg_modeling.ipynb",
        ROOT_DIR / "notebooks" / "07_multitarget_shap_evaluation.ipynb",
    ]
    for path in checked_files:
        text = path.read_text(encoding="utf-8")
        assert "CHEMBL223405" not in text, f"Referência incorreta CHEMBL223405 encontrada em {path.name}"
        assert "Cc1nc2c(n1)n(c(=O)n(c2=O)C)C" not in text, f"SMILES inválido encontrado em {path.name}"


def test_dynamic_structure_name_resolution_prevents_mismatch() -> None:
    """
    Garante que o nome e o identificador exibidos correspondam estritamente à estrutura
    SMILES efetivamente submetida ao modelo (mesmo quando um usuário altera o texto SMILES).
    """
    _, resolve_prov, _ = _load_app_control_symbols()

    # 1. Submetendo SMILES aromático/kekulé bruto da Istradefilina -> resolve para Istradefilina (CHEMBL431770)
    raw_istrad = "CCN1C(=O)C2=C(N=C(N2C)/C=C/C3=CC(=C(C=C3)OC)OC)N(C1=O)CC"
    prov_istrad = resolve_prov(canonicalize_smiles(raw_istrad))
    assert prov_istrad["name"] == "Istradefilina"
    assert prov_istrad["chembl_id"] == "CHEMBL431770"
    assert prov_istrad["a2a_sample_status"] == "Amostra de Treinamento (In-Sample)"

    # 2. Submetendo uma estrutura arbitrária fora do dataset (ex.: benzeno) -> nunca mantém rótulo de controle
    prov_custom = resolve_prov(canonicalize_smiles("c1ccccc1"))
    assert prov_custom["is_reference_control"] is False
    assert prov_custom["name"] == "Molécula Personalizada"
    assert prov_custom["a2a_in_sample"] is False
    assert prov_custom["a2a_sample_status"] == "Fora da Amostra de Treino (Out-of-Sample)"


def test_presentation_separates_prediction_measurement_and_insample_tag() -> None:
    """
    Verifica que a camada de apresentação distingue claramente:
    - Predição do modelo para A2A (pIC50 predito);
    - Medição experimental presente no dataset utilizado (pIC50 exp.) e tipo de ensaio;
    - Escore de probabilidade estimado para hERG (P(hERG)) e índice complementar (1 - P);
    - Valores de referência externa;
    - Marcação visual explícita 'Amostra de Treinamento (In-Sample)' para Cafeína e Istradefilina.
    """
    ref_controls, resolve_prov, build_html = _load_app_control_symbols()

    # Caso 1: Istradefilina (In-Sample em A2A, ensaio funcional celular cAMP assay_type='F')
    istrad_canon = ref_controls["💊 Istradefilina (CHEMBL431770)"]["canonical_smiles"]
    prov_istrad = resolve_prov(istrad_canon)
    html_istrad = build_html(prov_istrad, pred_pIC50=5.5053, prob_herg=0.4594)

    assert "Amostra de Treinamento (In-Sample)" in html_istrad
    assert "insample-badge" in html_istrad
    assert "CHEMBL431770" in html_istrad
    assert "pIC50 predito = 5.51" in html_istrad
    assert "pIC50 exp. = 5.28" in html_istrad
    assert "assay_type = 'F'" in html_istrad
    assert "CHEMBL2384072" in html_istrad
    assert "P(hERG) = 45.9%" in html_istrad
    assert "1 - P = 54.1%" in html_istrad
    assert "Referência Bibliográfica Externa" in html_istrad

    # Caso 2: Cafeína (In-Sample em A2A, ensaio de ligação assay_type='B')
    caf_canon = ref_controls["☕ Cafeína (CHEMBL113)"]["canonical_smiles"]
    prov_caf = resolve_prov(caf_canon)
    html_caf = build_html(prov_caf, pred_pIC50=4.9465, prob_herg=0.1045)

    assert "Amostra de Treinamento (In-Sample)" in html_caf
    assert "CHEMBL113" in html_caf
    assert "pIC50 predito = 4.95" in html_caf
    assert "pIC50 exp. = 4.85" in html_caf
    assert "assay_type = 'B'" in html_caf

    # Caso 3: Terfenadina (Out-of-Sample em A2A e hERG)
    terf_canon = ref_controls["⚠️ Terfenadina (CHEMBL22)"]["canonical_smiles"]
    prov_terf = resolve_prov(terf_canon)
    html_terf = build_html(prov_terf, pred_pIC50=5.2413, prob_herg=0.8048)

    assert prov_terf["a2a_in_sample"] is False
    assert prov_terf["herg_in_sample"] is False
    assert "Fora da Amostra de Treino (Out-of-Sample)" in html_terf
    assert "Não disponível no dataset local" in html_terf
    assert "P(hERG) = 80.5%" in html_terf


def test_herg_terminology_thresholds_and_validation_notices() -> None:
    """
    Verifica a terminologia de hERG (escore estimado pelo modelo, índice complementar 1 - P_hERG,
    classes operacionais), o aviso sobre reajuste (refit 100%) pós-holdout 80/20 e a preservação
    dos limiares operacionais canônicos (5.0, 6.5 e [0.30, 0.70]).
    """
    app_text = APP_PATH.read_text(encoding="utf-8")
    dev_text = DEV_MD_PATH.read_text(encoding="utf-8")
    prd_text = PRD_MD_PATH.read_text(encoding="utf-8")
    herg_meta = json.loads(HERG_META_PATH.read_text(encoding="utf-8"))

    # Terminologia exigida em app.py
    assert "Escore Estimado P(hERG)" in app_text
    assert "Índice Complementar (1 - P_hERG)" in app_text or "Índice Complementar Derivado: 1 - P(hERG)" in app_text
    assert "Classes Operacionais" in app_text
    assert "Holdout 80/20 prévio" in app_text
    assert "reajustado em 100%" in app_text
    assert "Amostra de Treinamento (In-Sample)" in app_text

    # Preservação estrita dos limiares operacionais canônicos em app.py
    assert "pred_pIC50 >= 6.5" in app_text
    assert "pred_pIC50 >= 5.0" in app_text
    assert "prob_herg < 0.30" in app_text
    assert "prob_herg <= 0.70" in app_text
    assert "prob_herg < 0.70" not in app_text

    # Terminologia e convenção canônica em DEVELOPMENT.md, PRD.md e herg_model_metadata.json
    assert "CHEMBL431770" in dev_text
    assert "escore de probabilidade estimado pelo modelo" in dev_text
    assert "índice complementar derivado do escore predito" in dev_text.lower()
    assert "reajustado (*refit*) sobre a totalidade" in dev_text
    assert "5,51" in dev_text
    assert "45,9%" in dev_text
    assert "$30\\% \\le P_{\\text{hERG}} \\le 70\\%$" in dev_text
    assert "$P_{\\text{hERG}} > 70\\%$" in dev_text
    assert "$30\\% \\le P_{\\text{hERG}} \\le 70\\%$" in prd_text
    assert "$P_{\\text{hERG}} > 70\\%$" in prd_text
    assert herg_meta["decision_thresholds"] == {
        "green_low_risk": "P(hERG) < 0.30",
        "yellow_moderate_risk": "0.30 <= P(hERG) < 0.70",
        "red_high_risk": "P(hERG) >= 0.70",
    }


def test_herg_canonical_boundary_values_and_y_axis_complement() -> None:
    """
    Testa explicitamente os valores de fronteira p in [0.299, 0.300, 0.699, 0.700, 0.701],
    garantindo que os limites da classe intermediária se comportem estritamente como [0.30, 0.70)
    (e 0.30 < y <= 0.70 no eixo y = 1 - p, com tolerância de ponto flutuante pytest.approx / np.isclose).
    """
    classify_herg, classify_y, classify_quad = _load_app_herg_classifiers()

    boundary_cases = [
        (0.29, 0.71, "baixo", "semaphore-green", "#00D2BE"),
        (0.299, 0.701, "baixo", "semaphore-green", "#00D2BE"),
        (0.300, 0.700, "intermediario", "semaphore-yellow", "#F59E0B"),
        (0.699, 0.301, "intermediario", "semaphore-yellow", "#F59E0B"),
        (0.700, 0.300, "intermediario", "semaphore-yellow", "#F59E0B"),
        (0.701, 0.299, "alto", "semaphore-red", "#FF4B4B"),
        (0.71, 0.29, "alto", "semaphore-red", "#FF4B4B"),
    ]

    for p_val, expected_y, expected_class, expected_cls, expected_hex in boundary_cases:
        # 1. Classificação direta sobre P_hERG
        res = classify_herg(p_val)
        assert res["risk_class"] == expected_class, f"Falha em classify_herg({p_val})"
        assert res["color_class"] == expected_cls
        assert res["bar_color"] == expected_hex

        # 2. Classificação no eixo complementar y = 1 - p (com verificação de precisão de ponto flutuante)
        y_val = 1.0 - p_val
        assert y_val == pytest.approx(expected_y, abs=1e-12)
        assert bool(np.isclose(y_val, expected_y, rtol=0.0, atol=1e-12))

        y_class = classify_y(y_val)
        assert y_class == expected_class, f"Falha no eixo y = 1 - {p_val} ({y_val!r})"

        # 3. Verificação nos quadrantes para potência alta (x = 7.0 >= 6.5) e baixa (x = 4.5 < 5.0)
        quad_high_pot = classify_quad(7.0, y_val)
        assert quad_high_pot["herg_class"] == expected_class
        if expected_class == "baixo":
            assert "Quadrante 1" in quad_high_pot["quad_title"]
        elif expected_class == "intermediario":
            assert "Zona de Transição" in quad_high_pot["quad_title"]
        else:
            assert "Quadrante 4" in quad_high_pot["quad_title"]
            assert "P_hERG > 0,70" in quad_high_pot["quad_text"]

        quad_low_pot = classify_quad(4.5, y_val)
        assert quad_low_pot["herg_class"] == expected_class
        if expected_class == "baixo":
            assert "Quadrante 2" in quad_low_pot["quad_title"]
        elif expected_class == "intermediario":
            assert "Zona de Transição" in quad_low_pot["quad_title"]
        else:
            assert "Quadrante 3" in quad_low_pot["quad_title"]
            assert "P_hERG > 0,70" in quad_low_pot["quad_text"]

    # 4. Verifica também as funções auxiliares get_herg_semaforo (NB06) e semaforo_herg (NB07)
    nb06_data = json.loads(NB06_PATH.read_text(encoding="utf-8"))
    nb07_data = json.loads(NB07_PATH.read_text(encoding="utf-8"))

    nb06_func_src = None
    for cell in nb06_data["cells"]:
        if cell.get("cell_type") == "code":
            src = "".join(cell.get("source", []))
            if "def get_herg_semaforo(p):" in src:
                tree = ast.parse(src)
                for node in tree.body:
                    if isinstance(node, ast.FunctionDef) and node.name == "get_herg_semaforo":
                        nb06_func_src = node
                        break
    assert nb06_func_src is not None
    ns06: Dict[str, Any] = {}
    exec(compile(ast.Module(body=[nb06_func_src], type_ignores=[]), filename=str(NB06_PATH), mode="exec"), ns06)
    get_herg_semaforo = ns06["get_herg_semaforo"]

    assert "Baixo" in get_herg_semaforo(0.29)
    assert "Baixo" in get_herg_semaforo(0.299)
    assert "Moderado" in get_herg_semaforo(0.300)
    assert "Moderado" in get_herg_semaforo(0.699)
    assert "Moderado" in get_herg_semaforo(0.700)
    assert "Alto" in get_herg_semaforo(0.701)
    assert "Alto" in get_herg_semaforo(0.71)

    for cell in nb07_data["cells"]:
        if cell.get("cell_type") in ("code", "markdown"):
            src = "".join(cell.get("source", []))
            if "def semaforo_herg(p):" in src:
                tree = ast.parse(src)
                for node in tree.body:
                    if isinstance(node, ast.FunctionDef) and node.name == "semaforo_herg":
                        ns07: Dict[str, Any] = {}
                        exec(compile(ast.Module(body=[node], type_ignores=[]), filename=str(NB07_PATH), mode="exec"), ns07)
                        semaforo_herg = ns07["semaforo_herg"]
                        assert "Baixo" in semaforo_herg(0.29)
                        assert "Baixo" in semaforo_herg(0.299)
                        assert "Moderado" in semaforo_herg(0.300)
                        assert "Moderado" in semaforo_herg(0.699)
                        assert "Moderado" in semaforo_herg(0.700)
                        assert "Alto" in semaforo_herg(0.701)
                        assert "Alto" in semaforo_herg(0.71)


