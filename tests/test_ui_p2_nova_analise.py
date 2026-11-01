"""
Suíte de Testes da Etapa 2 — Nova Experiência de Análise Molecular (src/ui/pages/p2_nova_analise.py).

Testa:
1. Validação química e canonicalização de SMILES (válido, inválido, vazio, espaços);
2. Catálogo de moléculas modelo de referência rápida;
3. Classificações canônicas de limiares operacionais (A2A, hERG e Quadrantes);
4. Execução completa do pipeline molecular unificado (execute_molecular_pipeline);
5. Estrutura e integridade do dicionário de resultados da análise (current_analysis);
6. Resiliência da identificação PubChem (não-bloqueante);
7. Disponibilidade e integração do componente Ketcher.
"""

from __future__ import annotations

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

from src.chem_utils import FEATURE_NAMES, canonicalize_smiles, featurize_dataframe, smiles_to_mol
from src.pubchem_utils import (
    STATUS_CONNECTION_ERROR,
    STATUS_INVALID_STRUCTURE,
    STATUS_NOT_FOUND,
    STATUS_SUCCESS,
    identify_compound_pubchem,
)
from src.ui.pages.p2_nova_analise import (
    PRESET_REFERENCE_COMPOUNDS,
    classify_a2a_potency_canonical,
    classify_herg_risk_canonical,
    classify_quadrant_canonical,
    execute_molecular_pipeline,
    load_models_bundle_fallback,
)


class TestChemicalValidation:
    """Validação química estrutural com RDKit."""

    @pytest.mark.parametrize(
        "raw_smi,expected_canonical",
        [
            ("Cn1c(=O)c2c(ncn2C)n(C)c1=O", "Cn1c(=O)c2c(ncn2C)n(C)c1=O"),
            ("c1ccccc1", "c1ccccc1"),
            ("  c1ccccc1  ", "c1ccccc1"),
            ("NCCS(=O)(=O)O", "NCCS(=O)(=O)O"),
        ],
    )
    def test_valid_smiles_canonicalization(self, raw_smi: str, expected_canonical: str):
        canon = canonicalize_smiles(raw_smi.strip())
        assert canon is not None
        assert canon == expected_canonical
        mol = Chem.MolFromSmiles(canon)
        assert mol is not None
        assert mol.GetNumAtoms() > 0

    @pytest.mark.parametrize(
        "invalid_smi",
        [
            "",
            "   ",
            "c1ccccc",  # Anel aberto
            "CN(C)(C)(C)(C)C",  # Valência de nitrogênio pentavalente neutro
            "INVALID_SMILES_STRING_123",
            "C1CC",  # Anel aberto incompleto
        ],
    )
    def test_invalid_and_empty_smiles(self, invalid_smi: str):
        canon = canonicalize_smiles(invalid_smi.strip()) if invalid_smi.strip() else None
        assert canon is None, f"Esperava None para '{invalid_smi}', mas obteve '{canon}'"


class TestPresetReferenceCompounds:
    """Verificação do catálogo de moléculas modelo de acesso rápido."""

    def test_all_preset_compounds_are_valid(self):
        assert len(PRESET_REFERENCE_COMPOUNDS) >= 4
        for name, meta in PRESET_REFERENCE_COMPOUNDS.items():
            smi = meta["smiles"]
            canon = canonicalize_smiles(smi)
            assert canon is not None, f"Controle {name} possui SMILES inválido: {smi}"
            mol = Chem.MolFromSmiles(canon)
            assert mol is not None
            formula = rdMolDescriptors.CalcMolFormula(mol)
            mol_wt = Descriptors.MolWt(mol)
            assert len(formula) > 0
            assert mol_wt > 0.0


class TestCanonicalThresholds:
    """Verificação rigorosa dos limiares canônicos de classificação."""

    @pytest.mark.parametrize(
        "pic50,expected_class",
        [
            (8.5, "alta"),
            (6.5, "alta"),
            (6.49, "moderada"),
            (5.0, "moderada"),
            (4.99, "baixa"),
            (3.2, "baixa"),
        ],
    )
    def test_a2a_thresholds(self, pic50: float, expected_class: str):
        res = classify_a2a_potency_canonical(pic50)
        assert res["potency_class"] == expected_class

    @pytest.mark.parametrize(
        "prob,expected_class",
        [
            (0.10, "baixo"),
            (0.299, "baixo"),
            (0.2999, "baixo"),
            (0.30, "intermediario"),
            (0.50, "intermediario"),
            (0.699, "intermediario"),
            (0.70, "intermediario"),
            (0.7001, "alto"),
            (0.95, "alto"),
        ],
    )
    def test_herg_thresholds(self, prob: float, expected_class: str):
        res = classify_herg_risk_canonical(prob)
        assert res["risk_class"] == expected_class

    @pytest.mark.parametrize(
        "prob,expected_class",
        [
            (0.2999, "baixo"),
            (0.30, "intermediario"),
            (0.70, "intermediario"),
            (0.7001, "alto"),
        ],
    )
    def test_herg_audit_critical_boundaries(self, prob: float, expected_class: str):
        """Testa explicitamente as 4 fronteiras críticas exigidas na Auditoria Corretiva: 0.2999, 0.30, 0.70 e 0.7001."""
        res = classify_herg_risk_canonical(prob)
        assert res["risk_class"] == expected_class

    def test_multiobjective_quadrants(self):
        # Quadrante 1: Alta potência e baixo risco hERG
        q1 = classify_quadrant_canonical(7.0, 0.85)  # 1 - P = 0.85 -> P = 0.15
        assert "Quadrante 1" in q1["quad_title"]

        # Quadrante 4: Alta potência porém alto risco hERG
        q4 = classify_quadrant_canonical(7.0, 0.20)  # 1 - P = 0.20 -> P = 0.80
        assert "Quadrante 4" in q4["quad_title"]

        # Quadrante 2: Baixa potência e baixo risco hERG
        q2 = classify_quadrant_canonical(4.5, 0.85)
        assert "Quadrante 2" in q2["quad_title"]

        # Quadrante 3: Baixa potência e alto risco hERG
        q3 = classify_quadrant_canonical(4.5, 0.20)
        assert "Quadrante 3" in q3["quad_title"]

        # Zona Intermediária
        q_mid = classify_quadrant_canonical(5.8, 0.50)
        assert "Otimização" in q_mid["quad_title"] or "Transição" in q_mid["quad_title"]


class TestMolecularPipelineExecution:
    """Verificação do pipeline unificado de predição e estrutura do current_analysis."""

    @pytest.fixture(scope="class")
    @classmethod
    def models_bundle(cls):
        return load_models_bundle_fallback()

    def test_execute_pipeline_caffeine(self, models_bundle):
        smi = "Cn1c(=O)c2c(ncn2C)n(C)c1=O"
        canon = canonicalize_smiles(smi)
        mol = Chem.MolFromSmiles(canon)

        result = execute_molecular_pipeline(
            canonical_smi=canon,
            mol=mol,
            input_method="smiles",
            raw_input=smi,
            models_bundle=models_bundle,
        )

        # Campos de primeiro nível obrigatórios
        assert result["canonical_smiles"] == canon
        assert result["input_method"] == "smiles"
        assert result["raw_input"] == smi
        assert result["mol"] is not None
        assert result["formula"] == "C8H10N4O2"
        assert 194.0 < result["mol_wt"] < 195.0
        assert "MolLogP" in result["descriptors"]

        # Predições A2A
        a2a = result["a2a"]
        assert isinstance(a2a["pred_pIC50"], float)
        assert isinstance(a2a["pred_IC50_nM"], float)
        assert a2a["pred_pIC50"] > 0.0
        assert "classification" in a2a
        assert "ad" in a2a
        assert a2a["ad"]["status"] == "ok"
        assert len(a2a["shap_values"]) == len(FEATURE_NAMES)

        # Predições hERG
        herg = result["herg"]
        assert isinstance(herg["prob_herg"], float)
        assert 0.0 <= herg["prob_herg"] <= 1.0
        assert np.isclose(herg["prob_herg"] + herg["safety_index"], 1.0)
        assert "classification" in herg
        assert "ad" in herg
        assert herg["ad"]["status"] == "ok"
        assert len(herg["shap_values"]) == len(FEATURE_NAMES)

        # Multiobjetivo
        multi = result["multiobjective"]
        assert "quadrant" in multi
        assert multi["x"] == a2a["pred_pIC50"]
        assert multi["y"] == herg["safety_index"]


class TestPubChemResilience:
    """Verificação da integração não-bloqueante do PubChem."""

    def test_pubchem_identification_known_compound(self):
        # Cafeína possui CID 2519
        smi = "Cn1c(=O)c2c(ncn2C)n(C)c1=O"
        res = identify_compound_pubchem(smi, timeout=4.0, use_cache=True)
        assert res["status"] in (STATUS_SUCCESS, STATUS_CONNECTION_ERROR)
        if res["status"] == STATUS_SUCCESS:
            assert res["cid"] == 2519
            assert "Caffeine" in (res["title"] or "") or "caffeine" in (res["title"] or "").lower()

    def test_pubchem_uncataloged_compound_non_blocking(self):
        # Estrutura sintética não comum
        smi = "c1ccc(cc1)c2nc3ccccc3o2"
        res = identify_compound_pubchem(smi, timeout=3.0, use_cache=True)
        # Deve retornar NOT_FOUND ou SUCCESS (se existir) ou CONNECTION_ERROR, mas NUNCA falhar
        assert res["status"] in (STATUS_NOT_FOUND, STATUS_SUCCESS, STATUS_CONNECTION_ERROR)
        assert "canonical_smiles" in res or "inchikey" in res


class TestKetcherComponentAvailability:
    """Verificação da disponibilidade da biblioteca streamlit-ketcher."""

    def test_ketcher_import_and_signature(self):
        import streamlit_ketcher
        assert hasattr(streamlit_ketcher, "st_ketcher")


class TestComponentImports:
    """Verificação da exportação e importabilidade dos componentes da UI."""

    def test_import_ui_components_functions(self):
        from src.ui.components import (
            render_html,
            render_molecule_dark_base64,
            render_page_header,
            render_provisional_page,
        )
        assert callable(render_html)
        assert callable(render_molecule_dark_base64)
        assert callable(render_page_header)
        assert callable(render_provisional_page)

    def test_import_p2_nova_analise_module(self):
        import src.ui.pages.p2_nova_analise as p2
        assert hasattr(p2, "render_nova_analise")
        assert callable(p2.render_nova_analise)
        assert hasattr(p2, "render_html")
        assert callable(p2.render_html)


class TestMandatoryChemicalValidation:
    """
    Testes Químicos Obrigatórios para Moléculas Válidas e Diagnóstico de Erros.
    Cobre: Cafeína, Teobromina/Teofilina, Água, Etanol, Benzeno, Istradefilina e SMILES inválido do Ketcher.
    """

    def test_caffeine_validation(self):
        """Validação estrita da Cafeína (C8H10N4O2): 14 átomos pesados, 6 heteroátomos, 3 N-metilas."""
        caffeine_smi = "Cn1c(=O)c2c(ncn2C)n(C)c1=O"
        # 1. Parse RDKit sem erro e canonicalização
        canon = canonicalize_smiles(caffeine_smi)
        assert canon is not None
        assert canon == caffeine_smi
        mol = smiles_to_mol(canon)
        assert mol is not None

        # 2. Fórmula molecular esperada
        formula = rdMolDescriptors.CalcMolFormula(mol)
        assert formula == "C8H10N4O2", f"Fórmula incorreta: {formula}"

        # 3. Massa molecular coerente (~194.19 Da)
        mw = Descriptors.MolWt(mol)
        assert 194.0 <= mw <= 194.3, f"Massa molecular inesperada: {mw}"

        # 4. Quantidade de átomos e heteroátomos
        assert mol.GetNumAtoms() == 14
        hetero = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() not in (1, 6))
        assert hetero == 6  # 4 N + 2 O

        # 5. Presença específica das 3 metilas ligadas aos nitrogênios (N1, N3, N7)
        n_methyls = sum(
            1
            for a in mol.GetAtoms()
            if a.GetAtomicNum() == 6
            and not a.GetIsAromatic()
            and len(a.GetNeighbors()) == 1
            and a.GetNeighbors()[0].GetAtomicNum() == 7
        )
        assert n_methyls == 3, f"Esperava 3 grupos N-metila, encontrado: {n_methyls}"

        # 6. Preservação da identidade química após conversão round-trip (SMILES -> MolBlock -> SMILES)
        molblock = Chem.MolToMolBlock(mol)
        canon_from_block = canonicalize_smiles(molblock)
        assert canon_from_block == caffeine_smi

    def test_theobromine_theophylline_validation(self):
        """Validação da Teobromina / Teofilina (uma metila a menos: C7H8N4O2, MW ~180.17)."""
        theo_smi = "Cn1c(=O)c2[nH]cnc2n(C)c1=O"
        canon = canonicalize_smiles(theo_smi)
        assert canon is not None
        mol = smiles_to_mol(canon)
        assert mol is not None
        assert rdMolDescriptors.CalcMolFormula(mol) == "C7H8N4O2"
        assert 180.0 <= Descriptors.MolWt(mol) <= 180.3
        assert mol.GetNumAtoms() == 13
        assert sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() not in (1, 6)) == 6
        # Round-trip MolBlock
        mb = Chem.MolToMolBlock(mol)
        assert canonicalize_smiles(mb) == canon

    def test_water_validation(self):
        """Validação da Água (H2O, MW ~18.02, 1 átomo pesado)."""
        water_smi = "O"
        canon = canonicalize_smiles(water_smi)
        assert canon is not None
        mol = smiles_to_mol(canon)
        assert mol is not None
        assert rdMolDescriptors.CalcMolFormula(mol) == "H2O"
        assert 18.0 <= Descriptors.MolWt(mol) <= 18.1
        assert mol.GetNumAtoms() == 1
        assert sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() not in (1, 6)) == 1

    def test_ethanol_validation(self):
        """Validação do Etanol (C2H6O, MW ~46.07, 3 átomos pesados)."""
        eth_smi = "CCO"
        canon = canonicalize_smiles(eth_smi)
        assert canon is not None
        mol = smiles_to_mol(canon)
        assert mol is not None
        assert rdMolDescriptors.CalcMolFormula(mol) == "C2H6O"
        assert 46.0 <= Descriptors.MolWt(mol) <= 46.2
        assert mol.GetNumAtoms() == 3
        assert sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() not in (1, 6)) == 1

    def test_benzene_validation(self):
        """Validação do Benzeno (C6H6, MW ~78.11, 6 átomos pesados, 0 heteroátomos)."""
        benz_smi = "c1ccccc1"
        canon = canonicalize_smiles(benz_smi)
        assert canon is not None
        mol = smiles_to_mol(canon)
        assert mol is not None
        assert rdMolDescriptors.CalcMolFormula(mol) == "C6H6"
        assert 78.0 <= Descriptors.MolWt(mol) <= 78.2
        assert mol.GetNumAtoms() == 6
        assert sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() not in (1, 6)) == 0

    def test_istradefylline_validation(self):
        """Validação da Istradefilina clínica (C20H24N4O4, MW ~384.44, 28 átomos pesados)."""
        ist_smi = "CCn1c(=O)c2c(nc(/C=C/c3ccc(OC)c(OC)c3)n2C)n(CC)c1=O"
        canon = canonicalize_smiles(ist_smi)
        assert canon is not None
        mol = smiles_to_mol(canon)
        assert mol is not None
        assert rdMolDescriptors.CalcMolFormula(mol) == "C20H24N4O4"
        assert 384.0 <= Descriptors.MolWt(mol) <= 384.6
        assert mol.GetNumAtoms() == 28
        assert sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() not in (1, 6)) == 8

    def test_reproduction_of_invalid_ketcher_smiles(self):
        """
        Reprodução documental do erro de SMILES gerado pelo Ketcher:
        '[C]n1c(=O)nc2nc[n](C)c2c1=O'.
        Demonstra por que o RDKit rejeita a estrutura:
        1. Falha de kekulização (átomos 4, 5, 6, 7, 10);
        2. Notação [C] com 0 hidrogênios implícitos gerando valência radicalar inválida.
        """
        invalid_smi = "[C]n1c(=O)nc2nc[n](C)c2c1=O"
        # 1. RDKit rejeita no MolFromSmiles direto
        mol_direct = Chem.MolFromSmiles(invalid_smi)
        assert mol_direct is None, "RDKit não deveria interpretar este SMILES inválido"

        # 2. Canonicalizador do projeto rejeita retornando None
        assert canonicalize_smiles(invalid_smi) is None

        # 3. Validação analítica sem sanitização confirma anel não kekulizável
        mol_raw = Chem.MolFromSmiles(invalid_smi, sanitize=False)
        assert mol_raw is not None
        with pytest.raises(Exception):
            Chem.SanitizeMol(mol_raw)


class TestKetcherIntegrationPipeline:
    """
    Testes de Integração do Fluxo Completo:
    Ketcher (Molfile/SMILES) -> RDKit -> Canonicalização -> Featurização -> Sessão.
    """

    def test_ketcher_molfile_to_rdkit_canonical_smiles_and_featurization(self):
        """Simula o retorno de um Molfile V2000 gerado pelo Ketcher ao clicar Apply."""
        caffeine_mol = Chem.MolFromSmiles("Cn1c(=O)c2c(ncn2C)n(C)c1=O")
        molblock = Chem.MolToMolBlock(caffeine_mol)

        # O Ketcher retorna o Molfile
        canon_smi = canonicalize_smiles(molblock)
        assert canon_smi == "Cn1c(=O)c2c(ncn2C)n(C)c1=O"

        # Prossegue para a featurização 1030-D
        import pandas as pd
        df_single = pd.DataFrame([{"canonical_smiles": canon_smi}])
        df_feat = featurize_dataframe(df_single, "canonical_smiles")
        assert df_feat.shape[1] >= 1030
        for feat in FEATURE_NAMES:
            assert feat in df_feat.columns

    def test_ketcher_empty_molfile_handling(self):
        """Verifica que Molfile vazio (canvas resetado) não causa erro nem exceção."""
        empty_molblock = """
  Ketcher 

  0  0  0  0  0  0  0  0  0  0999 V2000
M  END
"""
        canon = canonicalize_smiles(empty_molblock)
        assert canon is None

    def test_mode_switching_state_consistency(self):
        """
        Simula a sincronização bidirecional entre Ketcher e Text Input:
        - Desenho no Ketcher gera Molfile;
        - Canonicalização converte para SMILES canônico limpo;
        - O estado input_smiles_field recebe a string linear, permitindo
          alternar para o modo texto sem poluir com o Molfile bruto.
        """
        caffeine_mol = Chem.MolFromSmiles("Cn1c(=O)c2c(ncn2C)n(C)c1=O")
        ketcher_output = Chem.MolToMolBlock(caffeine_mol)

        # Lógica de sincronização da UI
        canon = canonicalize_smiles(ketcher_output)
        assert canon is not None
        assert not canon.startswith("\n")
        assert "M  END" not in canon
        assert canon == "Cn1c(=O)c2c(ncn2C)n(C)c1=O"
