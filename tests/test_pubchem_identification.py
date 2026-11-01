"""
Suíte de testes unitários para a identificação automatizada de moléculas via PubChem.

Verifica:
1. Validação e rejeição graciosa de estruturas químicas inválidas / vazias sem realizar requisições de rede.
2. Identificação bem-sucedida (status SUCCESS), extração de CID, título, IUPAC, nomes comerciais e sinônimos.
3. Tratamento robusto para moléculas ausentes / não catalogadas no PubChem (HTTP 404 -> status NOT_FOUND).
4. Resiliência e degradação graciosa em caso de falha de conexão, timeout ou erro de servidor (status CONNECTION_ERROR).
5. Funcionamento correto do cache em memória para evitar chamadas de rede redundantes.
6. Geração dos cartões HTML de apresentação visual (build_pubchem_card_html).
"""

from __future__ import annotations

import io
import json
import urllib.error
from unittest.mock import MagicMock, patch

import pytest
from rdkit import Chem

from src.pubchem_utils import (
    STATUS_CONNECTION_ERROR,
    STATUS_INVALID_STRUCTURE,
    STATUS_NOT_FOUND,
    STATUS_SUCCESS,
    build_pubchem_card_html,
    clean_and_categorize_synonyms,
    clear_pubchem_cache,
    identify_compound_pubchem,
)


@pytest.fixture(autouse=True)
def _reset_cache():
    """Garante cache limpo antes e após cada teste."""
    clear_pubchem_cache()
    yield
    clear_pubchem_cache()


# -----------------------------------------------------------------------------
# 1. Estruturas Inválidas e Casos Limite
# -----------------------------------------------------------------------------
def test_identify_compound_invalid_or_empty_structures():
    """Entradas vazias, None ou com sintaxe química inválida devem retornar STATUS_INVALID_STRUCTURE sem invocar rede."""
    with patch("urllib.request.urlopen") as mock_urlopen:
        # String vazia
        res_empty = identify_compound_pubchem("")
        assert res_empty["status"] == STATUS_INVALID_STRUCTURE
        assert res_empty["cid"] is None

        # Apenas espaços
        res_spaces = identify_compound_pubchem("   ")
        assert res_spaces["status"] == STATUS_INVALID_STRUCTURE

        # Sintaxe SMILES inválida
        res_invalid = identify_compound_pubchem("INVALID_SMILES_XYZ###")
        assert res_invalid["status"] == STATUS_INVALID_STRUCTURE

        # Valência impossível (ex: carbono pentavalente)
        res_hypervalent = identify_compound_pubchem("c1c=cc2c=cc3c=cc=c4c=cc=1=c=2c=3=4")
        assert res_hypervalent["status"] == STATUS_INVALID_STRUCTURE

        # Tipo não suportado
        res_type = identify_compound_pubchem(12345)  # type: ignore
        assert res_type["status"] == STATUS_INVALID_STRUCTURE

        # Nenhuma requisição de rede deve ter sido realizada
        mock_urlopen.assert_not_called()


# -----------------------------------------------------------------------------
# 2. Caso de Sucesso na Identificação (Mock de Resposta PubChem)
# -----------------------------------------------------------------------------
def test_identify_compound_success_pyrene():
    """Identificação bem-sucedida do Pireno via PubChem PUG REST API."""
    pyrene_smiles = "c1cc2ccc3cccc4ccc(c1)c2c34"

    # Mocks para as duas etapas da API (propriedades e sinônimos)
    mock_prop_resp = {
        "PropertyTable": {
            "Properties": [
                {
                    "CID": 31423,
                    "MolecularFormula": "C16H10",
                    "MolecularWeight": "202.25",
                    "IUPACName": "pyrene",
                    "Title": "Pyrene",
                }
            ]
        }
    }

    mock_syn_resp = {
        "InformationList": {
            "Information": [
                {
                    "CID": 31423,
                    "Synonym": [
                        "PYRENE",
                        "129-00-0",  # CAS deve ser filtrado
                        "Benzo[def]phenanthrene",
                        "Pyren",
                        "beta-Pyrene",
                        "Benzo(def)phenanthrene",
                        "DTXSID3024289",  # Código EPA deve ser filtrado
                        "NSC 17534",  # Código NSC deve ser filtrado
                        "CHEBI:39106",  # Código ChEBI deve ser filtrado
                    ],
                }
            ]
        }
    }

    def mock_urlopen_router(request, timeout=None):
        url = request.full_url if hasattr(request, "full_url") else str(request)
        if "property" in url:
            body = json.dumps(mock_prop_resp).encode("utf-8")
        elif "synonyms" in url:
            body = json.dumps(mock_syn_resp).encode("utf-8")
        else:
            raise ValueError(f"URL inesperada no mock: {url}")

        resp = MagicMock()
        resp.read.return_value = body
        resp.__enter__.return_value = resp
        return resp

    with patch("urllib.request.urlopen", side_effect=mock_urlopen_router):
        res = identify_compound_pubchem(pyrene_smiles)

        assert res["status"] == STATUS_SUCCESS
        assert res["cid"] == 31423
        assert res["title"] == "Pyrene"
        assert res["iupac_name"] == "pyrene"
        assert res["pubchem_url"] == "https://pubchem.ncbi.nlm.nih.gov/compound/31423"
        assert "Benzo[def]phenanthrene" in res["synonyms"]
        assert "Pyren" in res["trade_names"] or "Pyren" in res["synonyms"]

        # Confirma que códigos de catálogo e CAS foram filtrados
        all_returned = res["trade_names"] + res["synonyms"]
        assert not any("129-00-0" in s for s in all_returned)
        assert not any("DTXSID" in s for s in all_returned)
        assert not any("NSC" in s for s in all_returned)


# -----------------------------------------------------------------------------
# 3. Caso de Ausência de Correspondência (HTTP 404 / NOT_FOUND)
# -----------------------------------------------------------------------------
def test_identify_compound_not_found_404():
    """Moléculas inexistentes na base do PubChem devem retornar STATUS_NOT_FOUND sem inventar nomes."""
    novel_smiles = "CC(C)(C)CC1=CC=C(C=C1)C(C)(C)CC1=CC=C(C=C1)C(C)(C)C(C)(C)C(=O)O"

    err_404 = urllib.error.HTTPError(
        url="https://pubchem.ncbi.nlm.nih.gov/rest/pug/...",
        code=404,
        msg="Not Found",
        hdrs={},  # type: ignore
        fp=io.BytesIO(b'{"Fault": {"Code": "PUGREST.NotFound", "Message": "No CID found"}}'),
    )

    with patch("urllib.request.urlopen", side_effect=err_404):
        res = identify_compound_pubchem(novel_smiles)

        assert res["status"] == STATUS_NOT_FOUND
        assert res["cid"] is None
        assert res["title"] is None
        assert res["iupac_name"] is None
        assert res["trade_names"] == []
        assert res["synonyms"] == []
        assert "404" in res["error_message"] or "não encontrado" in res["error_message"]


# -----------------------------------------------------------------------------
# 4. Caso de Falha de Conexão / Timeout (CONNECTION_ERROR)
# -----------------------------------------------------------------------------
def test_identify_compound_connection_failure():
    """Falhas de rede ou DNS não devem causar crash na aplicação e devem retornar STATUS_CONNECTION_ERROR."""
    theophylline_smiles = "Cn1c(=O)c2nc[nH]c2n(C)c1=O"

    with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("getaddrinfo failed")):
        res = identify_compound_pubchem(theophylline_smiles)

        assert res["status"] == STATUS_CONNECTION_ERROR
        assert res["cid"] is None
        assert res["title"] is None
        assert "URLError" in res["error_message"] or "conectar" in res["error_message"]


# -----------------------------------------------------------------------------
# 5. Verificação do Mecanismo de Cache em Memória
# -----------------------------------------------------------------------------
def test_identify_compound_in_memory_caching():
    """Consultas sucessivas à mesma molécula devem usar o cache sem novas requisições de rede."""
    caffeine_smiles = "Cn1c(=O)c2c(ncn2C)n(C)c1=O"

    mock_resp = {
        "PropertyTable": {
            "Properties": [
                {
                    "CID": 2519,
                    "MolecularFormula": "C8H10N4O2",
                    "MolecularWeight": "194.19",
                    "IUPACName": "1,3,7-trimethylpurine-2,6-dione",
                    "Title": "Caffeine",
                }
            ]
        }
    }

    resp_mock = MagicMock()
    resp_mock.read.return_value = json.dumps(mock_resp).encode("utf-8")
    resp_mock.__enter__.return_value = resp_mock

    with patch("urllib.request.urlopen", return_value=resp_mock) as mock_call:
        # 1ª consulta -> executa requisição
        res1 = identify_compound_pubchem(caffeine_smiles)
        assert res1["status"] == STATUS_SUCCESS
        assert res1["cid"] == 2519
        assert mock_call.call_count >= 1
        initial_call_count = mock_call.call_count

        # 2ª consulta com a mesma estrutura -> deve vir do cache
        res2 = identify_compound_pubchem(caffeine_smiles)
        assert res2["status"] == STATUS_SUCCESS
        assert res2["cid"] == 2519
        assert mock_call.call_count == initial_call_count  # Nenhuma chamada de rede adicional!


# -----------------------------------------------------------------------------
# 6. Validação dos Cartões Visuais HTML
# -----------------------------------------------------------------------------
def test_build_pubchem_card_html_rendering():
    """Valida a renderização do HTML para os diferentes estados (sucesso, not found e offline)."""
    # Sucesso
    info_success = {
        "status": STATUS_SUCCESS,
        "cid": 2769,
        "title": "Cisapride",
        "iupac_name": "4-amino-5-chloro-N-[1-[3-(4-fluorophenoxy)propyl]-3-methoxypiperidin-4-yl]-2-methoxybenzamide",
        "trade_names": ["Propulsid", "Risamal"],
        "synonyms": ["Prepulsid"],
        "inchikey": "DCLAFBGRFZLKHM-UHFFFAOYSA-N",
        "pubchem_url": "https://pubchem.ncbi.nlm.nih.gov/compound/2769",
        "source": "PubChem (NCBI PUG REST API)",
    }
    html_success = build_pubchem_card_html(info_success)
    assert "Cisapride" in html_success
    assert "CID: 2769" in html_success
    assert "Propulsid" in html_success
    assert "https://pubchem.ncbi.nlm.nih.gov/compound/2769" in html_success
    assert "Fonte Oficial:" in html_success

    # Not Found
    info_not_found = {
        "status": STATUS_NOT_FOUND,
        "cid": None,
        "inchikey": "UNKNOWN-INCHIKEY-N",
        "source": "PubChem (NCBI PUG REST API)",
        "error_message": "Não encontrado",
    }
    html_nf = build_pubchem_card_html(info_not_found)
    assert "Composto Não Catalogado no PubChem" in html_nf
    assert "404 Not Found" in html_nf
    assert "UNKNOWN-INCHIKEY-N" in html_nf

    # Connection Error
    info_conn_err = {
        "status": STATUS_CONNECTION_ERROR,
        "cid": None,
        "inchikey": "SOME-KEY-N",
        "source": "PubChem (NCBI PUG REST API)",
        "error_message": "Timeout na conexão",
    }
    html_conn = build_pubchem_card_html(info_conn_err)
    assert "Consulta Externa ao PubChem Indisponível" in html_conn
    assert "Modo Local Ativo" in html_conn
    assert "O pipeline preditivo não é afetado" in html_conn

    # Invalid Structure
    html_invalid = build_pubchem_card_html({"status": STATUS_INVALID_STRUCTURE})
    assert html_invalid == ""

