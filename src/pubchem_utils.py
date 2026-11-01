"""
Módulo de Identificação Automatizada de Moléculas via PubChem (PUG REST API).

Permite consultar a base de dados pública do PubChem utilizando a estrutura
química validada e canônica (via InChIKey gerada pelo RDKit) para obter:
- Identificador PubChem (CID) com link direto;
- Nome preferencial / título;
- Nome sistemático oficial IUPAC;
- Nomes comerciais / marcas registradas identificadas;
- Sinônimos químicos e farmacológicos relevantes;
- Tratamento resiliente contra falhas de conexão, timeouts ou ausência de correspondência (404);
- Cache em memória para desempenho instantâneo em reexecuções.
"""

from __future__ import annotations

import json
import logging
import re
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Tuple, Union

from rdkit import Chem
from rdkit.Chem import inchi

from src.chem_utils import canonicalize_smiles

logger = logging.getLogger(__name__)

# Constantes de status
STATUS_SUCCESS = "SUCCESS"
STATUS_NOT_FOUND = "NOT_FOUND"
STATUS_CONNECTION_ERROR = "CONNECTION_ERROR"
STATUS_INVALID_STRUCTURE = "INVALID_STRUCTURE"

PUBCHEM_BASE_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
PUBCHEM_USER_AGENT = "BioPredicta/1.0 (AcademicVirtualScreening; mailto:biopredicta@local.dev)"

# Cache em memória para evitar chamadas de rede redundantes durante a sessão
_PUBCHEM_CACHE: Dict[str, Dict[str, Any]] = {}

# Expressões regulares para filtrar códigos de catálogo, fórmulas e registros internos
_RE_CATALOG_CODES = re.compile(
    r"^(CHEMBL|SCHEMBL|ZINC|AKOS|NSC|UNII|DTXSID|DTXCID|MLS|SMR|SR-|HMS|MCULE|Tox21|"
    r"NCGC|CCRIS|HSDB|EINECS|BRN|Prestwick|LSM-|Amb|ST|CS-|DB\d|SPECS|EN300|HY-|FT-|"
    r"RefChem|CID|CAS|EC\s|InChI|SMILES|PubChem|CID_\d+|T\s*\d+|KUC\d+|AB\d+|BDBM\d+)\b",
    re.IGNORECASE,
)
_RE_REGISTRY_NUMBERS = re.compile(r"^(\d{2,7}-\d{2}-\d|\d{3}-\d{3}-\d|[A-Z0-9]{8,14})$")
_RE_MOLECULAR_FORMULA = re.compile(r"^C\d+H\d+[A-Za-z0-9]*$")


def clear_pubchem_cache() -> None:
    """Limpa o cache em memória de consultas ao PubChem."""
    _PUBCHEM_CACHE.clear()


def clean_and_categorize_synonyms(
    raw_synonyms: List[str],
    title: Optional[str] = None,
    iupac_name: Optional[str] = None,
    max_trade_names: int = 8,
    max_synonyms: int = 12,
) -> Tuple[List[str], List[str]]:
    """
    Filtra códigos de catálogos e números de registro, categorizando os sinônimos
    em possíveis nomes comerciais/marcas e sinônimos químicos gerais.
    """
    if not raw_synonyms:
        return [], []

    trade_names: List[str] = []
    chemical_synonyms: List[str] = []
    seen_lower = set()

    # Adiciona título e IUPAC aos já vistos para não duplicar
    if title:
        seen_lower.add(title.strip().lower())
    if iupac_name:
        seen_lower.add(iupac_name.strip().lower())

    for syn in raw_synonyms:
        clean_syn = syn.strip()
        low = clean_syn.lower()

        if not clean_syn or low in seen_lower:
            continue

        # Filtros de exclusão: códigos de catálogos e identificadores numéricos
        if _RE_CATALOG_CODES.search(clean_syn):
            continue
        if _RE_REGISTRY_NUMBERS.match(clean_syn):
            continue
        if _RE_MOLECULAR_FORMULA.match(clean_syn):
            continue
        if len(clean_syn) > 120:  # Nomes sistemáticos excessivamente longos
            continue

        seen_lower.add(low)

        # Heurística para marcas/nomes comerciais:
        # Nomes curtos (<= 25 chars), 1 a 2 palavras, sem prefixos numéricos sistemáticos
        # (ex: não começar com '1,3-' ou '4-amino') e sem colchetes químicos pesados
        has_chemical_punctuation = any(c in clean_syn for c in "[]{}()=,;")
        starts_with_num_locant = bool(re.match(r"^\d+([,\-]|-[a-z])", clean_syn, re.IGNORECASE))

        if (
            len(clean_syn) <= 25
            and not has_chemical_punctuation
            and not starts_with_num_locant
            and len(clean_syn.split()) <= 2
            and not clean_syn.isupper()  # Evita acrônimos em caixa alta pura
            and len(trade_names) < max_trade_names
        ):
            trade_names.append(clean_syn)
        elif len(chemical_synonyms) < max_synonyms:
            chemical_synonyms.append(clean_syn)

    return trade_names, chemical_synonyms


def query_pubchem_api(inchikey: str, timeout: float = 3.5) -> Dict[str, Any]:
    """
    Realiza a requisição HTTP REST à PUG REST API do PubChem a partir de uma InChIKey.
    Garante tratamento seguro de HTTPError (404), URLError (offline) e timeouts.
    """
    prop_url = (
        f"{PUBCHEM_BASE_URL}/compound/inchikey/{inchikey}/property/"
        "Title,IUPACName,MolecularFormula,MolecularWeight/JSON"
    )
    req_prop = urllib.request.Request(prop_url, headers={"User-Agent": PUBCHEM_USER_AGENT})

    try:
        with urllib.request.urlopen(req_prop, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            props_list = data.get("PropertyTable", {}).get("Properties", [])
            if not props_list:
                return {
                    "status": STATUS_NOT_FOUND,
                    "cid": None,
                    "title": None,
                    "iupac_name": None,
                    "trade_names": [],
                    "synonyms": [],
                    "inchikey": inchikey,
                    "source": "PubChem (NCBI PUG REST API)",
                    "pubchem_url": None,
                    "error_message": "Estrutura não encontrada na base de dados do PubChem.",
                }
            props = props_list[0]
            cid = props.get("CID")
            title = props.get("Title")
            iupac_name = props.get("IUPACName")

    except urllib.error.HTTPError as err:
        if err.code == 404:
            return {
                "status": STATUS_NOT_FOUND,
                "cid": None,
                "title": None,
                "iupac_name": None,
                "trade_names": [],
                "synonyms": [],
                "inchikey": inchikey,
                "source": "PubChem (NCBI PUG REST API)",
                "pubchem_url": None,
                "error_message": "Nenhum composto com esta estrutura química foi encontrado no PubChem (HTTP 404).",
            }
        logger.warning(f"Erro HTTP do PubChem ao consultar {inchikey}: {err}")
        return {
            "status": STATUS_CONNECTION_ERROR,
            "cid": None,
            "title": None,
            "iupac_name": None,
            "trade_names": [],
            "synonyms": [],
            "inchikey": inchikey,
            "source": "PubChem (NCBI PUG REST API)",
            "pubchem_url": None,
            "error_message": f"Serviço PubChem retornou erro HTTP {err.code}.",
        }
    except Exception as exc:
        logger.warning(f"Falha de conexão com PubChem para {inchikey}: {exc}")
        return {
            "status": STATUS_CONNECTION_ERROR,
            "cid": None,
            "title": None,
            "iupac_name": None,
            "trade_names": [],
            "synonyms": [],
            "inchikey": inchikey,
            "source": "PubChem (NCBI PUG REST API)",
            "pubchem_url": None,
            "error_message": f"Não foi possível conectar ao PubChem ({type(exc).__name__}).",
        }

    # Consulta de sinônimos via CID obtido
    raw_syns: List[str] = []
    if cid:
        syn_url = f"{PUBCHEM_BASE_URL}/compound/cid/{cid}/synonyms/JSON"
        req_syn = urllib.request.Request(syn_url, headers={"User-Agent": PUBCHEM_USER_AGENT})
        try:
            with urllib.request.urlopen(req_syn, timeout=timeout) as resp_syn:
                syn_data = json.loads(resp_syn.read().decode("utf-8"))
                info_list = syn_data.get("InformationList", {}).get("Information", [])
                if info_list:
                    raw_syns = info_list[0].get("Synonym", [])
        except Exception as exc_syn:
            logger.debug(f"Aviso ao buscar sinônimos do CID {cid}: {exc_syn}")

    trade_names, clean_syns = clean_and_categorize_synonyms(raw_syns, title=title, iupac_name=iupac_name)

    return {
        "status": STATUS_SUCCESS,
        "cid": cid,
        "title": title,
        "iupac_name": iupac_name,
        "trade_names": trade_names,
        "synonyms": clean_syns,
        "inchikey": inchikey,
        "source": "PubChem (NCBI PUG REST API)",
        "pubchem_url": f"https://pubchem.ncbi.nlm.nih.gov/compound/{cid}" if cid else None,
        "error_message": None,
    }


def identify_compound_pubchem(
    mol_or_smiles: Union[str, Chem.Mol],
    timeout: float = 3.5,
    use_cache: bool = True,
) -> Dict[str, Any]:
    """
    Identifica automaticamente a molécula consultando a base do PubChem.
    Aceita string SMILES ou objeto Chem.Mol do RDKit.
    Valida e padroniza a estrutura antes do envio.

    Reutilizável tanto para moléculas desenhadas no Ketcher quanto inseridas manualmente por SMILES.
    """
    # 1. Validação e padronização estrutural com RDKit
    mol: Optional[Chem.Mol] = None
    canonical_smi: Optional[str] = None

    if isinstance(mol_or_smiles, str):
        clean_input = mol_or_smiles.strip()
        if not clean_input:
            return {
                "status": STATUS_INVALID_STRUCTURE,
                "cid": None,
                "title": None,
                "iupac_name": None,
                "trade_names": [],
                "synonyms": [],
                "inchikey": None,
                "canonical_smiles": "",
                "source": "PubChem (NCBI PUG REST API)",
                "pubchem_url": None,
                "error_message": "Entrada SMILES vazia fornecida.",
            }
        canonical_smi = canonicalize_smiles(clean_input)
        if canonical_smi is None:
            return {
                "status": STATUS_INVALID_STRUCTURE,
                "cid": None,
                "title": None,
                "iupac_name": None,
                "trade_names": [],
                "synonyms": [],
                "inchikey": None,
                "canonical_smiles": clean_input,
                "source": "PubChem (NCBI PUG REST API)",
                "pubchem_url": None,
                "error_message": "Estrutura química inválida ou não interpretável pelo RDKit.",
            }
        mol = Chem.MolFromSmiles(canonical_smi)

    elif isinstance(mol_or_smiles, Chem.Mol):
        mol = mol_or_smiles
        canonical_smi = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)

    else:
        return {
            "status": STATUS_INVALID_STRUCTURE,
            "cid": None,
            "title": None,
            "iupac_name": None,
            "trade_names": [],
            "synonyms": [],
            "inchikey": None,
            "canonical_smiles": str(mol_or_smiles),
            "source": "PubChem (NCBI PUG REST API)",
            "pubchem_url": None,
            "error_message": f"Tipo não suportado: {type(mol_or_smiles)}.",
        }

    if mol is None or mol.GetNumAtoms() == 0:
        return {
            "status": STATUS_INVALID_STRUCTURE,
            "cid": None,
            "title": None,
            "iupac_name": None,
            "trade_names": [],
            "synonyms": [],
            "inchikey": None,
            "canonical_smiles": canonical_smi or "",
            "source": "PubChem (NCBI PUG REST API)",
            "pubchem_url": None,
            "error_message": "Estrutura química sem átomos válidos.",
        }

    # 2. Geração da InChIKey padronizada
    try:
        inchikey = inchi.MolToInchiKey(mol)
    except Exception as exc_ik:
        logger.error(f"Erro ao gerar InChIKey via RDKit: {exc_ik}")
        return {
            "status": STATUS_INVALID_STRUCTURE,
            "cid": None,
            "title": None,
            "iupac_name": None,
            "trade_names": [],
            "synonyms": [],
            "inchikey": None,
            "canonical_smiles": canonical_smi,
            "source": "PubChem (NCBI PUG REST API)",
            "pubchem_url": None,
            "error_message": f"Falha na geração de InChIKey: {exc_ik}.",
        }

    # 3. Consulta ao cache em memória
    if use_cache and inchikey in _PUBCHEM_CACHE:
        cached_result = dict(_PUBCHEM_CACHE[inchikey])
        cached_result["canonical_smiles"] = canonical_smi
        return cached_result

    # 4. Consulta remota à PUG REST API
    res = query_pubchem_api(inchikey, timeout=timeout)
    res["canonical_smiles"] = canonical_smi

    if use_cache:
        _PUBCHEM_CACHE[inchikey] = res

    return res


def build_pubchem_card_html(info: Dict[str, Any]) -> str:
    """
    Constrói um cartão HTML responsivo e integrado para apresentação das informações
    de identificação pública obtidas no PubChem.
    """
    status = info.get("status")
    cid = info.get("cid")
    title = info.get("title") or "Não catalogado"
    iupac = info.get("iupac_name")
    trade_names = info.get("trade_names") or []
    synonyms = info.get("synonyms") or []
    inchikey = info.get("inchikey") or "N/A"
    source = info.get("source", "PubChem (NCBI PUG REST API)")
    url = info.get("pubchem_url")

    # Caso 1: Sucesso na identificação
    if status == STATUS_SUCCESS and cid is not None:
        trade_badges_html = ""
        if trade_names:
            items = "".join(
                f'<span style="background:rgba(56, 189, 248, 0.12); color:#38BDF8; border:1px solid rgba(56, 189, 248, 0.3); '
                f'border-radius:12px; padding:3px 10px; font-size:0.77rem; font-weight:600; margin:2px 5px 3px 0; display:inline-block;">{tn}</span>'
                for tn in trade_names
            )
            trade_badges_html = f"""
            <div style="margin-top:0.6rem;">
                <div style="font-size:0.76rem; text-transform:uppercase; color:#94A3B8; font-weight:600; letter-spacing:0.5px; margin-bottom:0.25rem;">
                    🏢 Possíveis Nomes Comerciais / Denominações:
                </div>
                <div style="display:flex; flex-wrap:wrap; gap:0.2rem;">{items}</div>
            </div>
            """

        syn_badges_html = ""
        if synonyms:
            items_syn = "".join(
                f'<span style="background:rgba(255, 255, 255, 0.05); color:#CBD5E1; border:1px solid rgba(255, 255, 255, 0.1); '
                f'border-radius:12px; padding:2px 8px; font-size:0.75rem; margin:2px 4px 3px 0; display:inline-block;">{s}</span>'
                for s in synonyms[:10]
            )
            syn_badges_html = f"""
            <div style="margin-top:0.6rem;">
                <div style="font-size:0.76rem; text-transform:uppercase; color:#94A3B8; font-weight:600; letter-spacing:0.5px; margin-bottom:0.25rem;">
                    🏷️ Sinônimos Químicos e Farmacológicos:
                </div>
                <div style="display:flex; flex-wrap:wrap; gap:0.2rem;">{items_syn}</div>
            </div>
            """

        iupac_block = (
            f'<div style="font-size:0.82rem; color:#CBD5E1; margin-top:0.35rem; line-height:1.4;">'
            f'<strong style="color:#FFFFFF;">Nome Sistemático IUPAC:</strong> <code>{iupac}</code>'
            f'</div>'
            if iupac
            else ""
        )

        link_block = (
            f'<a href="{url}" target="_blank" style="color:#00D2BE; text-decoration:none; font-size:0.82rem; font-weight:600; display:inline-flex; align-items:center; gap:4px;">'
            f'🔗 Acessar Registro Completo no PubChem (CID {cid}) &rarr;</a>'
            if url
            else ""
        )

        return f"""
        <div class="metric-card" style="padding:1.1rem 1.3rem; border-left:4px solid #38BDF8; margin-bottom:1.1rem; background:#181D26;">
            <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:0.5rem; margin-bottom:0.5rem; border-bottom:1px solid rgba(255, 255, 255, 0.07); padding-bottom:0.5rem;">
                <div style="display:flex; align-items:center; gap:8px;">
                    <span style="font-size:1.15rem;">🧬</span>
                    <span style="color:#38BDF8; font-weight:700; font-size:1.02rem;">
                        Identificação Automática (PubChem): <strong>{title}</strong>
                    </span>
                </div>
                <div style="display:flex; align-items:center; gap:8px;">
                    <span style="background:rgba(56, 189, 248, 0.15); color:#38BDF8; border:1px solid rgba(56, 189, 248, 0.35); border-radius:14px; padding:3px 12px; font-size:0.80rem; font-weight:700; font-family:'JetBrains Mono',monospace;">
                        CID: {cid}
                    </span>
                </div>
            </div>
            
            {iupac_block}
            {trade_badges_html}
            {syn_badges_html}
            
            <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:0.5rem; margin-top:0.85rem; padding-top:0.55rem; border-top:1px solid rgba(255, 255, 255, 0.06); font-size:0.77rem; color:#94A3B8;">
                <span>📚 <strong>Fonte Oficial:</strong> {source} · InChIKey: <code>{inchikey}</code></span>
                {link_block}
            </div>
        </div>
        """

    # Caso 2: Não encontrado no PubChem (404)
    elif status == STATUS_NOT_FOUND:
        return f"""
        <div class="metric-card" style="padding:0.95rem 1.2rem; border-left:4px solid #64748B; margin-bottom:1.1rem; background:#181D26;">
            <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:0.5rem; margin-bottom:0.4rem;">
                <div style="display:flex; align-items:center; gap:8px;">
                    <span style="font-size:1.1rem;">🔍</span>
                    <span style="color:#94A3B8; font-weight:700; font-size:0.95rem;">
                        Identificação Pública: Composto Não Catalogado no PubChem
                    </span>
                </div>
                <span style="background:rgba(100, 116, 139, 0.2); color:#94A3B8; border:1px solid rgba(100, 116, 139, 0.3); border-radius:14px; padding:2px 10px; font-size:0.78rem; font-weight:600;">
                    404 Not Found
                </span>
            </div>
            <p style="font-size:0.83rem; color:#CBD5E1; margin:0.35rem 0 0.5rem 0; line-height:1.45;">
                Nenhum registro com estrutura química equivalente foi localizado na base pública do PubChem para a InChIKey <code>{inchikey}</code>. 
                A molécula é processada pelo BioPredicta como <strong>composto inédito ou personalizado</strong>. Nenhuma correspondência incerta foi inferida.
            </p>
            <div style="font-size:0.75rem; color:#64748B; border-top:1px solid rgba(255, 255, 255, 0.05); padding-top:0.45rem;">
                📚 <strong>Fonte Consultada:</strong> {source}
            </div>
        </div>
        """

    # Caso 3: Erro de conexão / timeout
    elif status == STATUS_CONNECTION_ERROR:
        err_msg = info.get("error_message") or "Falha de conexão com os servidores do NCBI."
        return f"""
        <div class="metric-card" style="padding:0.95rem 1.2rem; border-left:4px solid #F59E0B; margin-bottom:1.1rem; background:#181D26;">
            <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:0.5rem; margin-bottom:0.4rem;">
                <div style="display:flex; align-items:center; gap:8px;">
                    <span style="font-size:1.1rem;">⚠️</span>
                    <span style="color:#F59E0B; font-weight:700; font-size:0.95rem;">
                        Consulta Externa ao PubChem Indisponível (Modo Local Ativo)
                    </span>
                </div>
                <span style="background:rgba(245, 158, 11, 0.15); color:#F59E0B; border:1px solid rgba(245, 158, 11, 0.3); border-radius:14px; padding:2px 10px; font-size:0.78rem; font-weight:600;">
                    Serviço Offline / Timeout
                </span>
            </div>
            <p style="font-size:0.83rem; color:#CBD5E1; margin:0.35rem 0 0.5rem 0; line-height:1.45;">
                Não foi possível conectar à API pública do PubChem ({err_msg}). 
                <strong>O pipeline preditivo não é afetado:</strong> as predições de afinidade A<sub>2A</sub>, triagem hERG, propriedades de Lipinski e domínio de aplicabilidade seguem operando normalmente via modelos locais.
            </p>
            <div style="font-size:0.75rem; color:#94A3B8; border-top:1px solid rgba(255, 255, 255, 0.05); padding-top:0.45rem;">
                📚 <strong>Serviço:</strong> {source} · Tentativa com InChIKey: <code>{inchikey}</code>
            </div>
        </div>
        """

    # Caso 4: Estrutura inválida
    return ""

