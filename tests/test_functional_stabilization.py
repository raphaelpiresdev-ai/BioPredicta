"""
Suíte de Testes de Estabilização Funcional — MultiTarget BioPredict.

Valida de ponta a ponta as correções estruturais dos 4 eixos críticos:
1. Navegação multipágina (resolução de objetos st.Page em todas as rotas e links);
2. Renderização HTML limpa (ausência de blocos indentados >= 4 espaços / código HTML literal
   e garantia de balanceamento de tags <div> por chamada st.markdown);
3. Fluxo completo da página Nova Análise (moléculas de acesso rápido, sincronização de
   field_smiles_text e input_smiles_field, botão Limpar, ativação pelo histórico,
   análise por SMILES e análise via Molfile V2000 do editor Ketcher);
4. Fluxo completo da página Comparação integrada ao history_manager (sincronização
   dinâmica de novas moléculas, resiliência à remoção da molécula de referência e
   renderização unificada dos cartões comparativos).
"""

from __future__ import annotations

import ast
from pathlib import Path
import re
from typing import Any, Dict, List

import pytest
from rdkit import Chem
import streamlit as st

from src.chem_utils import canonicalize_smiles
from src.ui.components import get_page_target
from src.ui.history_manager import get_history, init_history
from src.ui.pages.p2_nova_analise import (
    PRESET_REFERENCE_COMPOUNDS,
    load_models_bundle_fallback,
    render_nova_analise,
    select_preset_molecule,
)
from src.ui.pages.p3_resultados import render_resultados
from src.ui.pages.p4_interpretacao import render_interpretacao
from src.ui.pages.p5_comparacao import render_comparacao
from src.ui.pages.p6_como_utilizar import render_como_utilizar

ROOT_DIR = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def isolated_session_state(monkeypatch):
    """Isola st.session_state em um dicionário limpo para cada teste."""
    clean_state: Dict[str, Any] = {}
    monkeypatch.setattr(st, "session_state", clean_state)
    # Silencia abertura de rede externa ao PubChem durante testes funcionais locais
    monkeypatch.setattr(
        "src.ui.pages.p2_nova_analise.identify_compound_pubchem",
        lambda smi, **kwargs: {"status": "NOT_FOUND", "cid": None, "title": None},
    )
    yield clean_state


@pytest.fixture(scope="module")
def real_models_bundle() -> Dict[str, Any]:
    """Carrega o pacote real de modelos (.joblib) uma única vez para os testes integrados."""
    return load_models_bundle_fallback()


# =============================================================================
# 1. NAVEGAÇÃO MULTIPÁGINA (app.py + todas as páginas de UI)
# =============================================================================
class TestGlobalNavigationStabilization:
    """Garante que nenhuma página utilize strings soltas não resolvidas em st.page_link / st.switch_page."""

    def test_all_page_link_and_switch_page_calls_use_get_page_target(self):
        py_files = [ROOT_DIR / "app.py"] + sorted((ROOT_DIR / "src").rglob("*.py"))
        checked_calls = 0

        for path in py_files:
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and getattr(node.func, "attr", "") in ("page_link", "switch_page"):
                    checked_calls += 1
                    assert len(node.args) >= 1, f"Chamada sem destino em {path}:{node.lineno}"
                    first_arg = node.args[0]
                    assert isinstance(first_arg, ast.Call), (
                        f"{path}:{node.lineno} passa argumento direto {ast.unparse(first_arg)} "
                        "em vez de get_page_target(...)"
                    )
                    fn_name = getattr(first_arg.func, "id", "") or getattr(first_arg.func, "attr", "")
                    assert fn_name in ("get_page_target", "_get_page_target"), (
                        f"{path}:{node.lineno} usa {fn_name} em vez de get_page_target"
                    )

        assert checked_calls >= 14

    def test_all_registered_routes_resolve_to_st_page_instances(self):
        expected_keys = [
            "",
            "visao_geral",
            "visao-geral",
            "nova_analise",
            "nova-analise",
            "resultados",
            "interpretacao",
            "comparacao",
            "como_utilizar",
            "como-utilizar",
            "fundamentos",
            "governanca",
            "sobre",
        ]
        # Verifica presença estática de todas as chaves em app.py
        app_code = (ROOT_DIR / "app.py").read_text(encoding="utf-8")
        for key in expected_keys:
            assert f'"{key}":' in app_code, f"Chave de rota '{key}' ausente do pages_registry em app.py"

        # Simula registro real de objetos st.Page e valida get_page_target
        registry = {
            k: st.Page(lambda: None, title=f"P_{i}", url_path=f"p-{i}")
            for i, k in enumerate(expected_keys)
        }
        st.session_state["pages_registry"] = registry

        for key in expected_keys:
            resolved = get_page_target(key)
            assert resolved is registry[key]


# =============================================================================
# 2. RENDERIZAÇÃO HTML SEM BLOCOS DE CÓDIGO LITERAL (CommonMark Indentation)
# =============================================================================
class TestHtmlCardRenderingIntegrity:
    """Verifica que nenhum cartão HTML possui indentação >= 4 espaços, linhas em branco internas ou <div> órfão."""

    @staticmethod
    def _assert_clean_html_emissions(html_calls: List[str], context_name: str) -> None:
        assert len(html_calls) > 0, f"Nenhuma chamada HTML capturada em {context_name}"
        for idx, snippet in enumerate(html_calls):
            if "<style>" in snippet:
                continue
            lines = snippet.splitlines()
            for line_no, line in enumerate(lines, start=1):
                # CommonMark: linhas vazias dentro de bloco HTML tipo 6 encerram o bloco prematuramente
                assert line.strip() != "", (
                    f"[{context_name} call #{idx}] Linha em branco interna na linha {line_no} "
                    "quebra o bloco HTML CommonMark!"
                )
                # CommonMark: >= 4 espaços iniciais transformam HTML em <pre><code> literal
                leading_spaces = len(line) - len(line.lstrip(" "))
                assert leading_spaces < 4, (
                    f"[{context_name} call #{idx}] Linha {line_no} possui {leading_spaces} espaços iniciais "
                    f"(vira bloco de código literal): {line[:80]!r}"
                )
                assert not line.startswith("\t"), (
                    f"[{context_name} call #{idx}] Linha {line_no} inicia com tabulação!"
                )

            # Verifica balanceamento de abertura/fechamento de tags <div> dentro da mesma chamada
            open_divs = len(re.findall(r"<div(?:\s|>)", snippet))
            close_divs = len(re.findall(r"</div\s*>", snippet))
            assert open_divs == close_divs, (
                f"[{context_name} call #{idx}] Tags <div> desbalanceadas "
                f"(abertas={open_divs}, fechadas={close_divs}):\n{snippet[:250]}"
            )

    def test_all_pages_emit_valid_non_indented_html(self, monkeypatch, real_models_bundle):
        captured_html: List[str] = []
        orig_markdown = st.markdown

        def spy_markdown(body, *args, **kwargs):
            if kwargs.get("unsafe_allow_html"):
                captured_html.append(str(body))
            return orig_markdown(body, *args, **kwargs)

        monkeypatch.setattr(st, "markdown", spy_markdown)

        # 1. Estados vazios de Nova Análise, Resultados, Interpretação, Comparação e Como Utilizar
        captured_html.clear()
        render_nova_analise(models_bundle=real_models_bundle)
        self._assert_clean_html_emissions(captured_html, "p2_empty")

        captured_html.clear()
        render_resultados()
        self._assert_clean_html_emissions(captured_html, "p3_empty")

        captured_html.clear()
        render_interpretacao()
        self._assert_clean_html_emissions(captured_html, "p4_empty")

        captured_html.clear()
        render_comparacao()
        self._assert_clean_html_emissions(captured_html, "p5_empty")

        captured_html.clear()
        render_como_utilizar()
        self._assert_clean_html_emissions(captured_html, "p6_all_tabs")

        # 2. Estado de SMILES inválido em Nova Análise
        captured_html.clear()
        select_preset_molecule("c1ccccc_INVALID<script>")
        render_nova_analise(models_bundle=real_models_bundle)
        self._assert_clean_html_emissions(captured_html, "p2_invalid_smiles")
        assert any("&lt;script&gt;" in h for h in captured_html), "Entrada inválida deve ser escapada em HTML"

        # 3. Fluxo completo com 2 moléculas analisadas (Cafeína + Taurina) em p2, p3, p4 e p5
        orig_button = st.button

        # Analisa Cafeína
        select_preset_molecule(PRESET_REFERENCE_COMPOUNDS["☕ Cafeína (Controle A2A)"]["smiles"])
        monkeypatch.setattr(
            st,
            "button",
            lambda label, *a, **kw: True if "Analisar Molécula" in str(label) else orig_button(label, *a, **kw),
        )
        captured_html.clear()
        render_nova_analise(models_bundle=real_models_bundle)
        self._assert_clean_html_emissions(captured_html, "p2_analyzed_caffeine")

        # Analisa Taurina (2ª molécula para habilitar Comparação e botão Ativar no histórico)
        select_preset_molecule(PRESET_REFERENCE_COMPOUNDS["🌿 Taurina (Controle Negativo)"]["smiles"])
        captured_html.clear()
        render_nova_analise(models_bundle=real_models_bundle)
        self._assert_clean_html_emissions(captured_html, "p2_analyzed_taurine_with_history")
        monkeypatch.setattr(st, "button", orig_button)

        # Resultados com análise ativa
        captured_html.clear()
        render_resultados()
        self._assert_clean_html_emissions(captured_html, "p3_active_analysis")
        assert not any("$A_{2A}$" in h for h in captured_html), "LaTeX cru não deve aparecer dentro de bloco HTML"

        # Interpretação com análise ativa
        captured_html.clear()
        render_interpretacao()
        self._assert_clean_html_emissions(captured_html, "p4_active_analysis")

        # Comparação com 2 moléculas reais no histórico
        captured_html.clear()
        render_comparacao()
        self._assert_clean_html_emissions(captured_html, "p5_active_comparison")


# =============================================================================
# 3. PÁGINA NOVA ANÁLISE (Presets, Limpar, Histórico, SMILES e Ketcher)
# =============================================================================
class TestNovaAnaliseEndToEndFlows:
    """Valida todos os fluxos interativos e estados de sessão da página Nova Análise."""

    def test_quick_access_presets_populate_smiles_field_even_after_initial_empty_render(
        self, monkeypatch, real_models_bundle
    ):
        """Reproduz o bug original: render inicial vazio cria field_smiles_text='', depois clique em preset."""
        monkeypatch.setattr(st, "rerun", lambda: None)

        # Render 1: página aberta inicialmente vazia
        render_nova_analise(models_bundle=real_models_bundle)
        assert st.session_state["field_smiles_text"] == ""
        assert st.session_state["input_smiles_field"] == ""

        # Testa cada um dos 5 botões de acesso rápido
        orig_button = st.button
        for idx, (label, meta) in enumerate(PRESET_REFERENCE_COMPOUNDS.items()):
            expected_smi = meta["smiles"]
            monkeypatch.setattr(
                st,
                "button",
                lambda lbl, *a, _target=f"btn_preset_{idx}", **kw: (
                    True if kw.get("key") == _target else orig_button(lbl, *a, **kw)
                ),
            )
            # Clique no botão de acesso rápido
            render_nova_analise(models_bundle=real_models_bundle)
            monkeypatch.setattr(st, "button", orig_button)

            # Rerun subsequente: o campo field_smiles_text NÃO pode sobrescrever com string vazia
            render_nova_analise(models_bundle=real_models_bundle)
            assert st.session_state["field_smiles_text"] == expected_smi, f"Falha no preset {label}"
            assert st.session_state["input_smiles_field"] == expected_smi, f"Falha no preset {label}"
            assert st.session_state["input_mode_selector"] == "✏️ Inserir SMILES"

    def test_clear_button_resets_both_session_keys_on_rerun(self, monkeypatch, real_models_bundle):
        """Verifica que clicar em Limpar agenda _pending_smiles_update='' e zera o campo no rerun."""
        monkeypatch.setattr(st, "rerun", lambda: None)
        caffeine_smi = PRESET_REFERENCE_COMPOUNDS["☕ Cafeína (Controle A2A)"]["smiles"]
        select_preset_molecule(caffeine_smi)
        render_nova_analise(models_bundle=real_models_bundle)
        assert st.session_state["field_smiles_text"] == caffeine_smi

        orig_button = st.button
        monkeypatch.setattr(
            st,
            "button",
            lambda lbl, *a, **kw: True if kw.get("key") == "btn_clear_smiles" else orig_button(lbl, *a, **kw),
        )
        render_nova_analise(models_bundle=real_models_bundle)
        monkeypatch.setattr(st, "button", orig_button)

        # Rerun aplica _pending_smiles_update antes da criação do widget
        render_nova_analise(models_bundle=real_models_bundle)
        assert st.session_state["field_smiles_text"] == ""
        assert st.session_state["input_smiles_field"] == ""

    def test_ketcher_molfile_apply_and_analysis_flow(self, monkeypatch, real_models_bundle):
        """Simula desenho no Ketcher (Molfile V2000 com linha inicial vazia) -> Apply -> Analisar Molécula."""
        mol = Chem.MolFromSmiles("c1ccc(cc1)c2nc3ccccc3o2")
        molblock = Chem.MolToMolBlock(mol)  # Inicia com '\n     RDKit          2D\n...'
        expected_canon = canonicalize_smiles("c1ccc(cc1)c2nc3ccccc3o2")

        st.session_state["input_mode_selector"] = "🎨 Desenhar Molécula (Ketcher)"
        monkeypatch.setattr("src.ui.pages.p2_nova_analise.st_ketcher", lambda **kwargs: molblock)

        orig_button = st.button
        monkeypatch.setattr(
            st,
            "button",
            lambda lbl, *a, **kw: True if "Analisar Molécula" in str(lbl) else orig_button(lbl, *a, **kw),
        )
        render_nova_analise(models_bundle=real_models_bundle)
        monkeypatch.setattr(st, "button", orig_button)

        assert st.session_state["input_smiles_field"] == expected_canon
        assert st.session_state["field_smiles_text"] == expected_canon
        assert st.session_state["current_analysis"] is not None
        assert st.session_state["current_analysis"]["canonical_smiles"] == expected_canon
        assert st.session_state["current_analysis"]["input_method"] == "ketcher"
        assert len(get_history()) == 1

    def test_history_activate_button_updates_active_molecule_and_input_field(
        self, monkeypatch, real_models_bundle
    ):
        """Verifica que clicar em 'Ativar no Painel' no histórico atualiza current_analysis e o campo SMILES."""
        monkeypatch.setattr(st, "rerun", lambda: None)
        orig_button = st.button

        smi_caf = PRESET_REFERENCE_COMPOUNDS["☕ Cafeína (Controle A2A)"]["smiles"]
        smi_tau = PRESET_REFERENCE_COMPOUNDS["🌿 Taurina (Controle Negativo)"]["smiles"]

        # Analisa Cafeína e depois Taurina (Taurina fica ativa)
        monkeypatch.setattr(
            st,
            "button",
            lambda lbl, *a, **kw: True if "Analisar Molécula" in str(lbl) else orig_button(lbl, *a, **kw),
        )
        select_preset_molecule(smi_caf)
        render_nova_analise(models_bundle=real_models_bundle)
        select_preset_molecule(smi_tau)
        render_nova_analise(models_bundle=real_models_bundle)
        monkeypatch.setattr(st, "button", orig_button)

        assert st.session_state["current_analysis_smiles"] == smi_tau
        assert len(get_history()) == 2

        # Clica em "🔍 Ativar no Painel" para reativar a Cafeína
        monkeypatch.setattr(
            st,
            "button",
            lambda lbl, *a, **kw: True if "Ativar no Painel" in str(lbl) else orig_button(lbl, *a, **kw),
        )
        render_nova_analise(models_bundle=real_models_bundle)
        monkeypatch.setattr(st, "button", orig_button)

        # Rerun subsequente sincroniza field_smiles_text sem erro
        render_nova_analise(models_bundle=real_models_bundle)
        assert st.session_state["current_analysis_smiles"] == smi_caf
        assert st.session_state["field_smiles_text"] == smi_caf
        assert st.session_state["input_smiles_field"] == smi_caf


# =============================================================================
# 4. PÁGINA COMPARAÇÃO (Sincronização com Histórico e Estados de Seleção)
# =============================================================================
class TestComparacaoStabilizationFlows:
    """Valida a integração entre Nova Análise, history_manager e Comparação."""

    def test_comparacao_syncs_newly_analyzed_molecules_and_handles_ref_deselection(
        self, monkeypatch, real_models_bundle
    ):
        orig_button = st.button
        monkeypatch.setattr(
            st,
            "button",
            lambda lbl, *a, **kw: True if "Analisar Molécula" in str(lbl) else orig_button(lbl, *a, **kw),
        )

        smi_caf = canonicalize_smiles(PRESET_REFERENCE_COMPOUNDS["☕ Cafeína (Controle A2A)"]["smiles"])
        smi_ist = canonicalize_smiles(PRESET_REFERENCE_COMPOUNDS["💊 Istradefilina (Antagonista A2A)"]["smiles"])
        smi_tau = canonicalize_smiles(PRESET_REFERENCE_COMPOUNDS["🌿 Taurina (Controle Negativo)"]["smiles"])

        # 1. Analisa 2 moléculas e abre Comparação
        select_preset_molecule(smi_caf)
        render_nova_analise(models_bundle=real_models_bundle)
        select_preset_molecule(smi_ist)
        render_nova_analise(models_bundle=real_models_bundle)
        monkeypatch.setattr(st, "button", orig_button)

        render_comparacao()
        assert st.session_state["comp_multiselect_molecules"] == [smi_caf, smi_ist]
        assert st.session_state["comp_select_ref_molecule"] == smi_ist

        # 2. Analisa 3ª molécula (Taurina) e retorna para Comparação -> deve auto-incluir Taurina
        monkeypatch.setattr(
            st,
            "button",
            lambda lbl, *a, **kw: True if "Analisar Molécula" in str(lbl) else orig_button(lbl, *a, **kw),
        )
        select_preset_molecule(smi_tau)
        render_nova_analise(models_bundle=real_models_bundle)
        monkeypatch.setattr(st, "button", orig_button)

        render_comparacao()
        assert smi_tau in st.session_state["comp_multiselect_molecules"]
        assert len(st.session_state["comp_multiselect_molecules"]) == 3

        # 3. Usuário remove a molécula que estava como referência (Istradefilina) do multiselect
        st.session_state["comp_multiselect_molecules"] = [smi_caf, smi_tau]
        st.session_state["comp_select_ref_molecule"] = smi_ist  # Referência obsoleta fora das opções selecionadas

        # Não deve lançar ValueError/StreamlitAPIException; deve reajustar referência automaticamente
        render_comparacao()
        assert st.session_state["comp_select_ref_molecule"] in [smi_caf, smi_tau]

