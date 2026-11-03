"""
Suíte de testes automatizados para a Página de Tutorial: Como Utilizar a Plataforma (Fase 5.1 — Passo 1).

Testa:
1. Execução de render_como_utilizar() sem exceções em ambiente Streamlit.
2. Presença explícita de todos os 7 módulos obrigatórios de conteúdo.
3. Conformidade estrita dos limiares de decisão (A2A: 5.0, 6.5; hERG: 30%, 70%).
4. Presença das ressalvas científicas obrigatórias (não-substituição clínica, não-causalidade SHAP,
   natureza do eixo Y e distinção entre Tanimoto ECFP4 e scaffold de Bemis-Murcko).
5. Instrução operacional correta do botão 'Apply' no Ketcher.
6. Política de capacidade máxima de 10 registros FIFO e comparação de 2 a 5 moléculas.
7. Integração e consistência da rota em app.py.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
import streamlit as st

from src.ad_utils import (
    PROVENANCE_LEVEL_1_STRUCTURE,
    PROVENANCE_LEVEL_2_FEATURES,
    PROVENANCE_LEVEL_3_FINGERPRINT,
    PROVENANCE_LEVEL_4_NO_EXACT_MATCH,
)
from src.ui.pages.p6_como_utilizar import render_como_utilizar


class TestRenderComoUtilizarExecution:
    """Verificação da execução funcional da página de tutorial."""

    def test_render_como_utilizar_runs_without_exceptions(self):
        """A função render_como_utilizar deve executar do início ao fim sem levantar exceções."""
        try:
            render_como_utilizar()
        except Exception as exc:
            pytest.fail(f"render_como_utilizar() levantou exceção inesperada: {exc}")


class TestTutorialModulesContent:
    """Verificação do conteúdo obrigatório dos 7 módulos no código-fonte da página."""

    @pytest.fixture
    def page_source(self) -> str:
        """Lê o código-fonte de p6_como_utilizar.py para inspeção de conformidade documental."""
        page_path = Path(__file__).resolve().parents[1] / "src" / "ui" / "pages" / "p6_como_utilizar.py"
        with open(page_path, "r", encoding="utf-8") as f:
            return f.read()

    def test_modulo_1_plataforma_e_ressalva_experimental(self, page_source: str):
        """Módulo 1 deve abordar A2A, hERG, QT e ressalva de não substituição clínica."""
        assert "Módulo 1" in page_source
        assert "A2A" in page_source or "ADORA2A" in page_source
        assert "hERG" in page_source or "Kv11.1" in page_source
        assert "Torsades de Pointes" in page_source or "arritmias" in page_source
        assert "não substituem ensaios experimentais in vitro" in page_source

    def test_modulo_2_metodos_de_entrada_e_ketcher_apply(self, page_source: str):
        """Módulo 2 deve detalhar os 3 métodos de entrada e instrução do botão Apply no Ketcher."""
        assert "Módulo 2" in page_source
        assert "Controles de Referência" in page_source or "Compostos de Referência" in page_source
        assert "Istradefilina" in page_source
        assert "Terfenadina" in page_source
        assert "Cafeína" in page_source
        assert "SMILES" in page_source
        assert "Ketcher" in page_source
        # Instrução crítica do botão Apply no Ketcher
        assert "Apply" in page_source
        assert "canto inferior direito" in page_source

    def test_modulo_3_pipeline_computacional(self, page_source: str):
        """Módulo 3 deve resumir higienização RDKit, featurização 1030-D, modelos e AD."""
        assert "Módulo 3" in page_source
        assert "RDKit" in page_source
        assert "1.030" in page_source
        assert "XGBoost" in page_source
        assert "Random Forest" in page_source
        assert "Domínio de Aplicabilidade" in page_source

    def test_modulo_4_limiares_canonicos_e_ad_niveis(self, page_source: str):
        """Módulo 4 deve apresentar os limiares canônicos sem ambiguidades e os 4 níveis de AD."""
        assert "Módulo 4" in page_source
        # Limiares de A2A (alta >= 6.5, moderada 5.0 a 6.5, baixa < 5.0)
        assert "pIC₅₀ ≥ 6,5" in page_source
        assert "5,0 ≤ pIC₅₀" in page_source and "6,5" in page_source
        assert "pIC₅₀ &lt; 5,0" in page_source or "pIC₅₀ < 5,0" in page_source
        # Limiares de hERG (baixo < 30%, moderado 30% a 70% inclusive, alto > 70%)
        assert "P(hERG=1) &lt; 30%" in page_source or "P(hERG=1) < 30%" in page_source
        assert "30% até 70%, inclusive" in page_source
        assert "P(hERG=1) &gt; 70%" in page_source or "P(hERG=1) > 70%" in page_source
        # Advertência sobre escore hERG não ser probabilidade clínica calibrada
        assert "NÃO uma probabilidade clínica calibrada de cardiotoxicidade" in page_source
        # Níveis de AD (importados e referenciados no f-string)
        assert "PROVENANCE_LEVEL_1_STRUCTURE" in page_source
        assert "PROVENANCE_LEVEL_2_FEATURES" in page_source
        assert "PROVENANCE_LEVEL_3_FINGERPRINT" in page_source
        assert "PROVENANCE_LEVEL_4_NO_EXACT_MATCH" in page_source
        assert "sem ser confundido com erro de processamento" in page_source

    def test_modulo_5_visualizacao_multiobjetivo_e_quadrantes(self, page_source: str):
        """Módulo 5 deve explicar os eixos, advertência do eixo Y e os 4 quadrantes."""
        assert "Módulo 5" in page_source
        assert "1 - P(hERG=1)" in page_source or "1 - P" in page_source
        assert "não constitui uma medida clínica direta de segurança" in page_source
        assert "Quadrante 1 (Zona Ideal)" in page_source
        assert "Quadrante 2 (Baixo Interesse)" in page_source
        assert "Quadrante 3 (Descarte)" in page_source
        assert "Quadrante 4 (Alerta Crítico)" in page_source
        assert "Zona de Otimização" in page_source

    def test_modulo_6_shap_e_ressalva_nao_causalidade(self, page_source: str):
        """Módulo 6 deve explicar contribuições SHAP e ressaltar ausência de causalidade biológica direta."""
        assert "Módulo 6" in page_source
        assert "SHAP" in page_source
        assert "TreeExplainer" in page_source
        assert "não demonstram causalidade farmacológica" in page_source
        assert "Top 10" in page_source
        assert "Farmacofóricos" in page_source or "farmacóforos" in page_source
        assert "SVG" in page_source or "subambientes" in page_source

    def test_modulo_7_historico_fifo_e_tanimoto_scaffold(self, page_source: str):
        """Módulo 7 deve explicitar limite FIFO de 10 moléculas, comparação 2-5 e ressalva ECFP4 vs Scaffold."""
        assert "Módulo 7" in page_source
        assert "10 análises concluídas" in page_source
        assert "FIFO" in page_source or "First-In, First-Out" in page_source
        assert "2 a 5" in page_source
        assert "referência" in page_source or "Referência" in page_source
        assert "Tanimoto" in page_source
        assert "scaffold" in page_source
        assert "NÃO comprovam identidade ou conservação de scaffold químico" in page_source


class TestAppNavigationIntegration:
    """Valida a integração da rota como-utilizar em app.py via AST determinístico."""

    @pytest.fixture
    def app_source(self) -> str:
        """Lê o código-fonte de app.py."""
        app_path = Path(__file__).resolve().parents[1] / "app.py"
        with open(app_path, "r", encoding="utf-8") as f:
            return f.read()

    def test_page_como_utilizar_connected_to_render_como_utilizar(self, app_source: str):
        """Verifica se page_como_utilizar() em app.py importa e chama render_como_utilizar."""
        parsed = ast.parse(app_source)
        found_fn = False
        calls_render = False

        for node in ast.walk(parsed):
            if isinstance(node, ast.FunctionDef) and node.name == "page_como_utilizar":
                found_fn = True
                fn_source = ast.unparse(node)
                if "render_como_utilizar" in fn_source:
                    calls_render = True
                break

        assert found_fn, "Função page_como_utilizar não encontrada em app.py"
        assert calls_render, "page_como_utilizar em app.py deve invocar render_como_utilizar"

    def test_pages_navigation_preserves_como_utilizar_route(self, app_source: str):
        """Garante que a rota 'como-utilizar' permanece registrada com o título e ícone esperados."""
        assert 'url_path="como-utilizar"' in app_source
        assert 'title="Como Utilizar"' in app_source
        assert 'icon="📖"' in app_source


class TestPasso11Corrections:
    """Validações pontuais e rigorosas das 6 correções exigidas na Fase 5.1 — Passo 1.1."""

    @pytest.fixture
    def page_source(self) -> str:
        page_path = Path(__file__).resolve().parents[1] / "src" / "ui" / "pages" / "p6_como_utilizar.py"
        with open(page_path, "r", encoding="utf-8") as f:
            return f.read()

    def test_correcao_1_abordagem_hibrida_em_todos_os_modulos(self, page_source: str):
        """Todos os 7 módulos devem possuir apresentação essencial e aprofundamento técnico sob demanda."""
        for i in range(1, 8):
            assert f"Módulo {i}" in page_source, f"Módulo {i} não encontrado"
        assert page_source.count("#### Apresentação Essencial") == 7
        assert page_source.count("🔬 Aprofundamento Técnico") == 7

    def test_correcao_2_tabela_unica_de_limiares(self, page_source: str):
        """Módulo 4 deve apresentar uma tabela única visual consolidando A2A e hERG com faixas corretas."""
        assert "Tabela Unificada de Classificação e Limiares Operacionais" in page_source
        assert "Faixa nanomolar" in page_source
        assert "Faixa micromolar" in page_source
        assert "Faixa micromolar ou superior" in page_source

    def test_correcao_3_descricao_canonica_ad_nivel_4(self, page_source: str):
        """Nível 4 deve registrar ausência de correspondência exata sem extrapolações de validade/invalidade."""
        assert "não foi encontrada uma correspondência exata nos critérios de proveniência" in page_source
        assert "sem ser confundido com erro de processamento" in page_source
        # Proibições expressas da especificação:
        assert "não invalida o resultado" not in page_source
        assert "não significa que a predição é inválida" not in page_source

    def test_correcao_4_terminologia_cientifica_precisa(self, page_source: str):
        """Terminologia farmacológica e estatística precisa em toda a página."""
        assert "Potência predita frente ao receptor $A_{2A}$" in page_source
        assert "Escore de classificação hERG" in page_source
        assert "Perfil Computacional Favorável" in page_source
        assert "fração de votos das árvores do classificador Random Forest" in page_source
        assert "afinidade inibitória" not in page_source
        assert "afinidade global de ligação" not in page_source

    def test_correcao_5_smax_referencia_exploratoria(self, page_source: str):
        """S_max deve ser contextualizado como contínuo e a referência ~0,30 como observacional exploratória."""
        assert "métrica diagnóstica contínua entre 0 e 1" in page_source
        assert "não implementa um limiar rígido de 0,30" in page_source
        assert "não constitui um limiar formal de validade ou rejeição" in page_source
        assert "regime de extrapolação matemática acentuada" not in page_source

    def test_correcao_6_exportacao_estrita_em_pages_init(self):
        """src.ui.pages deve exportar render_como_utilizar e não deve exportar render_comparacao."""
        import src.ui.pages as pages_mod

        assert hasattr(pages_mod, "render_como_utilizar")
        assert not hasattr(pages_mod, "render_comparacao")
        assert "render_como_utilizar" in pages_mod.__all__
        assert "render_comparacao" not in pages_mod.__all__


class TestNavigationLinkTargets:
    """Verificação rigorosa de que os links de Como Utilizar apontam para páginas registradas."""

    def test_app_registers_pages_registry_in_session_state(self):
        """Verifica estaticamente que app.py registra pages_registry em st.session_state."""
        app_path = Path(__file__).resolve().parents[1] / "app.py"
        app_code = app_path.read_text(encoding="utf-8")

        assert 'pages_registry = {' in app_code
        assert 'st.session_state["pages_registry"] = pages_registry' in app_code
        # Verifica que todos os destinos do tutorial estão no registro
        assert '"nova-analise": page_obj_nova_analise' in app_code
        assert '"comparacao": page_obj_comparacao' in app_code
        assert '"fundamentos": page_obj_fundamentos' in app_code

    def test_get_page_target_resolves_registered_st_page_objects(self):
        """Valida que _get_page_target recupera o objeto st.Page de session_state."""
        from src.ui.pages.p6_como_utilizar import _get_page_target

        dummy_page_nova = st.Page(lambda: None, title="Nova Análise", icon="🧪", url_path="nova-analise")
        dummy_page_comp = st.Page(lambda: None, title="Comparação", icon="⚖️", url_path="comparacao")
        dummy_page_fund = st.Page(lambda: None, title="Fundamentos", icon="🔬", url_path="fundamentos")

        st.session_state["pages_registry"] = {
            "nova-analise": dummy_page_nova,
            "comparacao": dummy_page_comp,
            "fundamentos": dummy_page_fund,
        }

        assert _get_page_target("nova-analise") is dummy_page_nova
        assert _get_page_target("comparacao") is dummy_page_comp
        assert _get_page_target("fundamentos") is dummy_page_fund

    def test_render_como_utilizar_with_registered_pages(self):
        """Valida que render_como_utilizar executa perfeitamente quando os objetos st.Page estão registrados."""
        from src.ui.pages.p6_como_utilizar import render_como_utilizar

        dummy_page = st.Page(lambda: None, title="Teste", icon="🧪", url_path="teste")
        st.session_state["pages_registry"] = {
            "nova-analise": dummy_page,
            "comparacao": dummy_page,
            "fundamentos": dummy_page,
        }

        try:
            render_como_utilizar()
        except Exception as exc:
            pytest.fail(f"render_como_utilizar() falhou com st.Page registrados: {exc}")
