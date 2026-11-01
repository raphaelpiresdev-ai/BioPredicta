"""Suíte de testes para a Fase 3 — Ciclo de Vida da Aplicação (Lifespan).

Verifica os requisitos obrigatórios definidos no escopo do projeto:
1. A aplicação FastAPI pode ser importada sem carregar os modelos antecipadamente.
2. A inicialização (startup) executa o carregamento de ambos os modelos (A2A e hERG).
3. Os modelos ficam disponíveis no estado compartilhado da aplicação (app.state).
4. O carregamento ocorre uma única vez por ciclo de vida (sem recarregamentos repetidos).
5. Falha no carregamento (arquivo ausente ou erro de desserialização) aborta explicitamente o startup.
6. O encerramento (shutdown) do ciclo de vida finaliza e limpa adequadamente o estado mantido.
7. A inicialização da API não causa efeitos colaterais na aplicação Streamlit ou no núcleo científico.
8. Compatibilidade ponta a ponta com os artefatos reais serializados em models/.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI

from src.api.main import (
    CAMINHO_MODELO_A2A,
    CAMINHO_MODELO_HERG,
    app,
    lifespan,
)


@contextmanager
def execute_lifespan(application: FastAPI):
    """Executa o ciclo de vida da aplicação FastAPI de forma determinística e síncrona.

    Permite testar o gerenciador de ciclo de vida (lifespan) da aplicação Starlette/FastAPI
    de forma limpa e isolada, sem dependência de loops de eventos assíncronos externos.
    """
    context = application.router.lifespan_context(application)
    enter_coroutine = context.__aenter__()
    try:
        enter_coroutine.send(None)
    except StopIteration:
        pass

    try:
        yield application
    finally:
        exit_coroutine = context.__aexit__(None, None, None)
        try:
            exit_coroutine.send(None)
        except StopIteration:
            pass


class TestApiLifespanLifecycle:
    """Testes unitários e de integração para o ciclo de vida (lifespan) da API."""

    def test_import_app_without_early_model_loading(self):
        """Requisito 1: A importação da aplicação não deve carregar os modelos antecipadamente."""
        a2a = getattr(app.state, "a2a_model", None)
        herg = getattr(app.state, "herg_model", None)
        assert a2a is None, "Modelo A2A não deve estar carregado antes do início do ciclo de vida."
        assert herg is None, "Modelo hERG não deve estar carregado antes do início do ciclo de vida."

    def test_lifespan_loads_both_models_and_populates_app_state(self):
        """Requisitos 2 e 3: A inicialização deve carregar ambos os modelos e disponibilizá-los em app.state."""
        mock_a2a = MagicMock(name="MockA2AModel")
        mock_herg = MagicMock(name="MockHergModel")

        def mock_load_side_effect(path):
            if Path(path) == CAMINHO_MODELO_A2A:
                return mock_a2a
            elif Path(path) == CAMINHO_MODELO_HERG:
                return mock_herg
            raise ValueError(f"Caminho inesperado: {path}")

        with patch("src.api.main.load_model", side_effect=mock_load_side_effect) as mocked_load:
            with execute_lifespan(app):
                # Verifica que load_model foi chamado para os dois caminhos
                assert mocked_load.call_count == 2
                # Verifica disponibilidade no app.state
                assert app.state.a2a_model is mock_a2a
                assert app.state.herg_model is mock_herg

    def test_lifespan_load_is_executed_once_per_lifecycle(self):
        """Requisito 4: O carregamento não deve ser repetido durante o mesmo ciclo de vida."""
        mock_a2a = MagicMock(name="MockA2A")
        mock_herg = MagicMock(name="MockHERG")

        with patch("src.api.main.load_model", side_effect=[mock_a2a, mock_herg]) as mocked_load:
            with execute_lifespan(app):
                assert mocked_load.call_count == 2
                # Simula múltiplas consultas/acessos concorrentes ao app.state
                for _ in range(10):
                    _ = app.state.a2a_model
                    _ = app.state.herg_model
                # A contagem de carregamentos em disco deve permanecer estritamente 2
                assert mocked_load.call_count == 2

    def test_lifespan_failure_missing_a2a_aborts_startup(self):
        """Requisito 5: Arquivo A2A ausente deve lançar FileNotFoundError e abortar a inicialização."""
        caminho_inexistente = Path("models_inexistentes/a2a_model.joblib")
        with patch("src.api.main.CAMINHO_MODELO_A2A", caminho_inexistente):
            with pytest.raises(FileNotFoundError, match="modelo A2A não encontrado"):
                with execute_lifespan(app):
                    pass

    def test_lifespan_failure_missing_herg_aborts_startup(self):
        """Requisito 5: Arquivo hERG ausente deve lançar FileNotFoundError e abortar a inicialização."""
        caminho_inexistente = Path("models_inexistentes/herg_model.joblib")
        with patch("src.api.main.CAMINHO_MODELO_HERG", caminho_inexistente):
            with pytest.raises(FileNotFoundError, match="modelo hERG não encontrado"):
                with execute_lifespan(app):
                    pass

    def test_lifespan_failure_deserialization_aborts_startup(self):
        """Requisito 5: Erro na desserialização de modelos deve lançar RuntimeError e abortar startup."""
        with patch("src.api.main.load_model", side_effect=IOError("Arquivo corrompido")):
            with pytest.raises(RuntimeError, match="Falha na desserialização"):
                with execute_lifespan(app):
                    pass

    def test_lifespan_shutdown_cleans_up_state(self):
        """Requisito 6: O encerramento do ciclo de vida deve liberar as referências dos modelos."""
        mock_a2a = MagicMock(name="MockA2A")
        mock_herg = MagicMock(name="MockHERG")

        with patch("src.api.main.load_model", side_effect=[mock_a2a, mock_herg]):
            with execute_lifespan(app):
                assert app.state.a2a_model is mock_a2a
                assert app.state.herg_model is mock_herg

            # Após o bloco de contexto (shutdown):
            assert app.state.a2a_model is None
            assert app.state.herg_model is None

    def test_lifespan_no_side_effects_on_streamlit(self):
        """Requisito 7: A inicialização da API não deve interferir no Streamlit ou alterar módulos científicos."""
        import src.chem_utils as chem_utils
        import src.model_utils as model_utils
        from src.ui.components import get_page_target

        # Verifica integridade das assinaturas e constantes do núcleo científico
        assert hasattr(chem_utils, "FEATURE_NAMES")
        assert hasattr(chem_utils, "smiles_to_mol")
        assert hasattr(model_utils, "load_model")
        assert hasattr(model_utils, "predict")

        # Verifica funcionamento do utilitário de navegação do Streamlit
        alvo = get_page_target("nova-analise")
        assert alvo is not None

    def test_lifespan_with_real_serialized_artifacts(self):
        """Requisito 8: Teste de integração real com os artefatos serializados em models/."""
        with execute_lifespan(app):
            # Modelos reais devem estar presentes
            assert app.state.a2a_model is not None
            assert app.state.herg_model is not None
            # Modelo A2A deve ter o método de predição contínua
            assert hasattr(app.state.a2a_model, "predict")
            # Modelo hERG deve ter o método de predição probabilística
            assert hasattr(app.state.herg_model, "predict_proba")

        # Após encerramento, o estado deve ser limpo
        assert app.state.a2a_model is None
        assert app.state.herg_model is None
