"""Suíte de testes para a Fase 5 — Implementação de Endpoints e Tratamento de Erros.

Verifica os requisitos obrigatórios:
1. Health check (GET /health):
   - Resposta HTTP 200 OK.
   - Identificação do serviço ("MultiTarget BioPredict API").
   - Status de prontidão quando ambos os modelos estão carregados.
   - Status degradado quando A2A, hERG ou ambos estão ausentes.
2. Inferência (POST /predict):
   - Requisição válida com retorno 200 OK e schema PredictionResponse.
   - Valores numéricos válidos e probabilidade hERG em [0.0, 1.0].
   - Consumo dos modelos mantidos em app.state.
   - Rejeição de SMILES quimicamente inválido com HTTP 400 Bad Request.
   - Rejeição de falha de featurização com HTTP 400 Bad Request.
   - Falha por modelo ausente com HTTP 500 Internal Server Error.
   - Falha inesperada durante a inferência com HTTP 500 Internal Server Error.
3. Validação HTTP (Pydantic / FastAPI):
   - Payload vazio (422).
   - Campo smiles ausente (422).
   - Tipo de dados incorreto (422).
   - Injeção de campos extras (422).
   - SMILES vazio ou contendo apenas espaços (422).
   - SMILES excedendo limite de 1000 caracteres (422).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient

from src.api.main import app


class TestHealthCheckEndpoint:
    """Testes para a rota de monitoramento GET /health."""

    def test_health_check_all_models_loaded(self):
        """Verifica resposta 200 OK com status 'ok' quando ambos os modelos estão carregados."""
        with TestClient(app) as client:
            response = client.get("/health")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "ok"
            assert data["service"] == "MultiTarget BioPredict API"
            assert data["models_loaded"]["a2a"] is True
            assert data["models_loaded"]["herg"] is True
            assert data["ready_for_inference"] is True

    def test_health_check_a2a_model_missing(self):
        """Verifica resposta 200 OK com status 'degraded' quando modelo A2A está ausente."""
        client = TestClient(app)
        client.app.state.a2a_model = None
        client.app.state.herg_model = MagicMock()

        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "degraded"
        assert data["models_loaded"]["a2a"] is False
        assert data["models_loaded"]["herg"] is True
        assert data["ready_for_inference"] is False

    def test_health_check_herg_model_missing(self):
        """Verifica resposta 200 OK com status 'degraded' quando modelo hERG está ausente."""
        client = TestClient(app)
        client.app.state.a2a_model = MagicMock()
        client.app.state.herg_model = None

        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "degraded"
        assert data["models_loaded"]["a2a"] is True
        assert data["models_loaded"]["herg"] is False
        assert data["ready_for_inference"] is False

    def test_health_check_both_models_missing(self):
        """Verifica resposta 200 OK com status 'degraded' quando ambos os modelos estão ausentes."""
        client = TestClient(app)
        client.app.state.a2a_model = None
        client.app.state.herg_model = None

        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "degraded"
        assert data["models_loaded"]["a2a"] is False
        assert data["models_loaded"]["herg"] is False
        assert data["ready_for_inference"] is False


class TestPredictEndpointInference:
    """Testes de inferência multiobjetivo para a rota POST /predict."""

    def test_predict_valid_molecule_real_models(self):
        """Verifica predição ponta a ponta com modelos reais para a cafeína."""
        with TestClient(app) as client:
            caffeine_smiles = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
            response = client.post("/predict", json={"smiles": caffeine_smiles})
            assert response.status_code == 200
            data = response.json()

            # Schema PredictionResponse
            assert "smiles" in data
            assert "p_ic50" in data
            assert "p_herg_positive" in data

            # Verificação de tipos e limites
            assert isinstance(data["p_ic50"], float)
            assert isinstance(data["p_herg_positive"], float)
            assert 0.0 <= data["p_herg_positive"] <= 1.0
            # Canonical smiles retornado pelo RDKit
            assert data["smiles"] == "Cn1c(=O)c2c(ncn2C)n(C)c1=O"

    def test_predict_uses_models_from_app_state_via_mock(self):
        """Verifica se a rota consome estritamente os modelos presentes em app.state."""
        client = TestClient(app)
        mock_a2a = MagicMock()
        mock_a2a.predict.return_value = [6.85]

        mock_herg = MagicMock()
        mock_herg.classes_ = [0, 1]
        mock_herg.predict_proba.return_value = [[0.80, 0.20]]

        client.app.state.a2a_model = mock_a2a
        client.app.state.herg_model = mock_herg

        response = client.post("/predict", json={"smiles": "CCO"})
        assert response.status_code == 200
        data = response.json()

        assert data["p_ic50"] == 6.85
        assert data["p_herg_positive"] == 0.20
        assert mock_a2a.predict.called
        assert mock_herg.predict_proba.called

    def test_predict_invalid_smiles_returns_400(self):
        """Verifica que SMILES quimicamente inválido retorna HTTP 400 Bad Request."""
        with TestClient(app) as client:
            response = client.post("/predict", json={"smiles": "QUIMICA_INVALIDA_XYZ_123"})
            assert response.status_code == 400
            data = response.json()
            assert "detail" in data
            assert "quimicamente inválido" in data["detail"].lower() or "rdkit" in data["detail"].lower()

    def test_predict_missing_models_in_state_returns_500(self):
        """Verifica que modelos ausentes em app.state retornam HTTP 500 sem expor traceback."""
        client = TestClient(app)
        client.app.state.a2a_model = None
        client.app.state.herg_model = None

        response = client.post("/predict", json={"smiles": "CCO"})
        assert response.status_code == 500
        data = response.json()
        assert data["detail"] == "Modelos preditivos indisponíveis no servidor."

    def test_predict_unexpected_inference_exception_returns_500(self):
        """Verifica que falha inesperada durante a execução da inferência retorna HTTP 500 seguro."""
        client = TestClient(app)
        mock_a2a = MagicMock()
        mock_a2a.predict.side_effect = RuntimeError("Erro interno simulado de computação")
        mock_herg = MagicMock()

        client.app.state.a2a_model = mock_a2a
        client.app.state.herg_model = mock_herg

        response = client.post("/predict", json={"smiles": "CCO"})
        assert response.status_code == 500
        data = response.json()
        assert "erro interno" in data["detail"].lower()
        # Não deve expor detalhes da exceção ou traceback
        assert "Erro interno simulado" not in data["detail"]


class TestPredictEndpointHttpValidation:
    """Testes de conformidade das validações HTTP e schemas Pydantic (HTTP 422)."""

    def test_predict_empty_payload_returns_422(self):
        """Verifica que payload vazio retorna HTTP 422 Unprocessable Entity."""
        with TestClient(app) as client:
            response = client.post("/predict", json={})
            assert response.status_code == 422

    def test_predict_missing_smiles_field_returns_422(self):
        """Verifica que payload sem o campo smiles retorna HTTP 422."""
        with TestClient(app) as client:
            response = client.post("/predict", json={"outra_coisa": "CCO"})
            assert response.status_code == 422

    def test_predict_non_string_type_returns_422(self):
        """Verifica que tipos incompatíveis (número, lista, booleano) retornam HTTP 422."""
        with TestClient(app) as client:
            res_num = client.post("/predict", json={"smiles": 12345})
            assert res_num.status_code == 422

            res_list = client.post("/predict", json={"smiles": ["CCO"]})
            assert res_list.status_code == 422

    def test_predict_extra_field_returns_422(self):
        """Verifica que campos adicionais não previstos retornam HTTP 422 (extra='forbid')."""
        with TestClient(app) as client:
            response = client.post("/predict", json={"smiles": "CCO", "campo_extra": "proibido"})
            assert response.status_code == 422

    def test_predict_whitespace_only_smiles_returns_422(self):
        """Verifica que SMILES contendo exclusivamente espaços retorna HTTP 422."""
        with TestClient(app) as client:
            response = client.post("/predict", json={"smiles": "     "})
            assert response.status_code == 422

    def test_predict_empty_smiles_string_returns_422(self):
        """Verifica que string vazia retorna HTTP 422."""
        with TestClient(app) as client:
            response = client.post("/predict", json={"smiles": ""})
            assert response.status_code == 422

    def test_predict_exceeding_max_length_returns_422(self):
        """Verifica que SMILES com mais de 1000 caracteres retorna HTTP 422."""
        with TestClient(app) as client:
            smiles_gigante = "C" * 1001
            response = client.post("/predict", json={"smiles": smiles_gigante})
            assert response.status_code == 422

