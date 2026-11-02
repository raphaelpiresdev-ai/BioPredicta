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
   - Arredondamento estrito de p_ic50 e p_herg_positive para duas casas decimais.
   - Consumo dos modelos mantidos em app.state.
   - Rejeição de SMILES quimicamente inválido com HTTP 400 Bad Request.
   - Rejeição de falha de featurização com HTTP 400 Bad Request.
   - Falha por modelo ausente com HTTP 500 Internal Server Error.
   - Falha inesperada durante a inferência com HTTP 500 Internal Server Error.
3. Validação HTTP e Tratamento em Português (Pydantic / FastAPI):
   - Payload com JSON malformado / sem aspas (422) com mensagem específica em português.
   - Payload vazio (422) com mensagem em português.
   - Campo smiles ausente (422) com mensagem em português.
   - Tipo de dados incorreto (422) com mensagem em português.
   - Injeção de campos extras proibidos (422) com mensagem em português.
   - SMILES contendo apenas espaços em branco (422) com mensagem em português.
   - SMILES vazio (422) com mensagem em português.
   - SMILES excedendo limite de 1000 caracteres (422) com mensagem em português.
   - Ausência absoluta de mensagens padrão em inglês e de vazamento de tracebacks.
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
        """Verifica predição ponta a ponta com modelos reais para a cafeína com arredondamento."""
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

            # Arredondamento estrito para 2 casas decimais (valores reais cafeína)
            assert data["p_ic50"] == 4.95
            assert data["p_herg_positive"] == 0.1  # 0.10 float representation in JSON
            # Garantir no máximo 2 casas decimais na representação numérica
            assert len(str(data["p_ic50"]).split(".")[1]) <= 2
            assert len(str(data["p_herg_positive"]).split(".")[1]) <= 2

    def test_predict_uses_models_from_app_state_via_mock(self):
        """Verifica se a rota consome estritamente os modelos presentes em app.state e arredonda."""
        client = TestClient(app)
        mock_a2a = MagicMock()
        mock_a2a.predict.return_value = [6.854321]

        mock_herg = MagicMock()
        mock_herg.classes_ = [0, 1]
        mock_herg.predict_proba.return_value = [[0.803, 0.197]]

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
    """Testes de conformidade das validações HTTP e schemas Pydantic com mensagens em português (HTTP 422)."""

    def test_predict_malformed_json_returns_422_with_portuguese_message(self):
        """Verifica que requisição com JSON sintaticamente inválido (sem aspas) retorna 422 em português."""
        with TestClient(app) as client:
            raw_payload = '{"smiles": CN1C=NC2=C1C(=O)N(C(=O)N2C)C}'
            response = client.post(
                "/predict",
                content=raw_payload,
                headers={"Content-Type": "application/json"},
            )
            assert response.status_code == 422
            data = response.json()
            assert "detail" in data
            assert len(data["detail"]) > 0
            first_err = data["detail"][0]
            assert first_err["loc"] == ["body"]
            assert first_err["type"] in ("json_invalid", "json_type")
            assert first_err["msg"] == (
                "JSON inválido. Verifique a sintaxe da requisição e utilize aspas duplas nos valores de texto."
            )
            # Garantir ausência de tracebacks ou mensagens padrão em inglês
            assert "traceback" not in response.text.lower()
            assert "expecting value" not in response.text.lower()
            assert "json decode error" not in response.text.lower()

    def test_predict_empty_payload_returns_422(self):
        """Verifica que payload vazio retorna HTTP 422 Unprocessable Entity em português."""
        with TestClient(app) as client:
            response = client.post("/predict", json={})
            assert response.status_code == 422
            data = response.json()
            assert "detail" in data
            assert data["detail"][0]["msg"] == "O campo 'smiles' é obrigatório."
            assert data["detail"][0]["type"] == "missing"
            assert "field required" not in response.text.lower()

    def test_predict_missing_smiles_field_returns_422(self):
        """Verifica que payload sem o campo smiles retorna HTTP 422 com mensagem em português."""
        with TestClient(app) as client:
            response = client.post("/predict", json={"outra_coisa": "CCO"})
            assert response.status_code == 422
            data = response.json()
            assert data["detail"][0]["msg"] == "O campo 'smiles' é obrigatório."
            assert data["detail"][0]["type"] == "missing"
            assert "field required" not in response.text.lower()

    def test_predict_non_string_type_returns_422(self):
        """Verifica que tipos incompatíveis retornam HTTP 422 com mensagem em português."""
        with TestClient(app) as client:
            res_num = client.post("/predict", json={"smiles": 12345})
            assert res_num.status_code == 422
            data_num = res_num.json()
            assert data_num["detail"][0]["msg"] == "O campo 'smiles' deve ser uma string."
            assert data_num["detail"][0]["type"] == "string_type"
            assert "input should be a valid string" not in res_num.text.lower()

            res_list = client.post("/predict", json={"smiles": ["CCO"]})
            assert res_list.status_code == 422
            data_list = res_list.json()
            assert data_list["detail"][0]["msg"] == "O campo 'smiles' deve ser uma string."
            assert data_list["detail"][0]["type"] == "string_type"
            assert "input should be a valid string" not in res_list.text.lower()

    def test_predict_extra_field_returns_422(self):
        """Verifica que campos adicionais não previstos retornam HTTP 422 com mensagem em português."""
        with TestClient(app) as client:
            response = client.post("/predict", json={"smiles": "CCO", "campo_extra": "proibido"})
            assert response.status_code == 422
            data = response.json()
            assert data["detail"][0]["msg"] == "O campo adicional 'campo_extra' não é permitido."
            assert data["detail"][0]["type"] == "extra_forbidden"
            assert "extra inputs are not permitted" not in response.text.lower()

    def test_predict_whitespace_only_smiles_returns_422(self):
        """Verifica que SMILES contendo exclusivamente espaços retorna HTTP 422 em português."""
        with TestClient(app) as client:
            response = client.post("/predict", json={"smiles": "     "})
            assert response.status_code == 422
            data = response.json()
            assert data["detail"][0]["msg"] == "O campo 'smiles' não pode ser vazio ou conter apenas espaços."
            assert "string should have at least 1 character" not in response.text.lower()

    def test_predict_empty_smiles_string_returns_422(self):
        """Verifica que string vazia retorna HTTP 422 em português."""
        with TestClient(app) as client:
            response = client.post("/predict", json={"smiles": ""})
            assert response.status_code == 422
            data = response.json()
            assert data["detail"][0]["msg"] == "O campo 'smiles' não pode ser vazio ou conter apenas espaços."
            assert "string should have at least 1 character" not in response.text.lower()

    def test_predict_exceeding_max_length_returns_422(self):
        """Verifica que SMILES com mais de 1000 caracteres retorna HTTP 422 em português."""
        with TestClient(app) as client:
            smiles_gigante = "C" * 1001
            response = client.post("/predict", json={"smiles": smiles_gigante})
            assert response.status_code == 422
            data = response.json()
            assert data["detail"][0]["msg"] == (
                "O campo 'smiles' excede o comprimento máximo permitido de 1000 caracteres."
            )
            assert data["detail"][0]["type"] == "string_too_long"
            assert "string should have at most 1000 characters" not in response.text.lower()

    def test_openapi_docs_and_redoc_expose_official_identity(self):
        """Verifica que /openapi.json, /docs e /redoc expõem a identidade oficial do projeto."""
        official_identity = (
            "Plataforma de Triagem Virtual Multiobjetivo: "
            "Potência no Receptor de Adenosina A2A × Segurança Cardíaca hERG"
        )
        with TestClient(app) as client:
            resp_openapi = client.get("/openapi.json")
            assert resp_openapi.status_code == 200
            info = resp_openapi.json()["info"]
            assert info["title"] == "MultiTarget BioPredict API"
            assert info["summary"] == official_identity
            assert official_identity in info["description"]

            resp_docs = client.get("/docs")
            assert resp_docs.status_code == 200
            assert "MultiTarget BioPredict API" in resp_docs.text

            resp_redoc = client.get("/redoc")
            assert resp_redoc.status_code == 200
            assert "MultiTarget BioPredict API" in resp_redoc.text

    def test_error_logs_do_not_expose_raw_molecular_content(self, caplog):
        """Garante higienização de logs: erros HTTP 400 e 422 não registram SMILES ou payloads brutos."""
        import logging

        secret_invalid_smiles = "MOLECULA_SECRETA_INVALIDA_XYZ999"
        secret_extra_smiles = "CCO_CONFIDENCIAL_422"
        secret_malformed_token = "TOKEN_MOLECULAR_BRUTO_SEM_ASPAS"

        with caplog.at_level(logging.WARNING), TestClient(app) as client:
            r400 = client.post("/predict", json={"smiles": secret_invalid_smiles})
            assert r400.status_code == 400

            r422_extra = client.post(
                "/predict",
                json={"smiles": secret_extra_smiles, "campo_extra": "VALOR_SECRETO_EXTRA"},
            )
            assert r422_extra.status_code == 422

            r422_json = client.post(
                "/predict",
                content=f'{{"smiles": {secret_malformed_token}}}',
                headers={"Content-Type": "application/json"},
            )
            assert r422_json.status_code == 422

        combined_logs = "\n".join(rec.getMessage() for rec in caplog.records)
        assert len(caplog.records) >= 3
        assert secret_invalid_smiles not in combined_logs
        assert secret_extra_smiles not in combined_logs
        assert "VALOR_SECRETO_EXTRA" not in combined_logs
        assert secret_malformed_token not in combined_logs

