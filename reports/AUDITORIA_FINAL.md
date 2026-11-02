# AUDITORIA FINAL — MULTITARGET BIOPREDICT
## Avaliação Técnico-Científica, Arquitetural, Funcional e de Qualidade de Software

---

## 1. Identificação do Projeto e Data da Auditoria

| Campo | Descrição |
| :--- | :--- |
| **Projeto** | **MultiTarget BioPredict** — Plataforma de Triagem Virtual Multiobjetivo ($\text{A}_{2\text{A}} \times \text{hERG}$) |
| **Autor do Projeto** | Raphael Pires |
| **Data da Auditoria** | `01/11/2026` (execução local iniciada às `23:11 BRT` / `2026-11-02T02:11Z`) |
| **Natureza da Atividade** | Auditoria final exclusivamente diagnóstica (**modo estritamente somente leitura**) |
| **Composição da Banca Auditora** | Engenharia de Software & Arquitetura Python · Engenharia de Validação de Machine Learning · Química Medicinal & Modelagem Molecular · Engenharia de APIs REST (FastAPI, Pydantic & AppSec) · Qualidade de Software & Testes Automatizados (QA) |
| **Ambiente Auditado** | Windows (Win32) · Python `3.12.10` (`.venv`) · RDKit `2026.9.1` · scikit-learn `1.9.1` · XGBoost `3.4.1` · SHAP `0.53.0` · Streamlit `1.65.0` · FastAPI `0.162.0` · Pydantic `2.14.2` · Starlette `1.7.0` · Uvicorn `0.54.0` · Pytest `9.1.1` |
| **Branch e Commit Base (`HEAD`)** | Branch `develop` · Commit `96d1b245e91ade260acb484e7c0e24d9a97a43c7` (*Merge pull request #13 — `feat/fase-10-endpoints-fastapi`*) |
| **Parecer Final Sintético** | **APROVADO COM RESSALVAS** (`0` achados Críticos, `0` Altos, `4` Moderados, `3` Baixos, `3` Informativos) |

---

## 2. Escopo e Metodologia

### 2.1 Escopo Auditado
A auditoria abrangeu a totalidade do sistema **MultiTarget BioPredict** em suas três camadas funcionais e artefatos correlatos:
1. **Núcleo Científico Compartilhado ([`src/`](file:///c:/Users/Raphael/Documents/BioPredicta/src/)):** Padronização estrutural e featurização molecular 1030-D ([`src/chem_utils.py`](file:///c:/Users/Raphael/Documents/BioPredicta/src/chem_utils.py)), métricas e persistência de modelos ([`src/model_utils.py`](file:///c:/Users/Raphael/Documents/BioPredicta/src/model_utils.py)), curadoria ChEMBL/TDC ([`src/data_utils.py`](file:///c:/Users/Raphael/Documents/BioPredicta/src/data_utils.py)), motor de Domínio de Aplicabilidade em 4 níveis ([`src/ad_utils.py`](file:///c:/Users/Raphael/Documents/BioPredicta/src/ad_utils.py)) e enriquecimento externo não-bloqueante via PubChem ([`src/pubchem_utils.py`](file:///c:/Users/Raphael/Documents/BioPredicta/src/pubchem_utils.py)).
2. **API RESTful de Inferência ([`src/api/`](file:///c:/Users/Raphael/Documents/BioPredicta/src/api/)):** Aplicação FastAPI e ciclo de vida `lifespan` ([`src/api/main.py`](file:///c:/Users/Raphael/Documents/BioPredicta/src/api/main.py)), tratamento centralizado e tradução de exceções HTTP ([`src/api/exception_handlers.py`](file:///c:/Users/Raphael/Documents/BioPredicta/src/api/exception_handlers.py)), roteadores [`health.py`](file:///c:/Users/Raphael/Documents/BioPredicta/src/api/routers/health.py) e [`predict.py`](file:///c:/Users/Raphael/Documents/BioPredicta/src/api/routers/predict.py), contratos Pydantic v2 ([`schemas/health.py`](file:///c:/Users/Raphael/Documents/BioPredicta/src/api/schemas/health.py) e [`schemas/prediction.py`](file:///c:/Users/Raphael/Documents/BioPredicta/src/api/schemas/prediction.py)) e serviço de orquestração preditiva ([`services/prediction_service.py`](file:///c:/Users/Raphael/Documents/BioPredicta/src/api/services/prediction_service.py)).
3. **Aplicação Web Analítica Streamlit ([`app.py`](file:///c:/Users/Raphael/Documents/BioPredicta/app.py) e [`src/ui/`](file:///c:/Users/Raphael/Documents/BioPredicta/src/ui/)):** Roteamento multipágina, design system dark ([`theme.py`](file:///c:/Users/Raphael/Documents/BioPredicta/src/ui/theme.py)), componentes gráficos e comparativos ([`components.py`](file:///c:/Users/Raphael/Documents/BioPredicta/src/ui/components.py)), catálogos farmacofóricos ([`catalogs.py`](file:///c:/Users/Raphael/Documents/BioPredicta/src/ui/catalogs.py)), motor de explicabilidade química local SHAP ([`interpretation_engine.py`](file:///c:/Users/Raphael/Documents/BioPredicta/src/ui/interpretation_engine.py)), gerenciador FIFO de sessão ([`history_manager.py`](file:///c:/Users/Raphael/Documents/BioPredicta/src/ui/history_manager.py)) e páginas [`p2_nova_analise.py`](file:///c:/Users/Raphael/Documents/BioPredicta/src/ui/pages/p2_nova_analise.py) a [`p6_como_utilizar.py`](file:///c:/Users/Raphael/Documents/BioPredicta/src/ui/pages/p6_como_utilizar.py).
4. **Artefatos Científicos, Dependências, Testes e Documentação:** Modelos `.joblib`, metadados `.json`, datasets curados `.csv` e `.parquet`, suíte de 17 arquivos em [`tests/`](file:///c:/Users/Raphael/Documents/BioPredicta/tests/), [`requirements.txt`](file:///c:/Users/Raphael/Documents/BioPredicta/requirements.txt), [`.gitignore`](file:///c:/Users/Raphael/Documents/BioPredicta/.gitignore), [`README.md`](file:///c:/Users/Raphael/Documents/BioPredicta/README.md), [`DEVELOPMENT.md`](file:///c:/Users/Raphael/Documents/BioPredicta/DEVELOPMENT.md), [`DEVELOPMENT-API.md`](file:///c:/Users/Raphael/Documents/BioPredicta/DEVELOPMENT-API.md), [`context/PRD.md`](file:///c:/Users/Raphael/Documents/BioPredicta/context/PRD.md) e relatórios históricos em [`reports/`](file:///c:/Users/Raphael/Documents/BioPredicta/reports/).

### 2.2 Metodologia e Garantia de Preservação
Todas as verificações foram executadas sob regime estrito **somente leitura e não destrutivo**:
- Inspeção estática integral de código-fonte e varredura sintática via `ast` (Python `3.12.10` com flag `-B` para suprimir escrita de bytecode `.pyc`).
- Verificação criptográfica SHA-256 dos 8 artefatos protegidos antes e após a bateria de testes.
- Inspeção em memória dos objetos desserializados (`XGBRegressor` e `RandomForestClassifier`) confrontados contra `FEATURE_NAMES`, metadados JSON e schemas dos arquivos Parquet.
- Execução integral da suíte automatizada (`pytest tests -m "not benchmark" -q` e suíte dedicada da API) e sondagem funcional em processo via `TestClient` das rotas `/health`, `/predict`, `/docs`, `/redoc` e `/openapi.json`.
- **Nenhuma alteração** foi realizada em arquivos existentes do projeto, pacotes instalados, artefatos científicos ou histórico Git.

---

## 3. Estado do Git e Branches

A inspeção do controle de versão (`git status -u`, `git branch -a`, `git rev-parse HEAD`, `git log --oneline -n 20`) revelou o seguinte quadro exato:

| Item | Estado Verificado |
| :--- | :--- |
| **Branch atual** | `develop` (sincronizada com `remotes/origin/develop` no commit `96d1b24`) |
| **Branch principal remota (`origin/main`)** | Permanece no commit inicial `769d322` (*Initial commit*); todo o desenvolvimento consolidado reside em `develop` |
| **Commit de referência (`HEAD`)** | `96d1b245e91ade260acb484e7c0e24d9a97a43c7` — *Merge pull request #13 from raphaelpiresdev-ai/feat/fase-10-endpoints-fastapi* |
| **Fases já integradas à `develop`** | • Pipeline científico e Streamlit original: Fases 0 a 5 (PRs #1 a #6) + estabilização UX/UI multipágina (PRs #7 e #8)<br>• Cronograma FastAPI Fase 1 — Diagnóstico dos modelos (`feat/fase-6-preparacao-modelo-fastapi`, PR #9, `345c40f`)<br>• Cronograma FastAPI Fase 2 — Estrutura modular (`feat/fase-7-arquitetura-fastapi`, PR #10, `8dc530f`)<br>• Cronograma FastAPI Fase 3 — Lifespan (`feat/fase-8-lifespan`, PR #11, `c662a3a`)<br>• Cronograma FastAPI Fase 4 — Schemas Pydantic (`feat/fase-9-schemas-pydantic`, PR #12, `8d79921`)<br>• Cronograma FastAPI Fase 5 — Endpoints `/health` e `/predict` (`feat/fase-10-endpoints-fastapi`, PR #13, `96d1b24`) |
| **Alterações locais pendentes (não commitadas)** | **6 arquivos modificados (*unstaged*):**<br>• [`.gitignore`](file:///c:/Users/Raphael/Documents/BioPredicta/.gitignore)<br>• [`README.md`](file:///c:/Users/Raphael/Documents/BioPredicta/README.md)<br>• [`requirements.txt`](file:///c:/Users/Raphael/Documents/BioPredicta/requirements.txt)<br>• [`src/api/main.py`](file:///c:/Users/Raphael/Documents/BioPredicta/src/api/main.py)<br>• [`src/api/services/prediction_service.py`](file:///c:/Users/Raphael/Documents/BioPredicta/src/api/services/prediction_service.py)<br>• [`tests/test_api_endpoints.py`](file:///c:/Users/Raphael/Documents/BioPredicta/tests/test_api_endpoints.py)<br>**2 arquivos não rastreados (*untracked*):**<br>• [`DEVELOPMENT-API.md`](file:///c:/Users/Raphael/Documents/BioPredicta/DEVELOPMENT-API.md)<br>• [`src/api/exception_handlers.py`](file:///c:/Users/Raphael/Documents/BioPredicta/src/api/exception_handlers.py) |

**Diagnóstico de Integração:** As Fases 1 a 5 da API FastAPI estão devidamente mescladas na branch `develop`. Contudo, os **ajustes corretivos pós-Fase 5** (arredondamento de duas casas decimais na saída de `/predict`, criação do módulo `exception_handlers.py` com tradução de erros HTTP 422 para português brasileiro e atualizações de `README.md`, `requirements.txt`, `.gitignore` e `DEVELOPMENT-API.md`) permanecem **apenas na árvore de trabalho local** sobre `develop` (não commitados por restrição explícita das etapas anteriores). Todas essas alterações locais foram integralmente preservadas durante esta auditoria e encontram-se registradas no achado **`ACHADO-AF-01`**.

---

## 4. Resumo Executivo

O projeto **MultiTarget BioPredict** apresenta **elevada maturidade técnica, coerência quimioinformática e solidez arquitetural** para uma plataforma acadêmica e científica de triagem virtual multiobjetivo.

Os principais pilares verificados empiricamente demonstram que:
1. **Integridade Científica Irrestrita:** Os oito artefatos protegidos (modelos `.joblib`, metadados `.json`, datasets curados `.csv` e matrizes featurizadas `.parquet`) estão **100% íntegros** (correspondência exata de hashes SHA-256). O pipeline de inferência compartilha exatamente a mesma padronização molecular RDKit, ordem canônica de `1.030` variáveis (`FEATURE_NAMES`) e tipos numéricos (`float64`, `int64` e `uint8`) utilizados no treinamento.
2. **Desacoplamento Limpo entre Canais de Consumo:** A API REST FastAPI ([`src/api/`](file:///c:/Users/Raphael/Documents/BioPredicta/src/api/)) e a interface analítica Streamlit ([`app.py`](file:///c:/Users/Raphael/Documents/BioPredicta/app.py), [`src/ui/`](file:///c:/Users/Raphael/Documents/BioPredicta/src/ui/)) são mutuamente independentes — não há nenhuma importação cruzada entre `src/api/` e `streamlit`/`src/ui/` — e convergem de forma coesa sobre o núcleo científico compartilhado.
3. **Qualidade da API FastAPI e Contratos Pydantic v2:** O ciclo de vida via `@asynccontextmanager` carrega os dois modelos de forma atômica uma única vez no *startup* e limpa `app.state` no *shutdown*. Os contratos Pydantic aplicam validação estrita (`strict=True`, `extra="forbid"`, `allow_inf_nan=False`), o arredondamento a duas casas decimais ocorre exclusivamente na construção de `PredictionResponse` mantendo o tipo `float`, e os manipuladores de exceção traduzem falhas HTTP 422 e JSON malformado para português brasileiro sem vazar `input` bruto ou *tracebacks*.
4. **Aprovação Integral da Suíte Automatizada:** Todos os **218 testes funcionais e de regressão** (`pytest tests -m "not benchmark" -q`) e todos os **48 testes dedicados da API** foram aprovados (`0` falhas).

Não foram identificadas falhas **Críticas** nem **Altas**. Foram registrados **4 achados Moderados** (pendência de commit das alterações locais em `develop`, divergência residual de notação na fronteira exata $P_{\text{hERG}} = 0{,}70$ entre módulos de UI/documentação, dois links quebrados no `README.md` para `LICENSE` e `AUDITORIA.md` na raiz, e ausência de fixação direta das dependências transitivas `starlette` e `scipy` associada aos 9 avisos de depreciação do Starlette 1.7), além de **3 achados Baixos** e **3 Informativos**, fundamentando o parecer **APROVADO COM RESSALVAS**.

---

## 5. Avaliação Arquitetural

### 5.1 Organização Modular e Responsabilidades

| Camada / Módulo | Arquivos Principais | Responsabilidade Verificada | Isolamento / Acoplamento |
| :--- | :--- | :--- | :--- |
| **Núcleo Científico** | [`chem_utils.py`](file:///c:/Users/Raphael/Documents/BioPredicta/src/chem_utils.py), [`model_utils.py`](file:///c:/Users/Raphael/Documents/BioPredicta/src/model_utils.py), [`data_utils.py`](file:///c:/Users/Raphael/Documents/BioPredicta/src/data_utils.py), [`ad_utils.py`](file:///c:/Users/Raphael/Documents/BioPredicta/src/ad_utils.py), [`pubchem_utils.py`](file:///c:/Users/Raphael/Documents/BioPredicta/src/pubchem_utils.py) | Sanitização RDKit, dessalgação, canonicalização, descritores 2D, Morgan ECFP4 (`uint8`), persistência `joblib`, diagnóstico AD (Níveis 1–4) e consulta PubChem com cache e *timeout*. | **Excelente.** Isento de dependências de interface web ou framework HTTP. |
| **API REST FastAPI** | [`main.py`](file:///c:/Users/Raphael/Documents/BioPredicta/src/api/main.py), [`exception_handlers.py`](file:///c:/Users/Raphael/Documents/BioPredicta/src/api/exception_handlers.py), [`routers/`](file:///c:/Users/Raphael/Documents/BioPredicta/src/api/routers/), [`schemas/`](file:///c:/Users/Raphael/Documents/BioPredicta/src/api/schemas/), [`services/prediction_service.py`](file:///c:/Users/Raphael/Documents/BioPredicta/src/api/services/prediction_service.py) | Exposição HTTP síncrona de monitoramento (`GET /health`) e inferência (`POST /predict`), validação Pydantic v2, ciclo de vida `lifespan` e tradução de erros. | **Excelente.** Zero importações de `streamlit` ou `src.ui`. Divisão clara `Router -> Service -> Scientific Core`. |
| **Interface Web Streamlit** | [`app.py`](file:///c:/Users/Raphael/Documents/BioPredicta/app.py), [`src/ui/theme.py`](file:///c:/Users/Raphael/Documents/BioPredicta/src/ui/theme.py), [`components.py`](file:///c:/Users/Raphael/Documents/BioPredicta/src/ui/components.py), [`catalogs.py`](file:///c:/Users/Raphael/Documents/BioPredicta/src/ui/catalogs.py), [`interpretation_engine.py`](file:///c:/Users/Raphael/Documents/BioPredicta/src/ui/interpretation_engine.py), [`history_manager.py`](file:///c:/Users/Raphael/Documents/BioPredicta/src/ui/history_manager.py), [`pages/`](file:///c:/Users/Raphael/Documents/BioPredicta/src/ui/pages/) | Experiência visual interativa multipágina, editor molecular 2D Ketcher, matriz multiobjetivo, explicabilidade SHAP com subambientes Morgan e comparação de compostos. | **Bom.** Zero importações de `src.api`. Retém bloco legado duplicado em `app.py` (**`ACHADO-AF-05`**). |
| **Qualidade & Testes** | [`tests/`](file:///c:/Users/Raphael/Documents/BioPredicta/tests/) (17 suítes, 219 testes totais incluindo 1 benchmark) | Verificação unitária, contratual, de integração, regressão tabular em 14.500 moléculas, estabilização HTML/navegação e benchmark de latência/memória. | **Excelente.** Separação limpa via marcador `@pytest.mark.benchmark` em [`pytest.ini`](file:///c:/Users/Raphael/Documents/BioPredicta/pytest.ini). |

### 5.2 Critérios Arquiteturais Específicos
- **Ausência de dependências circulares:** Confirmada por inspeção estática da árvore `ast` e pelo teste automatizado `test_no_circular_dependencies` em [`tests/test_api_structure.py`](file:///c:/Users/Raphael/Documents/BioPredicta/tests/test_api_structure.py).
- **Resolução de caminhos:** Todos os módulos utilizam `Path(__file__).resolve().parents[...]` da biblioteca padrão `pathlib` (`DIRETORIO_RAIZ = Path(__file__).resolve().parents[2]` em [`src/api/main.py`](file:///c:/Users/Raphael/Documents/BioPredicta/src/api/main.py#L32); `_PROJECT_ROOT = Path(__file__).resolve().parents[1]` em [`src/ad_utils.py`](file:///c:/Users/Raphael/Documents/BioPredicta/src/ad_utils.py#L46)), garantindo independência em relação ao diretório corrente de execução (`cwd`).
- **Duplicação de código identificada:** O arquivo [`app.py`](file:///c:/Users/Raphael/Documents/BioPredicta/app.py) (2.341 linhas) preserva funções e catálogos legados (linhas 54–1750) acessíveis apenas por um expansor opcional `"Modo de Transição"` (linhas 2245–2249), duplicando funcionalidades já extraídas para `src/ui/` (**`ACHADO-AF-05`**, gravidade `BAIXO`).

---

## 6. Avaliação do Lifespan e Carregamento dos Modelos

A inspeção de [`src/api/main.py`](file:///c:/Users/Raphael/Documents/BioPredicta/src/api/main.py#L31-L86) confirma conformidade rigorosa com o padrão moderno de ciclo de vida do FastAPI/Starlette:

1. **Uso de `@asynccontextmanager`:** A função [`lifespan`](file:///c:/Users/Raphael/Documents/BioPredicta/src/api/main.py#L38-L85) é decorada com `@asynccontextmanager` e tipada com `AsyncIterator[None]`, sendo injetada diretamente em `FastAPI(..., lifespan=lifespan)` (linha 99), sem uso dos eventos depreciados `@app.on_event`.
2. **Ausência de desserialização na importação:** Importar `src.api.main` apenas define constantes `Path` e instancia o objeto `FastAPI`; nenhum acesso a disco ou chamada a `joblib.load` ocorre fora da execução do `lifespan`.
3. **Reutilização da função oficial e carga única:** Ambos os artefatos são carregados uma única vez no *startup* por meio de [`src.model_utils.load_model`](file:///c:/Users/Raphael/Documents/BioPredicta/src/model_utils.py#L118-L120) (linhas 65 e 68 de `src/api/main.py`), eliminando I/O repetido por requisição.
4. **Resolução de caminhos e variáveis de ambiente:** Suporta `BIOPREDICTA_MODELS_DIR`, `A2A_MODEL_PATH` e `HERG_MODEL_PATH` (linhas 33–35). Observou-se apenas que `os.getenv(...)` é avaliado no momento da importação do módulo e não dentro do corpo de `lifespan()` (**`ACHADO-AF-06`**, gravidade `BAIXO`).
5. **Atomicidade e falha rápida (*fail-fast*) no startup:**
   - Verifica explicitamente `CAMINHO_MODELO_A2A.exists()` e `CAMINHO_MODELO_HERG.exists()` antes da leitura, lançando `FileNotFoundError` imediato se qualquer arquivo faltar (linhas 53–61).
   - Desserializa ambos os modelos em variáveis locais (`a2a_model`, `herg_model`) dentro de bloco `try...except`, convertendo qualquer erro de leitura/corrupção em `RuntimeError(...) from exc` (linhas 63–72).
   - **Garantia contra estado parcial corrompido:** As atribuições `app.state.a2a_model = a2a_model` e `app.state.herg_model = herg_model` (linhas 75–76) ocorrem **somente após** o carregamento bem-sucedido de ambos os modelos. Se o segundo modelo (`hERG`) falhar, o *startup* é abortado e `app.state` não fica parcialmente populado.
6. **Limpeza determinística no shutdown:** O bloco `try: yield finally:` (linhas 79–85) redefine `app.state.a2a_model = None` e `app.state.herg_model = None` no encerramento do servidor.

---

## 7. Avaliação dos Schemas Pydantic

Foram auditados os quatro modelos Pydantic v2 definidos em [`src/api/schemas/prediction.py`](file:///c:/Users/Raphael/Documents/BioPredicta/src/api/schemas/prediction.py) e [`src/api/schemas/health.py`](file:///c:/Users/Raphael/Documents/BioPredicta/src/api/schemas/health.py):

| Schema | Configuração (`ConfigDict`) | Campos e Restrições Verificadas | Conformidade |
| :--- | :--- | :--- | :---: |
| [`PredictionRequest`](file:///c:/Users/Raphael/Documents/BioPredicta/src/api/schemas/prediction.py#L13-L40) | `extra="forbid"`, `str_strip_whitespace=True`, `strict=True` | `smiles: str` obrigatório (`...`), `min_length=1`, `max_length=1000`. Rejeita `int`/`float`/`bool`/`list`/`dict`, strings vazias, strings contendo apenas espaços (pós-*strip*) e chaves extras. | **100% Conforme** |
| [`PredictionResponse`](file:///c:/Users/Raphael/Documents/BioPredicta/src/api/schemas/prediction.py#L42-L90) | `extra="forbid"`, `strict=True`, `allow_inf_nan=False` | `smiles: str` (`1..1000`), `p_ic50: float` obrigatório e finito, `p_herg_positive: float` obrigatório, finito e restrito a `ge=0.0, le=1.0`. Rejeita `NaN`, `±Inf` e coerção implícita de strings. | **100% Conforme** |
| [`ModelsLoadedStatus`](file:///c:/Users/Raphael/Documents/BioPredicta/src/api/schemas/health.py#L12-L25) | `extra="forbid"`, `strict=True` | `a2a: bool` obrigatório, `herg: bool` obrigatório. | **100% Conforme** |
| [`HealthResponse`](file:///c:/Users/Raphael/Documents/BioPredicta/src/api/schemas/health.py#L27-L51) | `extra="forbid"`, `strict=True` | `status: str`, `service: str` (default `"MultiTarget BioPredict API"`), `models_loaded: ModelsLoadedStatus`, `ready_for_inference: bool`. | **100% Conforme** |

**Destaques Técnicos:**
- Confirmado empiricamente (`type(v).__name__ == 'float'`) que `p_ic50` e `p_herg_positive` permanecem estritamente numéricos (`float`) na serialização JSON.
- A documentação OpenAPI gerada em `/openapi.json` inclui descrições científicas detalhadas, exemplos realistas e a ressalva explícita em `p_herg_positive` (linhas 84–86 de `schemas/prediction.py`) de que o valor não representa probabilidade clínica direta de arritmia em humanos.
- A tradução das mensagens de validação em [`exception_handlers.py`](file:///c:/Users/Raphael/Documents/BioPredicta/src/api/exception_handlers.py) preserva integralmente os metadados técnicos `loc` e `type` do Pydantic (`missing`, `string_type`, `extra_forbidden`, `string_too_short`, `string_too_long`, `json_invalid`), permitindo rastreamento programático inequívoco pelos clientes HTTP.

---

## 8. Avaliação dos Endpoints FastAPI

A inspeção do esquema `/openapi.json` confirma que **apenas os dois endpoints oficiais** estão registrados na aplicação: `GET /health` e `POST /predict`.

### 8.1 Monitoramento (`GET /health` — [`src/api/routers/health.py`](file:///c:/Users/Raphael/Documents/BioPredicta/src/api/routers/health.py))
- Retorna sempre `HTTP 200 OK` com schema `HealthResponse`.
- Inspeciona `app.state.a2a_model` e `app.state.herg_model` de forma segura via `getattr(..., None) is not None` (linhas 24–25), sem executar nenhuma rotina de featurização ou inferência.
- Quando ambos os modelos estão presentes: `status="ok"`, `models_loaded={"a2a": true, "herg": true}`, `ready_for_inference=true`.
- Quando um ou ambos estão ausentes: `status="degraded"`, `ready_for_inference=false`, indicando exatamente qual modelo falta em `models_loaded`.
- Não expõe caminhos locais de sistema de arquivos, variáveis de ambiente ou detalhes internos de infraestrutura.

### 8.2 Inferência Multiobjetivo (`POST /predict` — [`src/api/routers/predict.py`](file:///c:/Users/Raphael/Documents/BioPredicta/src/api/routers/predict.py) e [`prediction_service.py`](file:///c:/Users/Raphael/Documents/BioPredicta/src/api/services/prediction_service.py))
- Recebe `payload: PredictionRequest` validado pelo Pydantic antes de entrar na camada de serviço.
- A função de rota [`predict_molecule`](file:///c:/Users/Raphael/Documents/BioPredicta/src/api/routers/predict.py#L30-L61) é declarada de forma **síncrona (`def`)**, decisão arquitetural correta no FastAPI/Starlette para cargas *CPU-bound* (RDKit C++, XGBoost e Random Forest), pois despacha a execução para o *threadpool* externo sem bloquear o *event loop* ASGI principal.
- Consome exclusivamente as instâncias residentes em `request.app.state.a2a_model` e `request.app.state.herg_model`.
- Identifica dinamicamente a coluna da classe positiva (`1`) em `herg_model.classes_` (linhas 87–95 de `prediction_service.py`), aplica *clipping* defensivo `max(0.0, min(1.0, ...))` contra imprecisões IEEE 754 e realiza o arredondamento `round(val, 2)` **somente** no retorno de `PredictionResponse` (linhas 106–110).

### 8.3 Verificação Empírica da Semântica HTTP e Sondagem Manual (Seção 16)

| Cenário Testado na Auditoria | Rota | HTTP Esperado | HTTP Obtido | Evidência da Resposta JSON Obtida |
| :--- | :--- | :---: | :---: | :--- |
| Health check operacional | `GET /health` | `200` | **`200`** | `{"status": "ok", "service": "MultiTarget BioPredict API", "models_loaded": {"a2a": true, "herg": true}, "ready_for_inference": true}` |
| Predição válida (Cafeína em Kekulé `CN1C=NC2=C1C(=O)N(C(=O)N2C)C`) | `POST /predict` | `200` | **`200`** | `{"smiles": "Cn1c(=O)c2c(ncn2C)n(C)c1=O", "p_ic50": 4.95, "p_herg_positive": 0.1}` (`str`, `float`, `float`) |
| SMILES quimicamente inválido (`SMILES_INVALIDO_XYZ`) | `POST /predict` | `400` | **`400`** | `{"detail": "O código SMILES fornecido é quimicamente inválido ou não pôde ser interpretado pelo RDKit."}` |
| JSON sintaticamente malformado (`{smiles: invalid}`) | `POST /predict` | `422` | **`422`** | `{"detail": [{"loc": ["body"], "msg": "JSON inválido. Verifique a sintaxe da requisição e utilize aspas duplas nos valores de texto.", "type": "json_invalid"}]}` |
| Payload estruturalmente inválido (campo extra `extra_key`) | `POST /predict` | `422` | **`422`** | `{"detail": [{"loc": ["body", "extra_key"], "msg": "O campo adicional 'extra_key' não é permitido.", "type": "extra_forbidden"}]}` |
| Falha inesperada na inferência | `POST /predict` | `500` | **`500`** | `{"detail": "Ocorreu um erro interno durante o processamento da predição."}` |
| Modelos ausentes em `app.state` | `POST /predict` | `500` | **`500`** | `{"detail": "Modelos preditivos indisponíveis no servidor."}` |
| Documentação interativa Swagger / ReDoc / OpenAPI | `GET /docs`, `/redoc`, `/openapi.json` | `200` | **`200`** | Renderização íntegra de Swagger UI, ReDoc e JSON Schema OpenAPI 3.1. |

---

## 9. Avaliação do Tratamento de Exceções e Segurança

A revisão de segurança de aplicações (AppSec) sobre [`src/api/exception_handlers.py`](file:///c:/Users/Raphael/Documents/BioPredicta/src/api/exception_handlers.py), [`src/api/main.py`](file:///c:/Users/Raphael/Documents/BioPredicta/src/api/main.py) e [`src/api/routers/predict.py`](file:///c:/Users/Raphael/Documents/BioPredicta/src/api/routers/predict.py) constatou:

1. **Interceptação e tradução completa de `RequestValidationError` e `JSONDecodeError`:** Registrados globalmente em `main.py` (linhas 107–108), cobrem todos os tipos de violação contratual do Pydantic v2 (`json_invalid`, `json_type`, `missing`, `string_type`, `int_type`/`float_type`/`number_type`, `extra_forbidden`, `string_too_short`, `string_too_long`, `greater_than_equal`, `less_than_equal`, `finite_number` e *fallback* seguro em português).
2. **Supressão de `input` bruto e caminhos internos nas respostas HTTP:** A função [`translate_validation_error`](file:///c:/Users/Raphael/Documents/BioPredicta/src/api/exception_handlers.py#L21-L134) retorna estritamente `{"loc": clean_loc, "msg": msg, "type": err_type}`, omitindo a chave `input` e o dicionário `ctx` interno do Pydantic. Nas falhas HTTP 500, o cliente recebe mensagens genéricas estáticas sem *stack trace*, nomes de classes internas ou caminhos de diretórios do servidor.
3. **Preservação de erros HTTP intencionais:** No roteador [`predict.py`](file:///c:/Users/Raphael/Documents/BioPredicta/src/api/routers/predict.py#L44-L61), as exceções específicas `InvalidSmilesError` (HTTP 400) e `ModelsNotLoadedError` (HTTP 500 específico) são capturadas **antes** do `except Exception as exc:` final, evitando que erros de negócio sejam mascarados.
4. **Mitigação de negação de serviço por entrada excessiva:** O limite estrito `max_length=1000` em `PredictionRequest.smiles` impede o envio de strings gigantescas antes que o parser C++ do RDKit seja invocado.
5. **Ausência de credenciais ou segredos hardcoded:** Nenhum token, chave de API, senha ou credencial foi encontrado no código-fonte; [`.gitignore`](file:///c:/Users/Raphael/Documents/BioPredicta/.gitignore) protege `.env`, `*.env` e `.streamlit/secrets.toml`.
6. **Observações pontuais registradas:**
   - Em [`exception_handlers.py`](file:///c:/Users/Raphael/Documents/BioPredicta/src/api/exception_handlers.py#L146), `logger.warning("Falha de validação da requisição: %s", exc.errors())` grava o dicionário bruto de erros (que inclui `input`) no log interno do servidor (**`ACHADO-AF-07`**, gravidade `BAIXO`).
   - Não há `CORSMiddleware` configurado em `src/api/main.py`, comportamento restritivo adequado por padrão para execução local e consumo M2M (**`ACHADO-AF-10`**, `INFORMATIVO`).

---

## 10. Avaliação Científica da Inferência

Esta etapa confrontou diretamente os objetos binários serializados em `models/*.joblib`, os metadados `models/*.json`, os datasets `data/processed/*.parquet` e o código de inferência em [`src/chem_utils.py`](file:///c:/Users/Raphael/Documents/BioPredicta/src/chem_utils.py) e [`src/api/services/prediction_service.py`](file:///c:/Users/Raphael/Documents/BioPredicta/src/api/services/prediction_service.py).

### 10.1 Paridade Estrita entre Treinamento e Inferência (1030-D)

| Verificação Científica | Evidência Empírica Obtida na Auditoria | Status |
| :--- | :--- | :---: |
| **Padronização estrutural RDKit** | [`standardize_mol`](file:///c:/Users/Raphael/Documents/BioPredicta/src/chem_utils.py#L58-L83) e [`canonicalize_smiles`](file:///c:/Users/Raphael/Documents/BioPredicta/src/chem_utils.py#L144-L166) aplicam `rdMolStandardize.Cleanup`, `LargestFragmentChooser(preferOrganic=True)` (remoção de sais/contraíons), `Uncharger()` e `Chem.MolToSmiles(..., canonical=True, isomericSmiles=True)`. | **Aprovado** |
| **Dimensionalidade e Ordem das Features** | `len(FEATURE_NAMES) == 1030` (`6` descritores 2D: `MolWt`, `MolLogP`, `TPSA`, `NumHDonors`, `NumHAcceptors`, `NumRotatableBonds` seguidos por `1.024` bits `fp_0..fp_1023` Morgan raio 2). Confirmado `m_a2a.n_features_in_ == 1030`, `m_herg.n_features_in_ == 1030`, `list(m_a2a.feature_names_in_) == list(FEATURE_NAMES)` e `list(m_herg.feature_names_in_) == list(FEATURE_NAMES)`. | **Aprovado** |
| **Compatibilidade de `dtypes`** | Comparação coluna a coluna entre `a2a_featurized.parquet` e `featurize_dataframe()` na inferência resultou em **zero divergências (`[]`)**: `MolWt`, `MolLogP`, `TPSA` em `float64`; `NumHDonors`, `NumHAcceptors`, `NumRotatableBonds` em `int64`; `fp_0..fp_1023` em `uint8`. | **Aprovado** |
| **Paridade em Lote (14.500 moléculas)** | Recomputação completa de `a2a_curated.csv` (`1.364` compostos) e `herg_curated.csv` (`13.136` compostos) reproduz exatamente os arquivos Parquet dentro de `1e-12` para descritores contínuos e igualdade binária estrita (`np.array_equal`) para todos os `1.024` bits. | **Aprovado** |
| **Unidade de $\text{pIC}_{50}$ ($\text{A}_{2\text{A}}$)** | Treinado sobre registros ChEMBL `CHEMBL251` com `standard_type == "IC50"`, `standard_relation == "="`, `standard_units == "nM"`, transformados por $\text{pIC}_{50} = 9{,}0 - \log_{10}(\text{IC}_{50\text{ [nM]}})$. | **Aprovado** |
| **Seleção da Classe Positiva hERG** | `m_herg.classes_` verificado em memória como `[0, 1]` (`0 = Non-blocker`, `1 = Blocker`). `prediction_service.py` localiza `classes_list.index(1)` -> índice `1` e extrai `predict_proba(X_sample)[0, 1]`. | **Aprovado** |
| **Arredondamento isolado na camada HTTP** | Para a Cafeína, os modelos preveem internamente $\text{pIC}_{50} = 4{,}9465$ e $P_{\text{hERG}} = 0{,}104508$; o arredondamento `round(..., 2)` ocorre exclusivamente ao instanciar `PredictionResponse` (`4.95` e `0.1`), sem truncar features ou cálculos internos. | **Aprovado** |

### 10.2 Verificação das Métricas Históricas Documentadas vs Metadados Serializados
*(Nota metodológica: os valores abaixo correspondem à verificação documental cruzada contra [`models/a2a_model_metadata.json`](file:///c:/Users/Raphael/Documents/BioPredicta/models/a2a_model_metadata.json) e [`models/herg_model_metadata.json`](file:///c:/Users/Raphael/Documents/BioPredicta/models/herg_model_metadata.json), não tendo sido realizado novo treinamento durante a auditoria.)*

- **Modelo $\text{A}_{2\text{A}}$ (`XGBRegressor`, `n_estimators=250`, `learning_rate=0.08`, `max_depth=6`, `subsample=0.8`, `colsample_bytree=0.8`, `tree_method='hist'`, `1.364` amostras):**
  - *Holdout 20% (273 amostras):* $R^2 = 0{,}7229$ (`0,723`), $\text{RMSE} = 0{,}5777$ (`0,578`), $\text{MAE} = 0{,}4224$ (`0,422`).
  - *CV 5-Fold:* $R^2 = 0{,}6451 \pm 0{,}0495$ (`0,645 ± 0,050`), $\text{RMSE} = 0{,}6823 \pm 0{,}0626$ (`0,682 ± 0,063`), $\text{MAE} = 0{,}4860 \pm 0{,}0326$ (`0,486 ± 0,033`).
  - **Correspondência:** 100% exata com os metadados JSON e hiperparâmetros reais de `a2a_model.joblib`.
- **Modelo $\text{hERG}$ (`RandomForestClassifier`, `n_estimators=150`, `min_samples_split=4`, `13.136` amostras de `herg_karim`):**
  - *Holdout 20% (2.628 amostras):* $\text{ROC-AUC} = 0{,}9299$ (`0,930`), $\text{PR-AUC} = 0{,}9361$ (`0,936`), $\text{F1} = 0{,}8532$ (`0,853`), $\text{Brier} = 0{,}1118$ (`0,112`), $\text{MCC} = 0{,}7086$ (`0,709`).
  - *CV 5-Fold:* $\text{ROC-AUC} = 0{,}9172 \pm 0{,}0046$ (`0,917 ± 0,005`), $\text{PR-AUC} = 0{,}9209 \pm 0{,}0047$ (`0,921 ± 0,005`), $\text{F1} = 0{,}8393 \pm 0{,}0050$ (`0,839 ± 0,005`).
  - **Correspondência:** 100% exata com os metadados JSON e hiperparâmetros reais de `herg_model.joblib`.

---

## 11. Avaliação da Aplicação Streamlit

A aplicação Streamlit ([`app.py`](file:///c:/Users/Raphael/Documents/BioPredicta/app.py) + [`src/ui/`](file:///c:/Users/Raphael/Documents/BioPredicta/src/ui/)) permanece **plenamente funcional e íntegra**, sem qualquer interferência decorrente da adição da API FastAPI:

1. **Navegação Multipágina e Design System:** Utiliza `st.navigation` com registro centralizado em `st.session_state["pages_registry"]` e resolução segura via [`get_page_target`](file:///c:/Users/Raphael/Documents/BioPredicta/src/ui/components.py#L43-L75). O tema escuro local ([`src/ui/theme.py`](file:///c:/Users/Raphael/Documents/BioPredicta/src/ui/theme.py)) elimina dependência de fontes externas e todos os cartões HTML passam por [`render_html`](file:///c:/Users/Raphael/Documents/BioPredicta/src/ui/components.py#L78-L95) para evitar blocos CommonMark indentados e escapar entradas do usuário (`html.escape`, prevenindo XSS).
2. **Páginas Analíticas Implementadas (5 páginas operacionais):**
   - [`p2_nova_analise.py`](file:///c:/Users/Raphael/Documents/BioPredicta/src/ui/pages/p2_nova_analise.py): Presets farmacológicos de 1-clique (Cafeína, Istradefilina, Terfenadina, Taurina, L-Teanina), entrada SMILES, editor 2D Ketcher (`streamlit-ketcher` com parser Molfile V2000), enriquecimento PubChem resiliente a falhas offline e salvamento automático no histórico FIFO (até 10 moléculas).
   - [`p3_resultados.py`](file:///c:/Users/Raphael/Documents/BioPredicta/src/ui/pages/p3_resultados.py): KPIs de potência $\text{A}_{2\text{A}}$ ($\text{pIC}_{50}$ e $\text{IC}_{50}$ molar), escore $P_{\text{hERG}}$ e índice complementar $1 - P_{\text{hERG}}$, diagnóstico completo do Domínio de Aplicabilidade (Níveis 1 a 4 com auto-exclusão seletiva em Níveis 1 e 2) e matriz cartesiana multiobjetivo em coordenadas estritas de dados (`fill_between`).
   - [`p4_interpretacao.py`](file:///c:/Users/Raphael/Documents/BioPredicta/src/ui/pages/p4_interpretacao.py) e [`interpretation_engine.py`](file:///c:/Users/Raphael/Documents/BioPredicta/src/ui/interpretation_engine.py): Explicabilidade SHAP local Top 10 (`TreeExplainer`), detecção transparente de colisões de hash Morgan (múltiplos subgrafos distintos mapeados no mesmo bit), renderização 2D dos subambientes ativos, tratamento explícito de bits ausentes (`valor = 0` sem geração enganosa de imagem) e linguagem cientificamente neutra para o classificador hERG.
   - [`p5_comparacao.py`](file:///c:/Users/Raphael/Documents/BioPredicta/src/ui/pages/p5_comparacao.py): Comparação simultânea de 2 a 5 moléculas do histórico, cálculo algébrico de deltas ($\Delta \text{pIC}_{50}$, razão de $\text{IC}_{50}$, $\Delta P_{\text{hERG}}$, $\Delta$ descritores 2D), matriz simétrica de Tanimoto ECFP4 e gráfico multiobjetivo comparativo.
   - [`p6_como_utilizar.py`](file:///c:/Users/Raphael/Documents/BioPredicta/src/ui/pages/p6_como_utilizar.py): Tutorial didático estruturado em 7 módulos com todas as ressalvas metodológicas.
3. **Limitação de Escopo Deliberada (4 páginas institucionais provisórias):** As páginas `Visão Geral` (configurada com `default=True`), `Fundamentos Científicos`, `Modelos e Governança` e `Sobre o Projeto` exibem o cartão provisório planejado (`render_provisional_page`) com botão direto de navegação para `Nova Análise` (**`ACHADO-AF-08`**, `INFORMATIVO`).
4. **Inconsistência Pontual de Fronteira em $P_{\text{hERG}} = 0{,}70$:** Enquanto a lógica ativa da UI multipágina (`p2_nova_analise.py:144`, `p5_comparacao.py:308`, `p6_como_utilizar.py:273`) implementa $0{,}30 \le P \le 0{,}70$ (Moderado) e $P > 0{,}70$ (Alto), há textos residuais com `P ≥ 0,70` em `p2_nova_analise.py:206` e sobreposição textual na tabela de `p3_resultados.py:452-454`, além da convenção anterior `[0.30, 0.70)` / `>= 0.70` retida nas funções legadas de `app.py` e nos metadados JSON (**`ACHADO-AF-02`**, `MODERADO`).

---

## 12. Resultado dos Testes Automatizados

### 12.1 Execuções Realizadas na Auditoria

1. **Suíte Global Funcional e de Regressão (excluindo benchmark):**
   - **Comando:** `.\.venv\Scripts\python -m pytest tests -m "not benchmark" -q`
   - **Resultado:** **`218 passed, 1 deselected, 9 warnings in 96.51s (0:01:36)`** (`exit code 0`).
2. **Suíte Dedicada da API FastAPI:**
   - **Comando:** `.\.venv\Scripts\python -m pytest tests/test_api_structure.py tests/test_api_lifespan.py tests/test_api_schemas.py tests/test_api_endpoints.py -v`
   - **Resultado:** **`48 passed, 9 warnings in 10.50s`** (`exit code 0`).

*(Nota: Não há ferramenta de medição de cobertura de linhas como `pytest-cov` instalada no ambiente; portanto, em conformidade com as diretrizes de auditoria, não se reporta percentual numérico de cobertura de linhas, avaliando-se a cobertura funcional efetiva por suíte).*

### 12.2 Inventário e Avaliação Qualitativa das 17 Suítes de Teste

| Arquivo de Teste | Nº de Testes | Escopo Funcional Verificado | Qualidade |
| :--- | :---: | :--- | :--- |
| [`tests/test_api_structure.py`](file:///c:/Users/Raphael/Documents/BioPredicta/tests/test_api_structure.py) | 4 | Instância `FastAPI`, importabilidade limpa de subpacotes, rotas oficiais `/health` e `/predict` e ausência de ciclos de importação. | Alta |
| [`tests/test_api_lifespan.py`](file:///c:/Users/Raphael/Documents/BioPredicta/tests/test_api_lifespan.py) | 9 | Ausência de carga precoce no `import`, carga única no *startup*, falha rápida (`FileNotFoundError`/`RuntimeError`) para artefatos ausentes ou corrompidos, limpeza no *shutdown*, isolamento do Streamlit e teste com `.joblib` reais. | Alta |
| [`tests/test_api_schemas.py`](file:///c:/Users/Raphael/Documents/BioPredicta/tests/test_api_schemas.py) | 18 | Contratos `PredictionRequest`/`PredictionResponse`: limites `1..1000`, *strip* de espaços, proibição de extras, tipagem estrita, fronteiras `[0.0, 1.0]`, rejeição de `NaN`/`±Inf` e geração de JSON Schema. | Alta |
| [`tests/test_api_endpoints.py`](file:///c:/Users/Raphael/Documents/BioPredicta/tests/test_api_endpoints.py) | 17 | Integração HTTP ponta a ponta com modelos reais e *mocks*: `/health` (`ok` e `degraded` em 3 combinações), `/predict` (`200` com arredondamento a 2 casas, `400` SMILES inválido, `422` JSON malformado e 7 violações Pydantic traduzidas para PT-BR sem vazar `input`, `500` sem *traceback*). | Alta |
| [`tests/test_featurization_regression.py`](file:///c:/Users/Raphael/Documents/BioPredicta/tests/test_featurization_regression.py) | 7* | Paridade Kekulé vs canônico, dessalgação (`HCl`/`NaCl`), resiliência em lote misto, ordem/tipos 1030-D, predições dos 5 controles e paridade exata contra os Parquets (`1.364` A2A + `13.136` hERG) com verificação SHA-256. | Muito Alta |
| [`tests/test_applicability_domain.py`](file:///c:/Users/Raphael/Documents/BioPredicta/tests/test_applicability_domain.py) | 10 | Hierarquia AD Níveis 1 a 4, auto-exclusão seletiva (`has_exclusion`), não-exclusão no Nível 3, não-invenção de $S_{\max}=0$ em falhas e isolamento A2A vs hERG. | Muito Alta |
| [`tests/test_ad_performance.py`](file:///c:/Users/Raphael/Documents/BioPredicta/tests/test_ad_performance.py) | 1 (`benchmark`) | Latência *cold/warm* (100 repetições) e alocação de memória via `tracemalloc` (desmarcado na corrida funcional via `-m "not benchmark"`). | Adequada |
| [`tests/test_controls_and_interpretation.py`](file:///c:/Users/Raphael/Documents/BioPredicta/tests/test_controls_and_interpretation.py) | 5 | Identidade química dos 5 controles, limiares de `app.py` e distinção *in-sample* vs *out-of-sample*. | Boa |
| [`tests/test_multiobjective_plot.py`](file:///c:/Users/Raphael/Documents/BioPredicta/tests/test_multiobjective_plot.py) | 3 | Geometria `ax.transData` (`fill_between`), linhas de referência e ausência de `axvspan` em `app.py` e no `notebook 07`. | Alta |
| [`tests/test_pubchem_identification.py`](file:///c:/Users/Raphael/Documents/BioPredicta/tests/test_pubchem_identification.py) | 6 | Rejeição offline de SMILES inválidos, parsing PUG REST, filtragem de códigos CAS/catálogo, HTTP 404, falha de DNS/timeout e cache em memória. | Alta |
| [`tests/test_functional_stabilization.py`](file:///c:/Users/Raphael/Documents/BioPredicta/tests/test_functional_stabilization.py) | 8 | Resolução global de `get_page_target`, auditoria CommonMark/XSS de emissões HTML em todas as páginas, fluxo Ketcher V2000, presets, histórico e sincronização da Comparação. | Muito Alta |
| `tests/test_ui_p2..p6*.py` (6 arquivos) | 131* | Testes unitários e parametrizados exaustivos das páginas `Nova Análise`, `Resultados`, `Interpretação` (aditividade SHAP e colisões Morgan), `Comparação` (Tanimoto e deltas), `history_manager` (FIFO 10 e *deep copy*) e `Como Utilizar`. | Muito Alta |

*\*Incluindo expansões de `@pytest.mark.parametrize`, totalizando exatamente `218` itens funcionais executados + `1` item `benchmark` = `219` itens coletados pelo Pytest.*

---

## 13. Análise dos Warnings

Os **9 warnings** emitidos na execução da suíte global e da suíte da API foram integralmente rastreados:

| Qtd. | Classe do Warning | Origem Exata | Mensagem e Causa Técnica | Impacto Funcional Atual |
| :---: | :--- | :--- | :--- | :--- |
| **1** | `StarletteDeprecationWarning` | `.venv\Lib\site-packages\fastapi\testclient.py:1` | `"Using httpx with starlette.testclient is deprecated; install httpx2 instead."` — Emitido internamente pelo próprio pacote `fastapi==0.162.0` ao reexportar `TestClient` sob `starlette==1.7.0` com `httpx==0.28.1`. | **Nulo** (restrito ao cliente de testes; nenhum impacto no servidor ASGI em execução). |
| **8** | `StarletteDeprecationWarning` | `.venv\Lib\site-packages\starlette\_exception_handler.py:59` (acionado nos 8 testes HTTP 422 de [`tests/test_api_endpoints.py`](file:///c:/Users/Raphael/Documents/BioPredicta/tests/test_api_endpoints.py)) | `"'HTTP_422_UNPROCESSABLE_ENTITY' is deprecated. Use 'HTTP_422_UNPROCESSABLE_CONTENT' instead."` — Acionado porque [`src/api/exception_handlers.py`](file:///c:/Users/Raphael/Documents/BioPredicta/src/api/exception_handlers.py#L149) (linhas 149 e 164) referencia `status.HTTP_422_UNPROCESSABLE_ENTITY`, constante renomeada pela RFC 9110 no Starlette `1.7.0` (ambas avaliam para o inteiro `422`). | **Nulo no presente** (retorna HTTP `422` normalmente), porém exige atualização preventiva para evitar quebra em versões futuras do Starlette (**`ACHADO-AF-04`**). |

---

## 14. Avaliação das Dependências e Reprodutibilidade

A auditoria cruzada entre [`requirements.txt`](file:///c:/Users/Raphael/Documents/BioPredicta/requirements.txt), `pip freeze`, `pip check` e todas as instruções `import` em `app.py`, `src/`, `tests/` e `notebooks/*.ipynb` demonstrou:

1. **Consistência do Ambiente Virtual:** `python -m pip check` retornou **`No broken requirements found`**, comprovando ausência de conflitos de versão no `.venv`.
2. **Fixação Estrita e Ausência de Duplicatas:** Todos os **18 pacotes declarados** em `requirements.txt` utilizam operador de igualdade estrita (`==`), sem entradas duplicadas:
   - *Quimioinformática e Dados:* `rdkit==2026.9.1`, `numpy==2.5.4`, `pandas==3.0.6`, `pyarrow==25.0.1`.
   - *Machine Learning e XAI:* `scikit-learn==1.9.1`, `xgboost==3.4.1`, `shap==0.53.0`, `joblib==1.6.0`.
   - *Web UI:* `streamlit==1.65.0`, `streamlit-ketcher==0.0.1`.
   - *API REST e ASGI:* `fastapi==0.162.0`, `pydantic==2.14.2`, `uvicorn==0.54.0`, `python-multipart==0.0.32`.
   - *Aquisição, Notebooks e QA:* `chembl_webresource_client==0.10.9`, `requests==2.34.2`, `ipykernel==7.4.0`, `matplotlib==3.11.2`, `pytest==9.1.1`, `psutil==7.2.2`, `httpx==0.28.1` *(total: 21 pacotes fixados)*.
   - O isolamento deliberado do `PyTDC==1.1.15` em [`requirements-tdc.txt`](file:///c:/Users/Raphael/Documents/BioPredicta/requirements-tdc.txt) está tecnicamente bem justificado (conflito de wheels `tiledbsoma` no Windows e travas `numpy<2` / `pandas<3`).
3. **Coerência entre Importações Reais e Declarações:**
   - Todas as 13 bibliotecas externas importadas em `app.py`, `src/` e `tests/` (`fastapi`, `joblib`, `matplotlib`, `numpy`, `pandas`, `pydantic`, `pytest`, `rdkit`, `requests`, `shap`, `sklearn`, `streamlit`, `streamlit_ketcher`) estão explicitamente declaradas e fixadas em `requirements.txt`.
   - **Ressalva de reprodutibilidade transitiva (`ACHADO-AF-04`):** Nos cadernos `notebooks/`, `scipy` (`scipy==1.18.1` instalada) é importada diretamente para testes estatísticos de EDA, mas não está fixada diretamente em `requirements.txt` (sendo instalada transitivamente via `scikit-learn`/`shap`). De modo análogo, `starlette` (`1.7.0`) não está fixada diretamente, ficando sujeita à faixa semântica aberta de `fastapi==0.162.0`.

---

## 15. Integridade dos Oito Artefatos Protegidos

Os hashes criptográficos SHA-256 foram calculados diretamente sobre os arquivos em disco via `Get-FileHash -Algorithm SHA256` e confrontados com os valores oficiais de referência:

| # | Artefato Protegido | SHA-256 de Referência | SHA-256 Calculado na Auditoria | Classificação |
| :---: | :--- | :--- | :--- | :---: |
| 1 | [`models/a2a_model.joblib`](file:///c:/Users/Raphael/Documents/BioPredicta/models/a2a_model.joblib) | `1907C7241EDF64FFAE3AA5C59A497037D427B1EAD6B83ABDF15ABBFE3B2A4787` | `1907C7241EDF64FFAE3AA5C59A497037D427B1EAD6B83ABDF15ABBFE3B2A4787` | **Íntegro** |
| 2 | [`models/a2a_model_metadata.json`](file:///c:/Users/Raphael/Documents/BioPredicta/models/a2a_model_metadata.json) | `DA3E05936977254E08C34DAB6B21009C3DA2C632647E062522662BD47A1D5F8F` | `DA3E05936977254E08C34DAB6B21009C3DA2C632647E062522662BD47A1D5F8F` | **Íntegro** |
| 3 | [`models/herg_model.joblib`](file:///c:/Users/Raphael/Documents/BioPredicta/models/herg_model.joblib) | `FCEDBECB89843870FE1B7DA9C501EB62B8E9E6A0A254A9ED8139C17D6AD154E6` | `FCEDBECB89843870FE1B7DA9C501EB62B8E9E6A0A254A9ED8139C17D6AD154E6` | **Íntegro** |
| 4 | [`models/herg_model_metadata.json`](file:///c:/Users/Raphael/Documents/BioPredicta/models/herg_model_metadata.json) | `781F4B07B68EEEFF8F7FE245AA7513073811BB78819FE16996902C4D3146115F` | `781F4B07B68EEEFF8F7FE245AA7513073811BB78819FE16996902C4D3146115F` | **Íntegro** |
| 5 | [`data/processed/a2a_curated.csv`](file:///c:/Users/Raphael/Documents/BioPredicta/data/processed/a2a_curated.csv) | `88354901075BC9058B6AE6B5AD8920A4A710F10B2EFD32ABD8AB02541B977A3E` | `88354901075BC9058B6AE6B5AD8920A4A710F10B2EFD32ABD8AB02541B977A3E` | **Íntegro** |
| 6 | [`data/processed/herg_curated.csv`](file:///c:/Users/Raphael/Documents/BioPredicta/data/processed/herg_curated.csv) | `DF7BE71479E64C05F735B17B35A6FD888CC6CA869615E9BDE50F489C715B3798` | `DF7BE71479E64C05F735B17B35A6FD888CC6CA869615E9BDE50F489C715B3798` | **Íntegro** |
| 7 | [`data/processed/a2a_featurized.parquet`](file:///c:/Users/Raphael/Documents/BioPredicta/data/processed/a2a_featurized.parquet) | `AC006ABB90B59AC74C6D3A14CAD25CEEE62C58CEE44FA263A8A69F90928B31B7` | `AC006ABB90B59AC74C6D3A14CAD25CEEE62C58CEE44FA263A8A69F90928B31B7` | **Íntegro** |
| 8 | [`data/processed/herg_featurized.parquet`](file:///c:/Users/Raphael/Documents/BioPredicta/data/processed/herg_featurized.parquet) | `612E115EBA6E47788ADB92A2EDD067E89D5AF9F5095DBA0DFCAD6FC817D7EB92` | `612E115EBA6E47788ADB92A2EDD067E89D5AF9F5095DBA0DFCAD6FC817D7EB92` | **Íntegro** |

**Resultado:** **8 de 8 artefatos 100% Íntegros.**

---

## 16. Avaliação da Documentação

Foram inspecionados [`README.md`](file:///c:/Users/Raphael/Documents/BioPredicta/README.md), [`DEVELOPMENT.md`](file:///c:/Users/Raphael/Documents/BioPredicta/DEVELOPMENT.md), [`DEVELOPMENT-API.md`](file:///c:/Users/Raphael/Documents/BioPredicta/DEVELOPMENT-API.md), [`context/PRD.md`](file:///c:/Users/Raphael/Documents/BioPredicta/context/PRD.md) e os 24 relatórios técnicos em [`reports/`](file:///c:/Users/Raphael/Documents/BioPredicta/reports/) (incluindo [`reports/AUDITORIA.md`](file:///c:/Users/Raphael/Documents/BioPredicta/reports/AUDITORIA.md) e [`reports/RELATORIO_DIAGNOSTICO_FASE_6.md`](file:///c:/Users/Raphael/Documents/BioPredicta/reports/RELATORIO_DIAGNOSTICO_FASE_6.md)).

### Avaliação da Jornada do Avaliador Externo (8 Critérios)
1. **Compreender o objetivo científico:** **Sim.** A motivação farmacológica ($\text{A}_{2\text{A}}$ vs cardiotoxicidade hERG/$K_v11.1$) está explicada com rigor exemplar em `README.md`, `DEVELOPMENT.md` e no tutorial interno `Como Utilizar`.
2. **Instalar as dependências:** **Sim.** Passo a passo claro para criação do `.venv` em Python 3.12 e execução de `pip install -r requirements.txt` (Windows PowerShell e Linux/macOS).
3. **Executar o Streamlit:** **Sim.** Comando `streamlit run app.py` e porta `http://localhost:8501` documentados.
4. **Executar a API:** **Sim.** Comando `uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --reload` documentado em `README.md` e `DEVELOPMENT-API.md`.
5. **Acessar o Swagger:** **Sim.** Endereços `/docs`, `/redoc` e `/openapi.json` claramente listados.
6. **Realizar uma predição:** **Sim.** Exemplo `curl` completo com payload JSON da Cafeína e resposta esperada (`{"smiles": "Cn1c(=O)c2c(ncn2C)n(C)c1=O", "p_ic50": 4.95, "p_herg_positive": 0.1}`) fidedigna ao comportamento real da API.
7. **Interpretar corretamente as saídas:** **Sim**, com a ressalva menor da fronteira pontual $P_{\text{hERG}} = 0{,}70$ (**`ACHADO-AF-02`**).
8. **Compreender as limitações científicas:** **Sim.** Invariância quiral 2D, regime *in silico* não clínico e extrapolação fora do Domínio de Aplicabilidade estão enfatizados em todos os documentos principais.

**Inconsistências Documentais Pontuais (`ACHADO-AF-03`):**
- [`README.md`](file:///c:/Users/Raphael/Documents/BioPredicta/README.md) aponta para `[MIT](LICENSE)` (linhas 16 e 283), porém o arquivo `LICENSE` não existe no repositório.
- [`README.md`](file:///c:/Users/Raphael/Documents/BioPredicta/README.md) aponta para [`AUDITORIA.md`](file:///c:/Users/Raphael/Documents/BioPredicta/README.md#L288) na raiz (linhas 265 e 288), mas o arquivo encontra-se em [`reports/AUDITORIA.md`](file:///c:/Users/Raphael/Documents/BioPredicta/reports/AUDITORIA.md).
- [`DEVELOPMENT.md`](file:///c:/Users/Raphael/Documents/BioPredicta/DEVELOPMENT.md#L9) ainda exibe no cabeçalho o metadado histórico `Branch Atual: feat/fase5-explicabilidade-e-streamlit`.

---

## 17. Lista de Achados Classificados por Gravidade

Resumo quantitativo por categoria de risco:
- **CRÍTICO:** `0`
- **ALTO:** `0`
- **MODERADO:** `4` (`ACHADO-AF-01` a `ACHADO-AF-04`)
- **BAIXO:** `3` (`ACHADO-AF-05` a `ACHADO-AF-07`)
- **INFORMATIVO:** `3` (`ACHADO-AF-08` a `ACHADO-AF-10`)

---

### `ACHADO-AF-01` — Alterações corretivas e documentais da API pendentes de consolidação (uncommitted) na árvore de trabalho da branch `develop`
- **Gravidade:** **MODERADO**
- **Arquivo ou componente envolvido:** Árvore de trabalho Git na branch `develop` ([`.gitignore`](file:///c:/Users/Raphael/Documents/BioPredicta/.gitignore), [`README.md`](file:///c:/Users/Raphael/Documents/BioPredicta/README.md), [`requirements.txt`](file:///c:/Users/Raphael/Documents/BioPredicta/requirements.txt), [`src/api/main.py`](file:///c:/Users/Raphael/Documents/BioPredicta/src/api/main.py), [`src/api/services/prediction_service.py`](file:///c:/Users/Raphael/Documents/BioPredicta/src/api/services/prediction_service.py), [`tests/test_api_endpoints.py`](file:///c:/Users/Raphael/Documents/BioPredicta/tests/test_api_endpoints.py), [`DEVELOPMENT-API.md`](file:///c:/Users/Raphael/Documents/BioPredicta/DEVELOPMENT-API.md) e [`src/api/exception_handlers.py`](file:///c:/Users/Raphael/Documents/BioPredicta/src/api/exception_handlers.py)).
- **Evidência concreta:** `git status -u` sobre `HEAD` `96d1b24` reporta 6 arquivos modificados (*unstaged*) e 2 arquivos não rastreados (*untracked*). Além disso, `git diff` no Windows sinaliza conversão `LF will be replaced by CRLF` nesses arquivos, e `origin/main` ainda aponta para o commit inicial `769d322`.
- **Consequência:** Qualquer avaliador externo que clonar o repositório remoto (`origin/develop` ou `origin/main`) neste instante não receberá os ajustes corretivos pós-Fase 5 (arredondamento de duas casas decimais, tradução de erros HTTP 422 para PT-BR em `exception_handlers.py` e documentação atualizada da API).
- **Recomendação de correção:** Após a avaliação deste relatório, versionar os 8 arquivos em commit/PR estruturado para `develop` e promover a release homologada para `main`.
- **Impede a aprovação final?** **Não** (estado local decorrente das restrições operacionais explícitas de não-commit nas etapas anteriores).

---

### `ACHADO-AF-02` — Divergência residual na convenção matemática do ponto exato de fronteira $P_{\text{hERG}} = 0{,}70$ entre módulos de UI multipágina, funções legadas e documentação
- **Gravidade:** **MODERADO**
- **Arquivo ou componente envolvido:** [`src/ui/pages/p2_nova_analise.py`](file:///c:/Users/Raphael/Documents/BioPredicta/src/ui/pages/p2_nova_analise.py#L144) (linhas 144, 191, 206), [`src/ui/pages/p3_resultados.py`](file:///c:/Users/Raphael/Documents/BioPredicta/src/ui/pages/p3_resultados.py#L452-L454) (linhas 399–400 vs 452–454), [`src/ui/pages/p5_comparacao.py`](file:///c:/Users/Raphael/Documents/BioPredicta/src/ui/pages/p5_comparacao.py#L308), [`src/ui/pages/p6_como_utilizar.py`](file:///c:/Users/Raphael/Documents/BioPredicta/src/ui/pages/p6_como_utilizar.py#L273-L274) vs [`app.py`](file:///c:/Users/Raphael/Documents/BioPredicta/app.py#L1241) (linhas 1241, 1409), [`models/herg_model_metadata.json`](file:///c:/Users/Raphael/Documents/BioPredicta/models/herg_model_metadata.json#L1076-L1079), [`src/model_utils.py`](file:///c:/Users/Raphael/Documents/BioPredicta/src/model_utils.py#L8-L10) e [`README.md`](file:///c:/Users/Raphael/Documents/BioPredicta/README.md#L89).
- **Evidência concreta:**
  1. As páginas ativas da UI multipágina (`p2_nova_analise.py:144`, `p5_comparacao.py:308`, `p6_como_utilizar.py:273`) e a suíte [`tests/test_ui_p2_nova_analise.py`](file:///c:/Users/Raphael/Documents/BioPredicta/tests/test_ui_p2_nova_analise.py#L119) classificam $0{,}30 \le P \le 0{,}70$ como Risco Moderado/Intermediário e $P > 0{,}70$ como Alto Risco.
  2. Entretanto, em `p2_nova_analise.py` linha 206, o ramo `elif x_cur >= 6.5 and y_cur < 0.30:` ($P > 0{,}70$) exibe a string residual `"P ≥ 0,70"`; e em `p3_resultados.py` linhas 452–454, a tabela expansível declara simultaneamente `"P_hERG ≥ 0,70"` (Alto) e `"0,30 ≤ P_hERG ≤ 0,70"` (Intermediário).
  3. As funções legadas em `app.py` (testadas por [`tests/test_controls_and_interpretation.py`](file:///c:/Users/Raphael/Documents/BioPredicta/tests/test_controls_and_interpretation.py#L298-L334)), `herg_model_metadata.json`, `src/model_utils.py` e `README.md:89` adotam $0{,}30 \le P < 0{,}70$ e $P \ge 0{,}70$. *(A API FastAPI `POST /predict` não aplica categorização por limiares, não sendo afetada).*
- **Consequência:** Ambiguidade documental e divergência de rótulo exclusivamente para o valor pontual exato $P_{\text{hERG}} = 0{,}7000$.
- **Recomendação de correção:** Padronizar uma única convenção canônica para o ponto $0{,}70$ em todos os textos de UI, `README.md` e funções auxiliares, eliminando a sobreposição textual na tabela de `p3_resultados.py:452-454`.
- **Impede a aprovação final?** **Não.**

---

### `ACHADO-AF-03` — Links quebrados para `LICENSE` e `AUDITORIA.md` na raiz em `README.md` e cabeçalho histórico desatualizado em `DEVELOPMENT.md`
- **Gravidade:** **MODERADO**
- **Arquivo ou componente envolvido:** [`README.md`](file:///c:/Users/Raphael/Documents/BioPredicta/README.md#L16) (linhas 16, 265, 283, 288) e [`DEVELOPMENT.md`](file:///c:/Users/Raphael/Documents/BioPredicta/DEVELOPMENT.md#L9) (linha 9).
- **Evidência concreta:** `README.md` contém links relativos para `LICENSE` (inexistente no diretório raiz) e `AUDITORIA.md` na raiz (removido da raiz no commit `87e5917`, residindo atualmente em [`reports/AUDITORIA.md`](file:///c:/Users/Raphael/Documents/BioPredicta/reports/AUDITORIA.md)). `DEVELOPMENT.md:9` indica `Branch Atual: feat/fase5-explicabilidade-e-streamlit`.
- **Consequência:** Links quebrados (HTTP 404 na visualização web do GitHub) ao clicar no selo/link de licença MIT ou na referência a `AUDITORIA.md`.
- **Recomendação de correção:** Adicionar o arquivo `LICENSE` (MIT) na raiz do projeto, atualizar as referências de auditoria no `README.md` para `reports/AUDITORIA.md` e `AUDITORIA_FINAL.md`, e atualizar o cabeçalho de `DEVELOPMENT.md`.
- **Impede a aprovação final?** **Não.**

---

### `ACHADO-AF-04` — Dependências transitivas `starlette` e `scipy` não fixadas diretamente em `requirements.txt` e emissão de 9 `StarletteDeprecationWarning`s
- **Gravidade:** **MODERADO**
- **Arquivo ou componente envolvido:** [`requirements.txt`](file:///c:/Users/Raphael/Documents/BioPredicta/requirements.txt), [`src/api/exception_handlers.py`](file:///c:/Users/Raphael/Documents/BioPredicta/src/api/exception_handlers.py#L149) (linhas 149 e 164) e `notebooks/03`–`04`.
- **Evidência concreta:** `scipy` é importada diretamente nos cadernos Jupyter (`notebooks/`), mas não consta em `requirements.txt` (presente apenas transitivamente como `scipy==1.18.1`). `starlette` também é resolvida transitivamente (`starlette==1.7.0`), gerando 8 `StarletteDeprecationWarning`s pelo uso de `status.HTTP_422_UNPROCESSABLE_ENTITY` em `exception_handlers.py` e 1 aviso interno em `fastapi/testclient.py:1` sobre `httpx` vs `httpx2`.
- **Consequência:** Risco moderado de deriva de versões transitivas em reinstalações futuras do zero e quebra futura quando o Starlette remover o alias depreciado `HTTP_422_UNPROCESSABLE_ENTITY`.
- **Recomendação de correção:** Fixar `starlette==1.7.0` e `scipy==1.18.1` em `requirements.txt` e atualizar `status.HTTP_422_UNPROCESSABLE_ENTITY` para `status.HTTP_422_UNPROCESSABLE_CONTENT` em `src/api/exception_handlers.py`.
- **Impede a aprovação final?** **Não.**

---

### `ACHADO-AF-05` — Retenção de código monolítico legado duplicado em `app.py` ("Modo de Transição")
- **Gravidade:** **BAIXO**
- **Arquivo ou componente envolvido:** [`app.py`](file:///c:/Users/Raphael/Documents/BioPredicta/app.py#L54-L1750) vs pacote [`src/ui/`](file:///c:/Users/Raphael/Documents/BioPredicta/src/ui/).
- **Evidência concreta:** Cerca de 1.700 linhas de catálogos farmacofóricos, estilos CSS, gráficos e pipeline de predição permanecem duplicadas em `app.py` para sustentar um `st.expander` secundário (`"Modo de Transição"`, linhas 2245–2249), paralelamente à implementação modular ativa em `src/ui/`.
- **Consequência:** Eleva o esforço de manutenção e favorece pequenas divergências evolutivas entre o código legado e as páginas modulares.
- **Recomendação de correção:** Em ciclo futuro de limpeza arquitetural, remover o modo legado de `app.py` (reduzindo-o a um orquestrador enxuto de `st.navigation`) e apontar os testes baseados em AST (`test_multiobjective_plot.py`) diretamente para `src/ui/components.py`.
- **Impede a aprovação final?** **Não.**

---

### `ACHADO-AF-06` — Resolução das variáveis de ambiente de caminho dos modelos no tempo de importação de `src/api/main.py`
- **Gravidade:** **BAIXO**
- **Arquivo ou componente envolvido:** [`src/api/main.py`](file:///c:/Users/Raphael/Documents/BioPredicta/src/api/main.py#L32-L35).
- **Evidência concreta:** `DIRETORIO_MODELOS`, `CAMINHO_MODELO_A2A` e `CAMINHO_MODELO_HERG` avaliam `os.getenv(...)` no escopo de módulo durante a importação, em vez de fazê-lo dinamicamente na entrada de `lifespan(app)`.
- **Consequência:** Alterações em `os.environ` efetuadas em tempo de execução *após* `import src.api.main` (por exemplo, em fixtures de teste sem *monkeypatch* das constantes) não alteram os caminhos utilizados pelo `lifespan`.
- **Recomendação de correção:** Mover a leitura de `os.getenv(...)` para uma função auxiliar chamada dentro de `lifespan(app)`.
- **Impede a aprovação final?** **Não.**

---

### `ACHADO-AF-07` — Exposição do payload bruto (`input`) nos logs internos do servidor em falhas HTTP 422
- **Gravidade:** **BAIXO**
- **Arquivo ou componente envolvido:** [`src/api/exception_handlers.py`](file:///c:/Users/Raphael/Documents/BioPredicta/src/api/exception_handlers.py#L146).
- **Evidência concreta:** Embora a resposta JSON enviada ao cliente omita corretamente o campo `input`, a linha 146 executa `logger.warning("Falha de validação da requisição: %s", exc.errors())`, registrando a estrutura completa do erro Pydantic (incluindo o valor integral de `input`) na saída de log do servidor.
- **Consequência:** Poluição de logs em caso de payloads rejeitados por excesso de tamanho (`string_too_long`) ou registro em texto claro de estruturas moleculares proprietárias malformadas nos logs de infraestrutura.
- **Recomendação de correção:** Registrar no log do servidor a lista já higienizada (`translated`) ou truncar a representação de `input` nos logs.
- **Impede a aprovação final?** **Não.**

---

### `ACHADO-AF-08` — Quatro páginas institucionais do Streamlit permanecem como placeholders planejados (incluindo a rota `default=True`)
- **Gravidade:** **INFORMATIVO**
- **Arquivo ou componente envolvido:** [`app.py`](file:///c:/Users/Raphael/Documents/BioPredicta/app.py#L2170-L2242) (`Visão Geral`, `Fundamentos Científicos`, `Modelos e Governança` e `Sobre o Projeto`).
- **Evidência concreta:** Essas 4 páginas institucionais invocam `render_provisional_page(...)`, exibindo aviso explícito de conteúdo previsto para etapa futura e botão de navegação imediata para `Nova Análise`. Todas as 5 páginas operacionais e científicas da plataforma estão completas.
- **Consequência:** Ao abrir `http://localhost:8501`, a página inicial padrão (`Visão Geral`, `default=True`) é um cartão provisório que requer 1 clique para ir à página `Nova Análise`.
- **Recomendação de correção:** Opcionalmente transferir `default=True` para `page_nova_analise` enquanto o texto institucional definitivo de `Visão Geral` não for redigido.
- **Impede a aprovação final?** **Não** (limitação de escopo deliberada e documentada).

---

### `ACHADO-AF-09` — Limitações científicas intrínsecas ao pipeline QSAR 2D (random split, invariância quiral ECFP4, refit 100% e escore hERG não calibrado clinicamente)
- **Gravidade:** **INFORMATIVO**
- **Arquivo ou componente envolvido:** Modelos [`a2a_model.joblib`](file:///c:/Users/Raphael/Documents/BioPredicta/models/a2a_model.joblib), [`herg_model.joblib`](file:///c:/Users/Raphael/Documents/BioPredicta/models/herg_model.joblib), [`src/chem_utils.py`](file:///c:/Users/Raphael/Documents/BioPredicta/src/chem_utils.py) e [`src/api/services/prediction_service.py`](file:///c:/Users/Raphael/Documents/BioPredicta/src/api/services/prediction_service.py).
- **Evidência concreta:** Uso de divisão aleatória 80/20 (em vez de *scaffold split*), fingerprints Morgan ECFP4 aquirais, retreinamento final (*refit*) em 100% das amostras curadas e ausência de diagnóstico de Domínio de Aplicabilidade no retorno enxuto da rota `POST /predict`.
- **Consequência:** Estimativas *in silico* adequadas exclusivamente para triagem virtual precoce e priorização acadêmica, sem validade clínica direta.
- **Recomendação de correção:** Manter as declarações de escopo vigentes; considerar em versões futuras (`v0.2+`) *scaffold split*, quiralidade explícita (`includeChirality=True`), calibração isotônica/Platt e inclusão opcional de métricas de Domínio de Aplicabilidade na API REST.
- **Impede a aprovação final?** **Não.**

---

### `ACHADO-AF-10` — Ausência deliberada de CORS, autenticação e rate limiting na API local
- **Gravidade:** **INFORMATIVO**
- **Arquivo ou componente envolvido:** [`src/api/main.py`](file:///c:/Users/Raphael/Documents/BioPredicta/src/api/main.py).
- **Evidência concreta:** A API foi projetada para operação local (`127.0.0.1:8000`) sem `CORSMiddleware`, autenticação ou limitação de taxa por IP.
- **Consequência:** Adequado e seguro por padrão para uso local e pipelines internos; exigiria configuração adicional caso exposto em ambiente web público multiusuário.
- **Recomendação de correção:** Configurar `CORSMiddleware` restritivo, autenticação e *rate limiting* em proxy reverso apenas se houver futura publicação externa da API.
- **Impede a aprovação final?** **Não.**

---

## 18. Limitações Científicas e Operacionais Conhecidas

Para total transparência acadêmica e científica, registram-se formalmente as limitações conhecidas do sistema:

1. **Particionamento Aleatório vs. *Scaffold Split*:** As métricas históricas de validação ($R^2 = 0{,}723$ para $\text{A}_{2\text{A}}$; $\text{ROC-AUC} = 0{,}930$ para $\text{hERG}$) foram obtidas sob *random split* 80/20 (estratificado para hERG) e validação cruzada 5-fold. Em química medicinal, séries análogas com mesmo esqueleto de Bemis-Murcko compartilhadas entre treino e teste tendem a elevar o desempenho aparente frente a quimiotipos inéditos (*out-of-scaffold*).
2. **Invariância Estereoquímica 2D:** Embora [`canonicalize_smiles`](file:///c:/Users/Raphael/Documents/BioPredicta/src/chem_utils.py#L144-L166) preserve marcações isoméricas no texto SMILES (`isomericSmiles=True`), o gerador `GetMorganGenerator(radius=2, fpSize=1024)` com configuração padrão aquiral e os 6 descritores físico-químicos topológicos produzem vetores 1030-D idênticos ($\Delta X = 0$) para pares enantioméricos. Na interface Streamlit, essa condição é adequadamente capturada e reportada pelo motor AD como **Nível 2 (Correspondência de Features)**.
3. **Modelos de Produção Retreinados em 100% dos Dados (*Refit* Integral):** Os binários finais `.joblib` foram ajustados sobre a totalidade das amostras curadas (`1.364` em $\text{A}_{2\text{A}}$ e `13.136` em $\text{hERG}$). Consequentemente, consultas de moléculas presentes nesses conjuntos (como Cafeína e Istradefilina em $\text{A}_{2\text{A}}$, ou Terfenadina em $\text{hERG}$) refletem comportamento *in-sample*, devendo ser interpretadas em conjunto com o diagnóstico de proveniência do Domínio de Aplicabilidade.
4. **Representatividade dos Datasets Primários e Extrapolação Química:** As predições dependem da cobertura estrutural e da heterogeneidade interlaboratorial das bases ChEMBL (`CHEMBL251`) e TDC (`herg_karim`). Compostos distantes desse espaço químico ($S_{\max} < 0{,}30$, como aminoácidos pequenos, peptídeos, organometálicos ou produtos naturais macrocíclicos) encontram-se em regime de extrapolação matemática.
5. **Distinção entre $P_{\text{hERG}}$ Computacional e Risco Clínico Cardíaco:** O campo `p_herg_positive` ($P_{\text{hERG}}$) expressa a proporção de votos das árvores do `RandomForestClassifier` (escore não submetido a calibração pós-hoc formal, embora apresente Brier Score de `0,112` no holdout) para atividade bloqueadora *in vitro* de $K_v11.1$. **Não constitui probabilidade clínica direta de prolongamento do intervalo QT ou *Torsades de Pointes* em humanos**, nem substitui ensaios eletrofisiológicos (*patch-clamp*) ou estudos regulatórios ICH S7B.
6. **Escopo Enxuto da Resposta REST (`POST /predict`):** Por decisão de contrato da API atual, `POST /predict` retorna o SMILES canonicalizado, `p_ic50` e `p_herg_positive`, enquanto diagnósticos avançados (Domínio de Aplicabilidade Níveis 1–4 e decomposição SHAP Top 10) permanecem exclusivos da interface analítica Streamlit.

---

## 19. Recomendações de Melhoria Priorizadas

As recomendações abaixo estão ordenadas por prioridade técnica para eventual etapa futura de consolidação (**nenhuma foi executada nesta auditoria**):

### Prioridade 1 — Consolidação Pré-Entrega Final (Baixo Esforço / Alto Valor de Governança)
1. **Versionar as alterações pendentes no Git (`ACHADO-AF-01`):** Realizar commit estruturado dos 6 arquivos modificados e 2 arquivos não rastreados na branch `develop` (ou branch dedicada de fechamento) e integrar `develop -> main`.
2. **Corrigir links documentais no `README.md` (`ACHADO-AF-03`):** Criar o arquivo `LICENSE` (MIT) na raiz, corrigir o link de `AUDITORIA.md` para `reports/AUDITORIA.md` / `AUDITORIA_FINAL.md` e atualizar o cabeçalho histórico de `DEVELOPMENT.md`.
3. **Unificar a notação de fronteira $P_{\text{hERG}} = 0{,}70$ (`ACHADO-AF-02`):** Eliminar a sobreposição textual em `src/ui/pages/p3_resultados.py` (linhas 452–454) e alinhar os rótulos de `p2_nova_analise.py:206`, `README.md:89` e `model_utils.py` a uma única convenção matemática.

### Prioridade 2 — Reprodutibilidade e Manutenção Preventiva (Baixo Esforço)
4. **Fixar `starlette` e `scipy` e eliminar avisos de depreciação (`ACHADO-AF-04`):** Adicionar `starlette==1.7.0` e `scipy==1.18.1` ao `requirements.txt` e substituir `status.HTTP_422_UNPROCESSABLE_ENTITY` por `status.HTTP_422_UNPROCESSABLE_CONTENT` em `src/api/exception_handlers.py`.
5. **Higienizar logs de validação e resolução dinâmica de caminhos (`ACHADO-AF-06`, `ACHADO-AF-07`):** Logar apenas `translated` (sem `input` bruto) em `exception_handlers.py:146` e resolver `os.getenv(...)` dentro de `lifespan()` em `src/api/main.py`.

### Prioridade 3 — Evoluções Futuras de Produto e Ciência (Fora do Escopo Atual)
6. **Remoção do bloco legado em `app.py` e página inicial direta (`ACHADO-AF-05`, `ACHADO-AF-08`):** Definir `Nova Análise` como `default=True` (ou concluir o conteúdo institucional das 4 páginas provisórias) e aposentar o código legado duplicado em `app.py`.
7. **Evolução Científica `v0.2+` (`ACHADO-AF-09`):** Incorporar benchmark com *scaffold split* de Bemis-Murcko, fingerprints quirais (`includeChirality=True`), calibração probabilística pós-hoc para hERG e endpoint opcional de Domínio de Aplicabilidade/predição em lote na API FastAPI.

---

## 20. Parecer Técnico Conclusivo

### Classificação Final: **APROVADO COM RESSALVAS**

### Justificativa Fundamentada em Evidências
A equipe multidisciplinar de auditoria classifica o projeto **MultiTarget BioPredict** como **APROVADO COM RESSALVAS** pelos seguintes fundamentos objetivos e reproduzíveis:

1. **Ausência Total de Falhas Críticas ou Altas:** Nenhum defeito que comprometa a execução, a segurança das respostas HTTP, a integridade dos dados ou a validade matemática e quimioinformática das predições foi encontrado.
2. **Consistência Científica e Integridade Criptográfica Comprovadas:** Os **8 artefatos protegidos** encontram-se **100% íntegros** via SHA-256. A cadeia `SMILES -> padronização RDKit -> 6 descritores 2D + 1.024 bits Morgan ECFP4 (uint8) -> XGBRegressor (A2A) + RandomForestClassifier (hERG)` apresenta paridade exata de dimensionalidade (`1.030`), ordem de colunas, tipos de dados e predições entre o treinamento histórico, a aplicação Streamlit e a API REST FastAPI.
3. **Arquitetura Desacoplada e API REST Robusta:** O módulo FastAPI (`src/api/`) cumpre integralmente os requisitos de engenharia de software: gerenciamento atômico de recursos via `lifespan`, validação rigorosa com Pydantic v2 (`strict=True`, `extra="forbid"`, `allow_inf_nan=False`), arredondamento restrito à camada de saída preservando `float`, semântica HTTP correta (`200`, `400`, `422`, `500`) e mensagens de validação traduzidas para português brasileiro sem vazamento de *inputs* ou *tracebacks*.
4. **Qualidade Verificada por Testes Automatizados:** **100% de aprovação** na execução real da suíte global (`218 passed, 0 failed`) e da suíte dedicada da API (`48 passed, 0 failed`).
5. **Natureza das Ressalvas:** As ressalvas identificadas são exclusivamente de gravidade **Moderada** ou **Baixa** — notadamente a necessidade de consolidar em commit/branch as alterações locais atualmente pendentes na árvore de trabalho de `develop` (`ACHADO-AF-01`), uniformizar textos residuais sobre o ponto exato de fronteira $P_{\text{hERG}} = 0{,}70$ na UI/documentação (`ACHADO-AF-02`), corrigir dois links relativos no `README.md` (`ACHADO-AF-03`) e fixar `starlette`/`scipy` atualizando a constante HTTP 422 depreciada (`ACHADO-AF-04`) — não impedindo a utilização funcional nem a apresentação acadêmica da plataforma.

