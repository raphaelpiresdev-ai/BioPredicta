# RELATÓRIO DE DIAGNÓSTICO TÉCNICO — FASE 6
## Preparação, Treinamento, Avaliação e Serialização dos Modelos de Machine Learning
**Projeto:** MultiTarget BioPredict  
**Branch:** `feat/fase-6-preparacao-modelo-fastapi`  
**Perfil:** Engenharia de Machine Learning Sênior / MLOps  
**Natureza do Documento:** Diagnóstico Técnico Formal

---

## 1. RESUMO EXECUTIVO

O projeto **MultiTarget BioPredict** possui dois modelos de Machine Learning consolidados, serializados e operacionais:
1. **$A_{2A}$ Regressor:** Algoritmo XGBoost (`tree_method='hist'`, 250 estimadores) para predição contínua da potência inibitória frente ao receptor purinérgico de adenosina $A_{2A}$ ($ADORA2A$), expresso em $pIC_{50}$.
2. **hERG Classificador:** Algoritmo Random Forest (150 estimadores, `min_samples_split=4`) para estimativa de probabilidade de bloqueio do canal de potássio cardíaco $K_v11.1$ ($KCNH2$), retornando o escore $P(\text{hERG}=1)$.

A presente auditoria avaliou a viabilidade técnica e a integridade dos pipelines para a subsequente disponibilização desses modelos por meio de uma API REST de alta performance desenvolvida com **FastAPI**. 

**Principais conclusões do diagnóstico:**
- **Pipeline de Featurização Consistente e Isento de Vazamento:** O cálculo dos 6 descritores 2D e 1.024 bits de Morgan ECFP4 (`src/chem_utils.py`) é determinístico e quimicamente independente por molécula. Há correspondência exata de $1.030$ atributos de entrada entre treino e inferência.
- **Artefatos Serializados Íntegros e Alinhados:** Os binários [`models/a2a_model.joblib`](file:///c:/Users/Raphael/Documents/BioPredicta/models/a2a_model.joblib) e [`models/herg_model.joblib`](file:///c:/Users/Raphael/Documents/BioPredicta/models/herg_model.joblib) carregam sem corrupção no ambiente Python 3.12 e seus parâmetros coincidem com os metadados JSON correspondentes.
- **Modelos de Produção Ajustados em 100% dos Dados:** Conforme praxe consolidada em quimioinformática para maximizar a cobertura do espaço químico, os modelos finais serializados foram retreinados sobre a totalidade das amostras curadas ($1.364$ compostos para $A_{2A}$ e $13.136$ para hERG), após a homologação das métricas na divisão 80/20.
- **Bloqueio Identificado para a Fase de API:** O arquivo [`requirements.txt`](file:///c:/Users/Raphael/Documents/BioPredicta/requirements.txt) atual **não contempla** `fastapi`, `pydantic` nem cliente HTTP de teste (`httpx`), demandando planejamento de dependências antes da implementação dos endpoints.

---

## 2. ESTRUTURA ATUAL DO PROJETO

A estrutura atual do repositório reflete uma arquitetura centrada em Ciência de Dados e no aplicativo demonstrativo em Streamlit:

```text
BioPredicta/
├── app.py                      # Ponto de entrada e controlador de navegação Streamlit
├── pytest.ini                  # Configuração de suítes de teste (unitários vs benchmark)
├── requirements.txt            # Dependências fixadas do ambiente local
├── requirements-tdc.txt        # Dependências opcionais para ambiente isolado PyTDC
├── data/
│   ├── raw/                    # Dados brutos (chembl_a2a_raw.csv, tdc_herg_karim.csv)
│   └── processed/              # Datasets curados e featurizados (.csv e .parquet)
├── models/
│   ├── a2a_model.joblib        # Binário XGBoost serializado (192 KB, compress=3)
│   ├── a2a_model_metadata.json # Metadados de auditoria e métricas do A2A
│   ├── herg_model.joblib       # Binário Random Forest serializado (9.66 MB, compress=3)
│   └── herg_model_metadata.json# Metadados de auditoria e métricas do hERG
├── notebooks/                  # Cadernos de curadoria (01, 02), EDA (03, 04),
│                               # modelagem (05, 06) e integração SHAP (07)
├── src/                        # Núcleo Python da aplicação
│   ├── chem_utils.py           # Padronização química RDKit e featurização 1030-D
│   ├── data_utils.py           # Ingestão e curadoria de ChEMBL e TDC
│   ├── model_utils.py          # Métricas supervisionadas e persistência joblib
│   ├── ad_utils.py             # Domínio de aplicabilidade (Tanimoto LOO / percentis)
│   ├── pubchem_utils.py        # Integração resiliente com NCBI PUG REST API
│   └── ui/                     # Componentes, catálogos, tema e páginas Streamlit
├── tests/                      # 170 testes automatizados cobrindo quimioinformática, AD e UI
└── reports/                    # Documentação formal e registros de auditoria do projeto
```

**Diagnóstico Arquitetural:** O diretório `src/` agrupa utilitários científicos na raiz (`chem_utils.py`, `model_utils.py`, `ad_utils.py`) e componentes visuais em subdiretórios (`src/ui/`). Não há separação formal de uma camada de API (`src/api/`) nem de um pacote isolado de regras de negócio (`src/core/`).

---

## 3. PREPARAÇÃO DOS DADOS $A_{2A}$

### 3.1. Rastreabilidade de Scripts e Notebooks
- **Notebooks:** [`notebooks/01_a2a_curation.ipynb`](file:///c:/Users/Raphael/Documents/BioPredicta/notebooks/01_a2a_curation.ipynb), [`notebooks/03_a2a_featurization_eda.ipynb`](file:///c:/Users/Raphael/Documents/BioPredicta/notebooks/03_a2a_featurization_eda.ipynb) e [`notebooks/05_a2a_modeling.ipynb`](file:///c:/Users/Raphael/Documents/BioPredicta/notebooks/05_a2a_modeling.ipynb).
- **Módulos Python:** [`src/data_utils.py`](file:///c:/Users/Raphael/Documents/BioPredicta/src/data_utils.py) (funções `fetch_chembl_a2a_dataset` e `curate_a2a_dataset`) e [`src/chem_utils.py`](file:///c:/Users/Raphael/Documents/BioPredicta/src/chem_utils.py) (`canonicalize_smiles`, `compute_2d_descriptors`, `compute_morgan_fingerprints`, `featurize_dataframe`).

### 3.2. Curadoria Química e Tratamento de Inconsistências
1. **Fonte:** ChEMBL REST API, Target `CHEMBL251` (Adenosine receptor $A_{2A}$ humano).
2. **Filtragem de Integridade:** Registros com `standard_relation == '='`, `standard_type == 'IC50'`, `standard_units == 'nM'`, `standard_value > 0` e presença obrigatória de SMILES estrutural.
3. **Padronização Química Estrita (RDKit):**
   - Função [`standardize_mol`](file:///c:/Users/Raphael/Documents/BioPredicta/src/chem_utils.py#L58-L83): aplicação de `rdMolStandardize.Cleanup` (sanitização, remoção de hidrogênios explícitos, normalização funcional e reionização).
   - Remoção de sais e contraíons via `rdMolStandardize.LargestFragmentChooser(preferOrganic=True)`.
   - Neutralização de cargas simples via `rdMolStandardize.Uncharger()`.
   - Preservação da estereoquímica via canonicalização isomérica em [`canonicalize_smiles`](file:///c:/Users/Raphael/Documents/BioPredicta/src/chem_utils.py#L144-L166).
4. **Transformação da Variável Alvo:** $pIC_{50} = 9.0 - \log_{10}(\text{standard\_value})$.
5. **Deduplicação Experimental e Filtragem de Ruído:**
   - Agrupamento por SMILES canônico unificado.
   - Definição do $pIC_{50}$ consensual pela **mediana** dos ensaios.
   - Cálculo do desvio-padrão amostral ($pIC_{50\_\text{std}}$). Descarte estrito de moléculas com alta discordância entre ensaios ($pIC_{50\_\text{std}} > 1.0$, ou seja, divergência superior a 1 ordem de grandeza).
6. **Volume Curado:** $1.364$ moléculas únicas com alvo contínuo de $3{,}08$ a $10{,}30$.

### 3.3. Estratégia de Divisão e Featurização
- **Divisão Treino/Teste:** Divisão aleatória holdout 80/20 via `train_test_split(test_size=0.20, random_state=42)` no notebook `05_a2a_modeling.ipynb` (1.091 treino / 273 teste).
- **Validação Cruzada:** 5-Fold cross-validation no conjunto de treino (`KFold(n_splits=5, shuffle=True, random_state=42)`).
- **Featurização 1.030-D:**
  - 6 descritores físico-químicos: `MolWt`, `MolLogP`, `TPSA`, `NumHDonors`, `NumHAcceptors`, `NumRotatableBonds`.
  - 1.024 bits de Morgan ECFP4 (raio 2, gerador `rdFingerprintGenerator.GetMorganGenerator`, vetor binário `uint8`).
- **Análise de Vazamento de Dados (Data Leakage):**
  - A geração dos descritores e dos fingerprints de Morgan é estritamente **amostra-independente** (depende exclusivamente da topologia atômica da molécula individual, sem utilizar parâmetros populacionais de média/desvio do dataset). Não houve vazamento na featurização prévia em Parquet.
  - Para algoritmos lineares/neurais testados no notebook (Ridge, MLP), o escalonamento `StandardScaler` foi devidamente encapsulado em `Pipeline` acoplado ao treino, impedindo vazamento.
  - *Observação metodológica:* A divisão foi puramente aleatória, não utilizando divisão por andaimes moleculares (*Bemis-Murcko Scaffold Split*).

---

## 4. PREPARAÇÃO DOS DADOS hERG

### 4.1. Rastreabilidade de Scripts e Notebooks
- **Notebooks:** [`notebooks/02_herg_curation.ipynb`](file:///c:/Users/Raphael/Documents/BioPredicta/notebooks/02_herg_curation.ipynb), [`notebooks/04_herg_featurization_eda.ipynb`](file:///c:/Users/Raphael/Documents/BioPredicta/notebooks/04_herg_featurization_eda.ipynb) e [`notebooks/06_herg_modeling.ipynb`](file:///c:/Users/Raphael/Documents/BioPredicta/notebooks/06_herg_modeling.ipynb).
- **Módulos Python:** [`src/data_utils.py`](file:///c:/Users/Raphael/Documents/BioPredicta/src/data_utils.py) (funções `download_tdc_dataset` e `curate_herg_dataset`) e [`src/chem_utils.py`](file:///c:/Users/Raphael/Documents/BioPredicta/src/chem_utils.py).

### 4.2. Curadoria Toxicológica e Resolução de Conflitos
1. **Fonte:** Therapeutics Data Commons (TDC / Harvard Dataverse), benchmark `herg_karim` (~13.4k compostos).
2. **Padronização Química:** Rigorosamente idêntica ao pipeline $A_{2A}$ (`standardize_mol`), garantindo compatibilidade estequiométrica e desassociação de sais.
3. **Auditoria de Conflitos e Duplicatas:**
   - Agrupamento por SMILES canônico idêntico.
   - Cálculo da média dos rótulos $Y \in \{0, 1\}$.
   - **Resolução de Conflito Biológico:** Eliminação sumária de compostos com dados antagônicos ($0 < \text{mean\_label} < 1$, isto é, reportados como ativo em um ensaio e inativo em outro).
4. **Volume Curado:** $13.136$ compostos consensuais.
5. **Balanço de Classes:** Alta paridade natural ($6.577$ bloqueadores cardiotóxicos / $6.559$ não-bloqueadores seguros, representando 50,07% / 49,93%).

### 4.3. Estratégia de Divisão e Featurização
- **Divisão Treino/Teste:** Divisão estratificada 80/20 via `train_test_split(test_size=0.20, stratify=y, random_state=42)` no notebook `06_herg_modeling.ipynb` (10.508 treino / 2.628 teste cego).
- **Validação Cruzada:** 5-Fold Stratified Cross-Validation (`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`).
- **Featurização:** Exatamente o mesmo vetor 1.030-D (6 descritores + 1.024 bits ECFP4).
- **Análise de Vazamento:** Inexistente por construção dos descritores RDKit.

---

## 5. AVALIAÇÃO DOS MODELOS E MÉTRICAS

### 5.1. Modelo $A_{2A}$ (Potência Inibitória — Regressão)
- **Algoritmo Campeão:** XGBoost Regressor (`XGBRegressor`, `tree_method='hist'`).
- **Hiperparâmetros:** `n_estimators=250`, `learning_rate=0.08`, `max_depth=6`, `subsample=0.8`, `colsample_bytree=0.8`, `random_state=42`, `n_jobs=2`.
- **Algoritmos Avaliados no Benchmark Comparativo:**
  1. *Dummy Regressor (Média):* $R^2 = -0{,}001$ | $\text{RMSE} = 1{,}098$ | $\text{MAE} = 0{,}878$
  2. *Ridge Regression ($\alpha=10.0$):* $R^2 = 0{,}695$ | $\text{RMSE} = 0{,}606$ | $\text{MAE} = 0{,}469$
  3. *Random Forest ($n=200$):* $R^2 = 0{,}704$ | $\text{RMSE} = 0{,}597$ | $\text{MAE} = 0{,}442$
  4. **XGBoost (Campeão):** $R^2 = 0{,}723$ | $\text{RMSE} = 0{,}578$ | $\text{MAE} = 0{,}422$
  5. *MLP Regressor ($128, 64$):* $R^2 = 0{,}590$ | $\text{RMSE} = 0{,}702$ | $\text{MAE} = 0{,}550$
- **Validação Cruzada 5-Fold (Treino 80%):**
  - $R^2 = 0{,}645 \pm 0{,}050$
  - $\text{RMSE} = 0{,}682 \pm 0{,}063$
  - $\text{MAE} = 0{,}486 \pm 0{,}033$
- **Critérios de Aceite:** Metas do PRD ($R^2 \ge 0{,}65$, $\text{RMSE} \le 0{,}80$, $\text{MAE} \le 0{,}60$) plenamente aprovadas.

### 5.2. Modelo hERG (Cardiotoxicidade — Classificação Binária)
- **Algoritmo Campeão:** Random Forest Classifier (`RandomForestClassifier`).
- **Hiperparâmetros:** `n_estimators=150`, `min_samples_split=4`, `random_state=42`, `n_jobs=2`.
- **Algoritmos Avaliados no Benchmark Comparativo:**
  1. *Regressão Logística ($C=1.0$):* $\text{ROC-AUC} = 0{,}852$ | $\text{PR-AUC} = 0{,}858$ | $\text{F1} = 0{,}769$ | $\text{Brier} = 0{,}158$
  2. **Random Forest (Campeão):** $\text{ROC-AUC} = 0{,}930$ | $\text{PR-AUC} = 0{,}936$ | $\text{F1} = 0{,}853$ | $\text{Brier} = 0{,}112$ | $\text{MCC} = 0{,}709$
  3. *XGBoost ($n=250$, hist):* $\text{ROC-AUC} = 0{,}911$ | $\text{PR-AUC} = 0{,}917$ | $\text{F1} = 0{,}834$ | $\text{Brier} = 0{,}121$
  4. *MLP Classifier ($128, 64$):* $\text{ROC-AUC} = 0{,}917$ (Apresentou instabilidade/MemoryError em 4 folds de CV com $n\_jobs=2$).
- **Validação Cruzada Estratificada 5-Fold (Treino 80%):**
  - $\text{ROC-AUC} = 0{,}917 \pm 0{,}005$
  - $\text{PR-AUC} = 0{,}921 \pm 0{,}005$
  - $\text{F1} = 0{,}839 \pm 0{,}005$
- **Critérios de Aceite:** Metas do PRD ($\text{ROC-AUC} \ge 0{,}80$, $\text{PR-AUC} \ge 0{,}80$, $\text{F1} \ge 0{,}75$, $\text{Brier} \le 0{,}15$) plenamente aprovadas.

---

## 6. INTEGRIDADE DA SERIALIZAÇÃO

### 6.1. Verificação Criptográfica e Binária dos Arquivos
Os hashes SHA-256 e tamanhos físicos foram inspecionados:

| Artefato | Tamanho em Disco | Hash SHA-256 Calculado | Integridade |
| :--- | :--- | :--- | :--- |
| `models/a2a_model.joblib` | 192.169 bytes (~188 KB) | `1907C7241EDF64FFAE3AA5C59A497037D427B1EAD6B83ABDF15ABBFE3B2A4787` | ✅ Íntegro |
| `models/a2a_model_metadata.json` | 16.708 bytes (~16 KB) | `DA3E05936977254E08C34DAB6B21009C3DA2C632647E062522662BD47A1D5F8F` | ✅ Íntegro |
| `models/herg_model.joblib` | 9.667.505 bytes (~9.22 MB)| `FCEDBECB89843870FE1B7DA9C501EB62B8E9E6A0A254A9ED8139C17D6AD154E6` | ✅ Íntegro |
| `models/herg_model_metadata.json`| 16.852 bytes (~16 KB) | `781F4B07B68EEEFF8F7FE245AA7513073811BB78819FE16996902C4D3146115F` | ✅ Íntegro |

### 6.2. Conteúdo dos Metadados JSON e Correspondência com os Modelos
- **Metadados JSON:** Registram descrição biológica do alvo, identificador ChEMBL/gene, tipo de tarefa, hiperparâmetros de treinamento, volume amostral, lista exata das 1.030 features, métricas de validação cruzada 5-fold, métricas de teste holdout e limiares operacionais.
- **Correspondência de Hiperparâmetros:** 100% de concordância atestada entre as chaves de `metadata.json` e os atributos internos extraídos via `.get_params()` dos objetos desserializados com `joblib.load()`.
- **Correspondência de Atributos:** Ambos os modelos exigem exatamente $1.030$ entradas. A inferência de teste executada com a molécula controle Cafeína comprovou execução idêntica à esperada:
  - $A_{2A}$ predição contínua: $pIC_{50} = 5{,}00$ (Faixa de Baixa Potência/Inativo no PRD).
  - hERG escore de probabilidade: $P(\text{hERG}=1) = 0{,}079$ (7,9%, Faixa de Baixo Risco no PRD).

---

## 7. ANÁLISE DO REQUIREMENTS.TXT

A análise minuciosa do arquivo [`requirements.txt`](file:///c:/Users/Raphael/Documents/BioPredicta/requirements.txt) evidenciou:

1. **Dependências Presentes e Suficientes para Inferência Científica:**
   - Quimioinformática: `rdkit==2026.9.1` (estritamente fixada).
   - Computação numérica e estruturas: `numpy==2.5.4`, `pandas==3.0.6`, `pyarrow==25.0.1`.
   - Motores de inferência e explicabilidade: `scikit-learn==1.9.1`, `xgboost==3.4.1`, `joblib==1.6.0`, `shap==0.53.0`.
   - Aplicação web existente: `streamlit==1.65.0`, `streamlit-ketcher==0.0.1`.
   - Testes e benchmarking: `pytest>=9.0.0`, `psutil>=5.9.0`.

2. **Dependências Ausentes para a Construção da API FastAPI:**
   - **`fastapi`:** Framework web assíncrono para construção dos endpoints.
   - **`pydantic`:** Validador e serializador de dados para schemas tipados de entrada e saída (Requests/Responses DTO).
   - **`uvicorn`:** Servidor ASGI para publicação e execução da API (embora presente na máquina como dependência indireta do Streamlit, não está declarado formalmente no manifesto).
   - **`httpx`:** Necessário para implementar testes de integração da API via `TestClient` do FastAPI/Starlette sem subir o servidor em porta física.

---

## 8. COMPONENTES CIENTÍFICOS QUE PODERÃO SER COMPARTILHADOS

Os seguintes componentes existentes no repositório são puramente algorítmicos e desacoplados de interface gráfica, estando prontos para compor o núcleo científico compartilhado (`core`):

1. **Padronização e Featurização Química ([`src/chem_utils.py`](file:///c:/Users/Raphael/Documents/BioPredicta/src/chem_utils.py)):**
   - `canonicalize_smiles()`: Canonicalização isomérica, neutralização e desassociação de contraíons.
   - `smiles_to_mol()`: Parsing seguro de representações SMILES e blocos MDL Molfile.
   - `compute_2d_descriptors()`: Extração dos 6 descritores físico-químicos fundamentais.
   - `compute_morgan_fingerprints()`: Geração dos 1.024 bits ECFP4.
   - `featurize_dataframe()`: Orquestração em lote dos $1.030$ atributos.
   - Tupla canônica `FEATURE_NAMES`.
2. **Carregamento e Inferência Unificada ([`src/model_utils.py`](file:///c:/Users/Raphael/Documents/BioPredicta/src/model_utils.py)):**
   - `load_model()`: Desserialização segura via `joblib`.
   - `predict()`: Roteamento unificado para predição contínua ($A_{2A}$) e probabilística (hERG).
3. **Domínio de Aplicabilidade e Proveniência ([`src/ad_utils.py`](file:///c:/Users/Raphael/Documents/BioPredicta/src/ad_utils.py)):**
   - `compute_ad_diagnostics()`: Cálculo da similaridade máxima de Tanimoto ($S_{\max}$) frente ao dataset de treino e enquadramento hierárquico nos Níveis 1 a 4.
   - `get_ad_reference()`: Estrutura imutável em cache contendo as moléculas de referência.
4. **Resolução Externa de Compostos ([`src/pubchem_utils.py`](file:///c:/Users/Raphael/Documents/BioPredicta/src/pubchem_utils.py)):**
   - `identify_compound_pubchem()`: Consulta via API PUG REST com cache e tratamento resiliente de 404/offline.
5. **Catálogo Bioquímico de Interpretação ([`src/ui/catalogs.py`](file:///c:/Users/Raphael/Documents/BioPredicta/src/ui/catalogs.py) e [`src/ui/interpretation_engine.py`](file:///c:/Users/Raphael/Documents/BioPredicta/src/ui/interpretation_engine.py)):**
   - Dicionários de papéis farmacofóricos e descrições mecanicistas para enriquecer os payloads JSON da API.

---

## 9. COMPONENTES QUE DEVERÃO PERMANECER EXCLUSIVOS DO STREAMLIT

Os seguintes componentes contêm acoplamento rígido ao framework Streamlit (`import streamlit as st`) ou à renderização em navegador web, devendo permanecer isolados na aplicação demonstrativa:

1. **Controlador Central e Páginas ([`app.py`](file:///c:/Users/Raphael/Documents/BioPredicta/app.py) e [`src/ui/pages/`](file:///c:/Users/Raphael/Documents/BioPredicta/src/ui/pages/)):**
   - Todas as rotas multipágina: `p2_nova_analise.py`, `p3_resultados.py`, `p4_interpretacao.py`, `p5_comparacao.py`, `p6_como_utilizar.py`.
   - Invocação de widgets visuais: `st.selectbox`, `st.button`, `st.text_input`, `st_ketcher`.
2. **Componentes Visuais e Renderização HTML ([`src/ui/components.py`](file:///c:/Users/Raphael/Documents/BioPredicta/src/ui/components.py)):**
   - `render_html()`, `render_page_header()`, `render_provisional_page()`.
   - `render_molecule_dark_base64()`: Geração de strings SVG/PNG base64 voltadas para estilização Dark Mode do navegador.
   - `build_multiobjective_figure()` e `build_shap_impact_figure()`: Geração de figuras Matplotlib para exibição em canvas `st.pyplot()`.
   - `get_page_target()`: Resolução de rotas internas no controller do Streamlit.
3. **Gerenciamento de Estado de Sessão ([`src/ui/history_manager.py`](file:///c:/Users/Raphael/Documents/BioPredicta/src/ui/history_manager.py)):**
   - Persistência baseada em `st.session_state` (histórico FIFO de até 10 moléculas ativas na aba do navegador do usuário).
4. **Estilos e Folhas de Estilo ([`src/ui/theme.py`](file:///c:/Users/Raphael/Documents/BioPredicta/src/ui/theme.py)):**
   - Regras CSS customizadas injetadas no DOM do Streamlit.

---

## 10. ORGANIZAÇÃO RECOMENDADA PARA A FUTURA API FASTAPI

Para assegurar isolamento de responsabilidades, reprodutibilidade e zero impacto na aplicação Streamlit existente, recomenda-se a seguinte estrutura arquitetural em fases futuras:

```text
BioPredicta/
├── app.py                      # Permanece o ponto de entrada do Streamlit
├── requirements.txt            # Atualizado com fastapi, pydantic, uvicorn e httpx
├── models/                     # Permanece como repositório único de artefatos
├── data/                       # Permanece como repositório único de dados curados
├── src/
│   ├── core/                   # NÚCLEO CIENTÍFICO COMPILADO E COMPARTILHADO
│   │   ├── __init__.py
│   │   ├── chemistry.py        # Migração/alias de chem_utils.py
│   │   ├── inference.py        # Migração/alias de model_utils.py (carregador e preditor)
│   │   ├── domain.py           # Migração/alias de ad_utils.py (Domínio de Aplicabilidade)
│   │   └── interpretation.py   # Regras mecanicistas e catálogos
│   ├── api/                    # CAMADA EXCLUSIVA DA API FASTAPI
│   │   ├── __init__.py
│   │   ├── main.py             # Instância FastAPI, ciclo de vida (lifespan) e rotas
│   │   ├── config.py           # Configurações de execução e caminhos
│   │   ├── schemas/            # Schemas Pydantic v2 (Input, Output, Health, Errors)
│   │   │   ├── __init__.py
│   │   │   ├── molecule.py     # SmilesRequest, MoleculeFeaturesResponse
│   │   │   ├── prediction.py   # A2APredictionResponse, HergPredictionResponse, MultitargetResponse
│   │   │   └── domain.py       # ApplicabilityDomainResponse
│   │   ├── services/           # Regras de orquestração de negócio da API
│   │   │   ├── __init__.py
│   │   │   └── prediction_service.py # Consome src/core/ para gerar as predições
│   │   └── routers/            # Endpoints REST segregados
│   │       ├── __init__.py
│   │       ├── health.py       # GET /health (status do sistema e modelos carregados)
│   │       ├── predict.py      # POST /predict/a2a, POST /predict/herg, POST /predict/multitarget
│   │       └── analysis.py     # POST /analyze/full (predição + AD + química básica)
│   └── ui/                     # Interface Streamlit (mantida intacta)
└── tests/
    ├── test_api_*.py           # Novos testes automatizados para endpoints FastAPI
    └── test_ui_*.py            # Testes existentes do Streamlit (preservados 100%)
```

> **Diretriz de Transição Segura:** Durante a refatoração conceitual futura, módulos em `src/` (como `chem_utils.py` e `model_utils.py`) deverão manter fachadas de importação (*shims* de compatibilidade) para que o Streamlit continue funcionando sem quebrar nenhuma linha de código pré-existente.

---

## 11. PROBLEMAS ENCONTRADOS, CLASSIFICADOS POR CRITICIDADE

### Criticidade Alta (Bloqueante para Construção da API)
1. **Ausência do Framework FastAPI e Pydantic no Ambiente:**
   - *Fato verificado:* `fastapi` e `pydantic` retornam `NOT INSTALLED` no ambiente `.venv` e não constam em `requirements.txt`.
   - *Impacto:* Impossível instanciar qualquer serviço ou definir schemas de validação para a API sem adicionar essas dependências.
2. **Ausência de Biblioteca de Testes HTTP (`httpx`):**
   - *Fato verificado:* `httpx` não está instalado. O `TestClient` nativo do FastAPI depende de `httpx` para testes de integração assíncronos e síncronos sem alocação de portas no SO.

### Criticidade Média (MLOps e Rastreabilidade Metodológica)
1. **Diferença entre Modelo Avaliado no Holdout (80%) e Modelo Serializado (100%):**
   - *Fato verificado:* Os cadernos `05_a2a_modeling.ipynb` (Célula 21) e `06_herg_modeling.ipynb` (Célula 19) executam `final_model.fit(X, y)` sobre 100% dos dados antes de `joblib.dump()`.
   - *Impacto:* Os metadados JSON reportam métricas obtidas na partição de teste cego de 20%. O modelo serializado é mais robusto (possui mais dados), mas suas métricas estritas de generalização out-of-sample não são idênticas às do JSON. Isso deve ser formalizado na documentação da API (`/health` ou `/models/info`) para evitar equívocos de auditoria.
2. **Metadados JSON Sem Registro de Versões de Bibliotecas:**
   - *Fato verificado:* Os arquivos JSON contêm parâmetros e métricas, mas não registram `scikit-learn==1.9.1`, `xgboost==3.4.1`, nem o checksum SHA-256 do arquivo `.joblib` acoplado.

### Criticidade Baixa (Metodologia e Quimioinformática)
1. **Divisão Aleatória vs. Scaffold Split:**
   - *Fato verificado:* Os dados foram divididos aleatoriamente via `train_test_split(random_state=42)`.
   - *Impacto:* Moléculas de uma mesma série conênere (estruturalmente análogas) podem ter sido distribuídas entre treino e teste, o que é comum em pipelines iniciais, mas requer menção na documentação metodológica acadêmica.

---

## 12. AÇÕES NECESSÁRIAS PARA ATENDER À FASE 6

Para preparar a transição para a disponibilização dos modelos em API FastAPI quando autorizado, as etapas lógicas necessárias são:

1. **Atualização Controlada das Dependências:**
   - Adicionar ao `requirements.txt`: `fastapi`, `pydantic>=2.0`, `uvicorn>=0.30` e `httpx`.
   - Manter as versões fixadas de `rdkit`, `scikit-learn`, `xgboost`, `joblib` e `numpy` para não invalidar a compatibilidade binária dos modelos serializados.
2. **Criação do Módulo de Configuração e Lifespan da API:**
   - Criar um mecanismo de carregamento antecipado (*warm-up*) dos modelos no evento de inicialização (`lifespan`) do FastAPI, evitando recarregar os arquivos `.joblib` (especialmente o de hERG com ~9.6 MB) a cada requisição HTTP.
3. **Definição dos Schemas de Dados (Pydantic v2):**
   - Especificar schemas rigorosos com validação sintática prévia de strings SMILES antes de repassar ao RDKit.
4. **Isolamento da Lógica de Inferência:**
   - Encapsular a cadeia `SMILES -> canonicalize_smiles -> compute_2d_descriptors + compute_morgan_fingerprints -> model.predict / predict_proba` em um serviço desacoplado de Streamlit.
5. **Criação de Testes de Integração da API:**
   - Implementar suíte de testes com `TestClient` para validar endpoints de inferência para as moléculas de controle farmacológico (Cafeína, Istradefilina, Terfenadina).

---

## 13. CONCLUSÃO

| Dimensão Auditada | Status | Justificativa Técnica |
| :--- | :---: | :--- |
| **Preparação dos Dados ($A_{2A}$ e hERG)** | **ATENDIDO** | Curadoria química estrita, desassociação de contraíons, neutralização, resolução de conflitos e featurização 1.030-D determinística e padronizada. |
| **Modelos Treinados e Métricas** | **ATENDIDO** | Algoritmos campeões (XGBoost e Random Forest) superaram expressivamente todos os critérios de aceite estabelecidos no PRD. |
| **Serialização e Integridade dos Artefatos** | **ATENDIDO** | Arquivos `.joblib` íntegros, hashes SHA-256 preservados, correspondência exata de hiperparâmetros e compatibilidade de inferência comprovada. |
| **Ambiente e Reprodutibilidade para FastAPI** | **PARCIALMENTE ATENDIDO** | O ambiente suporta com perfeição a execução dos modelos existentes, mas ainda não possui as dependências específicas de API (`fastapi`, `pydantic`, `httpx`). |
| **Status Geral da Fase 6** | **PARCIALMENTE ATENDIDO** | Os modelos e artefatos científicos estão 100% prontos e validados para consumo por API; o manifesto de dependências e a arquitetura de código necessitam da introdução da camada FastAPI nas próximas etapas. |

