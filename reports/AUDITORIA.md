# MultiTarget BioPredict — Relatório Consolidado de Auditoria Técnica e Verificação Dirigida (Fases 1, 2A, 3C-1, 3D-0 e 3D-1)

**Projeto:** MultiTarget BioPredict — Plataforma de Triagem Virtual Multiobjetivo ($\text{A}_{2\text{A}}$ vs. $\text{hERG}$)  
**Escopo:** Auditoria técnica integral de código, dados, modelos serializados, quimioinformática (RDKit), interpretabilidade (SHAP), domínio de aplicabilidade e interface (Streamlit).  
**Status:** Auditorias de leitura concluídas (Fases 1, 2A, 3D-0 e 3D-1) e correções controladas homologadas via `pytest` (Fases 3A, 3B, 3C e 3C-1).

---

## Sumário Executivo

Este documento consolida as etapas de auditoria técnica e científica realizadas sobre o repositório **MultiTarget BioPredict**:

1. **Parte I — Fase 1 (Auditoria Técnica Integral):** Inspeção estrutural ponta a ponta do pipeline de ingestão, curadoria, transformação $\text{IC}_{50} \to \text{pIC}_{50}$, featurização molecular (`1.030` variáveis), treinamento supervisionado, inferência probabilística ($P_{\text{hERG}}$), espaço de decisão multiobjetivo e explicabilidade SHAP, resultando na identificação de **8 achados principais (`ACHADO-01` a `ACHADO-08`)**.
2. **Parte II — Fase 2A (Verificação Dirigida e Reprodução Experimental):** Investigação quimioinformática e farmacológica aprofundada, diferenciando rigorosamente valores **`[Reproduzidos na Fase 2A]`** (obtidos pela execução controlada no ambiente `.venv` sobre os artefatos locais) de valores **`[Transcritos da Documentação/Metadados]`**.
3. **Parte III — Matriz Consolidada de Achados e Testes de Aceitação:** Tabela de rastreabilidade técnica relacionando evidência reproduzida, conclusão, recomendação, arquivos afetados, risco de regressão e critério de aceitação.
4. **Parte IV — Plano de Correção Controlada:** Separação entre correções imediatas de código/UX/documentação (**sem retreinamento de modelos**) e melhorias científicas que exigiriam nova curadoria ou retreinamento.
5. **Parte V — Fase 3D-0 e Fase 3C-1 (Auditoria de Leitura do Dataset hERG e Correção Canônica das Fronteiras hERG):** Reconstrução forense da origem de `models/herg_model.joblib`, resolução documentada da discrepância entre `13.136` e `652` moléculas, verificação dos atributos internos das árvores dos modelos `.joblib`, definição dos conjuntos de referência para o Domínio de Aplicabilidade e padronização canônica dos operadores de limiares hERG (`p < 0.30`, `0.30 <= p < 0.70`, `p >= 0.70`).
6. **Parte VI — Fase 3D-1 (Diagnóstico Exploratório do Domínio de Aplicabilidade):** Auditoria de leitura das estruturas e paridade CSV/Parquet, cálculo eficiente da distribuição de similaridade máxima de Morgan/Tanimoto ($S_{\max}$ Leave-One-Out via `DataStructs.BulkTanimotoSimilarity`) para $\text{A}_{2\text{A}}$ (`1.364` moléculas) e $\text{hERG}$ (`13.136` moléculas), avaliação dos 5 controles farmacológicos por endpoint e análise de anomalias estruturais (estereoisomeria em ECFP4 aquiral, tautômeros e *activity cliffs*).

> [!IMPORTANT]
> **Nota de Errata Resolvida na Fase 3D-0 (ver Parte V):** Na redação inicial da Parte I abaixo (linhas 31–32, 39, 65, 78–79, 87 e referências na Parte III), houve confusão textual entre o dataset `herg_karim` (efetivamente usado no projeto) e o dataset menor `herg` do TDC (~652 moléculas). Conforme demonstrado com provas de código, `.shape` físico no disco e inspeção binária das árvores em `models/herg_model.joblib` na **Parte V**, o único dataset hERG existente e utilizado no treinamento real do projeto é `data/processed/herg_featurized.parquet` com **13.136 moléculas únicas** (`6.570` ativos / `6.566` inativos) e `RandomForestClassifier(n_estimators=150, min_samples_split=4, random_state=42, n_jobs=2)`.

---

# Parte I — Fase 1: Auditoria Técnica Integral

## 1.1 Arquitetura Real Encontrada e Fluxo de Dados

O sistema **MultiTarget BioPredict** opera em duas trilhas preditivas paralelas que convergem em um painel de decisão multiobjetivo e auditoria local via SHAP (`TreeExplainer`):

```mermaid
flowchart TD
    subgraph Ingestão e Curadoria
        A1["ChEMBL API (CHEMBL251, IC50)"] -->|"data/raw/chembl_a2a_raw.csv (3.050 linhas)"| B1["curate_a2a_dataset()"]
        B1 -->|"Filtro '=', 'nM', canonicalize_smiles, mediana pIC50"| C1["data/processed/a2a_curated.csv (1.364 moléculas)"]
        A2["Harvard Dataverse / TDC (Karim et al. hERG)"] -->|"data/raw/tdc_herg_karim.csv (13.445 linhas)"| B2["curate_herg_dataset()"]
        B2 -->|"canonicalize_smiles, remoção de conflitos binários"| C2["data/processed/herg_curated.csv (13.136 moléculas)"]
    end

    subgraph Featurização RDKit 1030-D
        C1 --> D1["featurize_dataframe()"]
        C2 --> D2["featurize_dataframe()"]
        D1 -->|"1024 bits Morgan (r=2) + 6 Descritores 2D"| E1["data/processed/a2a_featurized.parquet (1.364 x 1.035 colunas)"]
        D2 -->|"1024 bits Morgan (r=2) + 6 Descritores 2D"| E2["data/processed/herg_featurized.parquet (13.136 x 1.034 colunas)"]
    end

    subgraph Modelos de Produção Serializados
        E1 -->|"Split 80/20 p/ métricas + Refit 100%"| M1["models/a2a_model.joblib (XGBRegressor)"]
        E2 -->|"Split 80/20 estratificado + Refit 100%"| M2["models/herg_model.joblib (RandomForestClassifier)"]
    end

    subgraph Inferência e Explicabilidade (app.py)
        S["SMILES de Entrada"] -->|"canonicalize_smiles() + featurize_dataframe()"| X["Vetor 1030-D (fp_0..fp_1023 + 6 descritores)"]
        X --> M1 --> P1["pIC50 Predito + SHAP Local Top 10"]
        X --> M2 --> P2["P_hERG (predict_proba[:, 1])"]
        P1 & P2 --> G["Gráfico Multiobjetivo (pIC50 vs 1 - P_hERG)"]
    end
```

### Inventário de Arquivos Críticos e Responsabilidades

| Arquivo | Responsabilidade Real no Repositório |
| :--- | :--- |
| [`src/__init__.py`](src/__init__.py) | Inicialização do pacote `src` e configuração de diretórios base (`PROJECT_ROOT`, `DATA_DIR`, `MODELS_DIR`). |
| [`src/data_utils.py`](src/data_utils.py) | Download ChEMBL (`fetch_chembl_a2a_dataset`), download TDC (`fetch_tdc_herg_dataset`), conversão $\text{IC}_{50} \to \text{pIC}_{50}$ (`convert_ic50_to_pic50`), deduplicação por mediana/voto majoritário (`deduplicate_molecules`) e pipelines de curadoria (`curate_a2a_dataset`, `curate_herg_dataset`). |
| [`src/chem_utils.py`](src/chem_utils.py) | Padronização e dessalgação (`standardize_mol`), canonização (`canonicalize_smiles`), geração de Morgan Fingerprint ECFP4 de 1024 bits (`compute_morgan_fp`), cálculo de 6 descritores físico-químicos (`compute_physicochemical_descriptors`) e featurização tabular (`featurize_dataframe`). |
| [`src/model_utils.py`](src/model_utils.py) | Separação de matrizes $X, y$ (`get_feature_Columns`, `prepare_xy`), avaliação de regressores e classificadores (`evaluate_regressor`, `evaluate_classifier`, `cross_validate_model`), serialização com metadados JSON (`save_model`) e extração SHAP (`compute_shap_values`, `get_top_shap_features`). |
| [`app.py`](app.py) | Aplicação web Streamlit em tema escuro: entrada de SMILES/controles, validação química, inferência dual ($\text{A}_{2\text{A}}$ + $\text{hERG}$), gráfico de espaço de decisão multiobjetivo, auditoria local SHAP Top 10 e cartões visuais de farmacóforos renderizados com `DrawMorganBit`. |
| [`models/a2a_model.joblib`](models/a2a_model.joblib) | Modelo de produção $\text{A}_{2\text{A}}$: `XGBRegressor` (`n_estimators=300`, `max_depth=6`, `learning_rate=0.05`, `subsample=0.8`, `colsample_bytree=0.8`, `random_state=42`), retreinado sobre 100% das `1.364` moléculas curadas. |
| [`models/herg_model.joblib`](models/herg_model.joblib) | Modelo de produção $\text{hERG}$: `RandomForestClassifier` (`n_estimators=150`, `min_samples_split=4`, `random_state=42`, `n_jobs=2`), retreinado sobre 100% das `13.136` moléculas curadas de `herg_curated.csv` (tamanho: 9,66 MB). |
| [`notebooks/01` a `07`](notebooks/) | Notebooks sequenciais de ingestão (`01`, `02`), curadoria e featurização (`03`, `04`), treinamento de modelos (`05`, `06`) e avaliação multiobjetivo/SHAP (`07`). |

---

## 1.2 Verificação do Pipeline de Dados, Featurização e Modelos

1. **Ingestão e Curadoria $\text{A}_{2\text{A}}$ (`src/data_utils.py`):**
   - `data/raw/chembl_a2a_raw.csv` contém `2.976` linhas brutas baixadas do alvo `CHEMBL251` (`Homo sapiens`, receptor de adenosina $\text{A}_{2\text{A}}$) com filtro `standard_type="IC50"`.
   - `filter_valid_bioactivity()` retém apenas `standard_relation == "="`, `standard_units == "nM"`, `standard_value > 0` e SMILES não nulos, reduzindo para `1.753` registros quantitativos.
   - `convert_ic50_to_pic50()` aplica $\text{pIC}_{50} = 9{,}0 - \log_{10}(\text{IC}_{50\text{ [nM]}})$ e filtra valores no intervalo fisiológico $[2{,}0;\ 12{,}0]$.
   - `deduplicate_molecules()` agrupa por `canonical_smiles`, descarta moléculas com desvio padrão entre replicatas $\text{std} > 1{,}0$ (quando $n > 1$) e consolida o valor final pela **mediana**, gerando `data/processed/a2a_curated.csv` com **`1.364` moléculas únicas**.
2. **Ingestão e Curadoria $\text{hERG}$ (`src/data_utils.py`):**
   - `data/raw/tdc_herg_karim.csv` contém `13.445` linhas da base de Karim et al. baixadas diretamente do Harvard Dataverse (`Y ∈ {0, 1}`, limiar de bloqueio em $10\ \mu\text{M}$).
   - Após canonização e eliminação de 16 registros conflitantes com atividade discordante, restam **`13.136` moléculas únicas** em `data/processed/herg_curated.csv` (`6.568` bloqueadores cardiotóxicos `Y=1` e `6.568` não-bloqueadores seguros `Y=0`, com balanceamento exato de 50/50).
3. **Ordem e Paridade das `1.030` Features (`src/chem_utils.py` e `src/model_utils.py`):**
   - Tanto no treinamento quanto na inferência (`app.py`), a função `get_feature_Columns()` monta exatamente a mesma lista ordenada de `1.030` colunas:
     - `fp_0` até `fp_1023`: Morgan Fingerprint (ECFP4, `radius=2`, `nBits=1024`).
     - `MolWt`, `MolLogP`, `TPSA`, `NumHDonors`, `NumHAcceptors`, `NumRotatableBonds`.
   - Os metadados em `models/a2a_model_metadata.json` e `models/herg_model_metadata.json` confirmam `n_features = 1030`.
4. **Métricas Registradas nos Metadados (`models/*_metadata.json`):**
   - **$\text{A}_{2\text{A}}$ (`XGBRegressor`, avaliação em hold-out 20% com $n=273$ e 5-fold CV):** $\text{RMSE}_{\text{test}} = 0{,}5567$, $\text{MAE}_{\text{test}} = 0{,}4145$, $R^2_{\text{test}} = 0{,}7713$, $r_{\text{Pearson}} = 0{,}8785$, $\text{RMSE}_{\text{CV}} = 0{,}5763 \pm 0{,}0420$ **`[Transcrito da Documentação/Metadados]`** e **`[Reproduzido na Fase 2A]`**.
   - **$\text{hERG}$ (`RandomForestClassifier`, avaliação em hold-out 20% estratificado com $n=2.628$ e 5-fold CV sobre as 13.136 amostras):** $\text{ROC-AUC}_{\text{test}} = 0{,}930$, $\text{PR-AUC}_{\text{test}} = 0{,}936$, $\text{F1}_{\text{test}} = 0{,}853$, $\text{Brier}_{\text{test}} = 0{,}112$, $\text{Balanced\_Acc} = 0{,}854$, $\text{ROC-AUC}_{\text{CV}} = 0{,}917 \pm 0{,}005$ **`[Auditado e Confirmado nos Metadados e Binários]`**.

---

# Parte II — Fase 2A: Verificação Dirigida dos Achados da Auditoria

## 2.1 Investigação 1 — Identidade e Endpoint da Istradefilina

### 1. Identidade Molecular, SMILES Canônico e Identificador ChEMBL
- **SMILES utilizado nos controles** ([`app.py`](app.py#L430-L433), [`05_model_a2a_training.ipynb`](notebooks/05_model_a2a_training.ipynb), [`06_model_herg_training.ipynb`](notebooks/06_model_herg_training.ipynb), [`07_multitarget_shap_evaluation.ipynb`](notebooks/07_multitarget_shap_evaluation.ipynb)):
  - Entrada bruta (Kekulé): `CCN1C(=O)C2=C(N=C(N2C)/C=C/C3=CC(=C(C=C3)OC)OC)N(C1=O)CC`
  - Saída de `canonicalize_smiles()` **`[Reproduzido na Fase 2A]`**: `CCn1c(=O)c2c(nc(/C=C/c3ccc(OC)c(OC)c3)n2C)n(CC)c1=O`
  - **InChIKey verificado via RDKit** **`[Reproduzido na Fase 2A]`**: `IQVRBWUUXZMOPW-PKNBQFBNSA-N`
  - **Fórmula e Massa Molecular** **`[Reproduzido na Fase 2A]`**: $\text{C}_{20}\text{H}_{24}\text{N}_4\text{O}_4$, $\text{MolWt} = 384{,}436\text{ Da}$.
- **Identificador ChEMBL real nos dados vs. código:**
  - Nos arquivos [`data/raw/chembl_a2a_raw.csv`](data/raw/chembl_a2a_raw.csv) e [`data/processed/a2a_curated.csv`](data/processed/a2a_curated.csv) (índice `1315`), esta estrutura exata (`InChIKey = IQVRBWUUXZMOPW-PKNBQFBNSA-N`) está registrada sob o identificador **`CHEMBL431770`** **`[Reproduzido na Fase 2A]`**.
  - Em [`app.py`](app.py#L431) e nos notebooks `05`, `06` e `07`, o composto foi rotulado equivocadamente como **`CHEMBL223405`** (ID que possui `0` registros em `chembl_a2a_raw.csv`).

### 2. Ausência de Confusão Estrutural no Código vs. Erro em `DEVELOPMENT.md`
- A estrutura `CCn1c(=O)c2c(nc(/C=C/c3ccc(OC)c(OC)c3)n2C)n(CC)c1=O` corresponde **autenticamente à istradefilina (KW-6002)**: possui o núcleo 1,3-dietil-7-metilxantina acoplado na posição C8 ao grupo *(E)*-2-(3,4-dimetoxifenil)vinila. Não houve confusão com a cafeína (`Cn1c(=O)c2c(ncn2C)n(C)c1=O`, `CHEMBL113`) no código Python **`[Reproduzido na Fase 2A]`**.
- **Divergência na documentação:** Na tabela de [`DEVELOPMENT.md`](DEVELOPMENT.md#L500), foi transcrito o SMILES truncado `Cc1nc2c(n1)n(c(=O)n(c2=O)C)C` para a istradefilina **`[Transcrito da Documentação/Metadados]`**. Ao submeter essa string ao RDKit (`Chem.MolFromSmiles`), o parser retorna `None` (`Can't kekulize mol. Unkekulized atoms: 1 2 3 4 5`) **`[Reproduzido na Fase 2A]`**.

### 3. Registros de Origem em `data/raw/chembl_a2a_raw.csv`
A busca por `CHEMBL431770` (ou por `InChIKey == IQVRBWUUXZMOPW-PKNBQFBNSA-N`) em [`data/raw/chembl_a2a_raw.csv`](data/raw/chembl_a2a_raw.csv) retorna exatamente **2 registros** **`[Reproduzido na Fase 2A]`**:

| Índice (`raw`) | `activity_id` | `molecule_chembl_id` | `assay_chembl_id` | `document_chembl_id` | Ano | `standard_type` | `standard_relation` | `standard_value` | `standard_units` | `pchembl_value` | `assay_type` | `assay_description` |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `1159` | `13300499` | `CHEMBL431770` | `CHEMBL2384072` | `CHEMBL2380258` | `2013` | `IC50` | `=` | `5250.0` | `nM` | `5.28` | `F` | *Antagonist activity at recombinant human adenosine A2A receptor expressed in CHO cells assessed as inhibition of NECA-induced cAMP production after 60 mins* |
| `1178` | `13300525` | `CHEMBL431770` | `CHEMBL2384072` | `CHEMBL2380258` | `2013` | `IC50` | `=` | `5248.07` | `nM` | `5.28` | `F` | *Antagonist activity at recombinant human adenosine A2A receptor expressed in CHO cells assessed as inhibition of NECA-induced cAMP production after 60 mins* |

### 4. Transformação de Unidades e Cálculo de $\text{pIC}_{50}$
Aplicando $\text{pIC}_{50} = 9{,}0 - \log_{10}(\text{IC}_{50\text{ [nM]}})$ **`[Reproduzido na Fase 2A]`**:
- Registro `1159`: $9{,}0 - \log_{10}(5250{,}0) = 5{,}2798407$
- Registro `1178`: $9{,}0 - \log_{10}(5248{,}07) = 5{,}2800004$
- Mediana agregada em `deduplicate_molecules()` (arredondada para 4 casas decimais): **`5.2799`** (`std = 0.0`, `num_records = 2`), exatamente o valor gravado na linha `1315` de [`data/processed/a2a_curated.csv`](data/processed/a2a_curated.csv).

### 5. Mistura de Tipos de Ensaio no Treinamento (`B`, `F`, `A`)
A ingestão filtrou `standard_type="IC50"`, mas **não separou `assay_type`**. Nos `1.753` registros quantitativos válidos que alimentaram a curadoria, os tipos de ensaio foram misturados **`[Reproduzido na Fase 2A]`**:
- **`B` (Binding / Radioligand Displacement):** `1.224` registros (`69,8%`) — $\text{pIC}_{50}$ médio $= 7{,}633 \pm 1{,}127$ (mediana $= 7{,}523$).
- **`F` (Functional / Cell-based cAMP, etc.):** `504` registros (`28,8%`) — $\text{pIC}_{50}$ médio $= 7{,}086 \pm 1{,}254$ (mediana $= 6{,}973$).
- **`A` (ADMET):** `25` registros (`1,4%`) — $\text{pIC}_{50}$ médio $= 6{,}736 \pm 0{,}744$ (mediana $= 6{,}478$).
- Nas `1.364` moléculas únicas de `a2a_curated.csv`: `931` provêm exclusivamente de ensaios `B`, `395` exclusivamente de ensaios `F`, `25` exclusivamente de ensaios `A`, e **`13` moléculas misturam registros `B` e `F` fundidos pela mediana** **`[Reproduzido na Fase 2A]`**.

### 6. Comparação Farmacológica: $\text{IC}_{50}$ Funcional vs. $K_i$ de Ligação
- **Ausência de $K_i$ no dataset local:** Existem `0` registros com `standard_type == "Ki"` em `chembl_a2a_raw.csv` porque a query restringiu `standard_type="IC50"` **`[Reproduzido na Fase 2A]`**.
- **Relação de Cheng-Prusoff:** Enquanto a constante de dissociação $K_i$ mede a afinidade termodinâmica direta pelo receptor $\text{hA}_{2\text{A}}$ (onde a istradefilina apresenta $\text{p}K_i \approx 7{,}5\text{–}8{,}1$ na literatura de radioligantes), o $\text{IC}_{50}$ funcional depende da concentração e potência do agonista desafiante:
  $$\text{IC}_{50\text{ (funcional)}} = K_b \cdot \left(1 + \frac{[\text{Agonista}]}{EC_{50}}\right)$$
  No ensaio `CHEMBL2384072`, células CHO superexpressando $\text{hA}_{2\text{A}}$ foram estimuladas por 60 minutos com **NECA** (agonista total de alta potência), deslocando o $\text{IC}_{50}$ aparente de inibição de cAMP para $5{,}25\ \mu\text{M}$ ($\text{pIC}_{50} = 5{,}28$).

### 7. Reprodução da Predição `5,51` no Pipeline Real
Ao carregar [`models/a2a_model.joblib`](models/a2a_model.joblib) e featurizar a istradefilina (`46` bits ativos no ECFP4 + `MolWt=384.436`, `MolLogP=2.1241`, `TPSA=80.28`, `NumHDonors=0`, `NumHAcceptors=5`, `NumRotatableBonds=6`) **`[Reproduzido na Fase 2A]`**:
- **Predição do modelo de produção (100% dos dados):** **`5.5053`** (`5,51` arredondado), idêntica para o SMILES bruto e canônico.
- **Predição do modelo 80% (`random_state=42`):** a istradefilina (`idx=1315`) caiu no **conjunto de treino de 80%**, onde o modelo prevê **`5.4467`** contra o rótulo real de treino **`5.2799`**.

---

## 2.2 Investigação 2 — Controles Farmacológicos

Todas as predições abaixo foram **`[Reproduzidas na Fase 2A]`** executando os artefatos atuais [`models/a2a_model.joblib`](models/a2a_model.joblib) e [`models/herg_model.joblib`](models/herg_model.joblib):

| Composto Controle | Identidade Verificada (`InChIKey` / ID ChEMBL) | SMILES Canônico Padronizado | Valor Experimental de Referência nos Dados Locais | Predição Atual ($\text{pIC}_{50\text{ A2A}}$ / $P_{\text{hERG}}$) | Status Amostral ($\text{A}_{2\text{A}}$ / $\text{hERG}$) | Similaridade $S_{\max}$ no Treino $\text{A}_{2\text{A}}$ (excl. auto) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Istradefilina** | `IQVRBWUUXZMOPW-PKNBQFBNSA-N` / `CHEMBL431770` *(rotulado `CHEMBL223405` no código)* | `CCn1c(=O)c2c(nc(/C=C/c3ccc(OC)c(OC)c3)n2C)n(CC)c1=O` | **$\text{A}_{2\text{A}}$:** $\text{pIC}_{50} = 5{,}2799$ (`CHEMBL2384072`, `assay_type='F'`, $n=2$: $5250\text{ nM}$ e $5248{,}07\text{ nM}$)<br>**$\text{hERG}$:** Ausente na base local | **$\text{pIC}_{50} = 5{,}5053$** (`5,51`)<br>**$P_{\text{hERG}} = 45{,}94\%$** (`0,4594`) | **$\text{A}_{2\text{A}}$:** **In-Sample** (Treino 80% e 100%, `idx=1315`)<br>**$\text{hERG}$:** Out-of-Dataset | $1{,}000$ (incl. auto)<br>**$0{,}840$** (excl. auto) |
| **Cafeína** | `RYYVLZVUVIJVGH-UHFFFAOYSA-N` / `CHEMBL113` | `Cn1c(=O)c2c(ncn2C)n(C)c1=O` | **$\text{A}_{2\text{A}}$:** $\text{pIC}_{50} = 4{,}8533$ (`assay_type='B'`, $n=2$: `activity_id=7604316` com $20472\text{ nM}$ e `activity_id=24973065` com $9600\text{ nM}$)<br>**$\text{hERG}$:** Ausente na base local | **$\text{pIC}_{50} = 4{,}9465$** (`4,95`)<br>**$P_{\text{hERG}} = 10{,}45\%$** (`0,1045`) | **$\text{A}_{2\text{A}}$:** **In-Sample** (Treino 80% e 100%, `idx=37`)<br>**$\text{hERG}$:** Out-of-Dataset | $1{,}000$ (incl. auto)<br>**$0{,}389$** (excl. auto) |
| **Terfenadina** | `GUGOEEXESWIERI-UHFFFAOYSA-N` / `CHEMBL22` *(rotulado `CHEMBL1678` no notebook 06)* | `CC(C)(C)c1ccc(C(O)CCCN2CCC(C(O)(c3ccccc3)c3ccccc3)CC2)cc1` | **$\text{A}_{2\text{A}}$:** Ausente na base local<br>**$\text{hERG}$:** Ausente em `herg_curated.csv` *(bloqueador clássico de referência na literatura, $\text{IC}_{50\text{ hERG}} \approx 50\text{–}300\text{ nM}$)* | **$\text{pIC}_{50} = 5{,}2413$** (`5,24`)<br>**$P_{\text{hERG}} = 80{,}48\%$** (`0,8048`) | **$\text{A}_{2\text{A}}$:** Out-of-Dataset<br>**$\text{hERG}$:** Out-of-Dataset | **$0{,}227$** |
| **Taurina** | `XOAAWQZATWQOTB-UHFFFAOYSA-N` / Aminoácido sulfônico | `NCCS(=O)(=O)O` | **$\text{A}_{2\text{A}}$:** Ausente na base local *(inativo fisiológico esperado)*<br>**$\text{hERG}$:** Ausente na base local | **$\text{pIC}_{50} = 5{,}4548$** (`5,45`)<br>**$P_{\text{hERG}} = 7{,}60\%$** (`0,0760`) | **$\text{A}_{2\text{A}}$:** Out-of-Dataset<br>**$\text{hERG}$:** Out-of-Dataset | **$0{,}117$** |
| **L-Teanina** | `DATAGRPVKZEWHA-UHFFFAOYSA-N` / Aminoácido análogo do glutamato | `CCNC(=O)CCC(N)C(=O)O` | **$\text{A}_{2\text{A}}$:** Ausente na base local *(inativo em receptores $\text{A}_{2\text{A}}$)*<br>**$\text{hERG}$:** Ausente na base local | **$\text{pIC}_{50} = 5{,}7962$** (`5,80`)<br>**$P_{\text{hERG}} = 11{,}37\%$** (`0,1137`) | **$\text{A}_{2\text{A}}$:** Out-of-Dataset<br>**$\text{hERG}$:** Out-of-Dataset | **$0{,}203$** |

### Diagnóstico de Interpretação dos Controles
1. **Istradefilina e Cafeína são amostras *in-sample* em $\text{A}_{2\text{A}}$:** Ambas pertencem ao conjunto de treinamento (`idx=1315` e `idx=37`). Não podem ser tratadas como validação externa independente para $\text{A}_{2\text{A}}$.
2. **Terfenadina é validação externa legítima para $\text{hERG}$:** Ausente em `herg_curated.csv`, sua predição ($P_{\text{hERG}} = 80{,}48\%$) confirma sensibilidade *out-of-sample* do classificador de cardiotoxicidade.
3. **Causa da inflação de $\text{pIC}_{50}$ para Taurina (`5,45`) e L-Teanina (`5,80`):**
   - O dataset `a2a_curated.csv` possui média $\text{pIC}_{50} = 7{,}295$ (`base_value` do XGBoost) e apenas `1,9%` (`26/1.364`) das moléculas possuem $\text{pIC}_{50} < 5{,}0$.
   - O menor peso molecular em todo o treino $\text{A}_{2\text{A}}$ é $\text{MolWt}_{\min} = 180{,}17\text{ Da}$. Taurina ($\text{MolWt} = 125{,}15\text{ Da}$, $S_{\max} = 0{,}117$) e L-Teanina ($\text{MolWt} = 174{,}20\text{ Da}$, $S_{\max} = 0{,}203$) estão **fora do envelope físico-químico e estrutural de treinamento**, sofrendo regressão à média do ensemble.

---

## 2.3 Investigação 3 — Gráfico Multiobjetivo

### 1. Diagnóstico Geométrico (`app.py` e `notebooks/07`)
Em [`app.py`](app.py#L868-L885) e [`notebooks/07_multitarget_shap_evaluation.ipynb`](notebooks/07_multitarget_shap_evaluation.ipynb#L202-L216), o gráfico utiliza:
```python
ax.axvspan(6.5, 11.5, ymin=0.70, ymax=1.0, color="#2EA043", alpha=0.14, ...)
ax.axvspan(3.0, 6.5, ymin=0.0, ymax=0.30, color="#F85149", alpha=0.10, ...)
ax.set_xlim(3.5, 10.5)
ax.set_ylim(-0.05, 1.05)
```
- **Mistura de sistemas de coordenadas:** `ax.axvspan` interpreta `xmin, xmax` em coordenadas de dados ($X$), mas `ymin, ymax` em **coordenadas relativas do eixo $Y$ (`[0, 1]`)**.
- Com `ylim = (-0.05, 1.05)` ($\Delta Y = 1{,}10$), o sombreamento real em coordenadas de dados ($y_{\text{dados}} = -0{,}05 + y_{\text{rel}} \times 1{,}10$) fica deslocado **`[Reproduzido na Fase 2A]`**:
  - Zona Ideal (`ymin=0.70`): inicia em **$y = 0{,}720$** (descolada $+0{,}020$ acima da linha tracejada `axhline(0.70)`).
  - Zona Crítica (`ymax=0.30`): termina em **$y = 0{,}280$** (descolada $-0{,}020$ abaixo do limiar real de $0{,}30$).
- **Representação incompleta das 9 regiões:** Os cortes da regra de negócio ($x \in \{5{,}0;\ 6{,}5\}$ e $y \in \{0{,}30;\ 0{,}70\}$) definem $3 \times 3 = 9$ regiões, mas o gráfico omite as linhas $x = 5{,}0$ e $y = 0{,}30$ e estende a "Zona Crítica (Baixa Potência + Cardiotóxico)" até $x = 6{,}5$, englobando a faixa de Potência Moderada ($5{,}0 \le \text{pIC}_{50} < 6{,}5$).

### 2. Correção Geométrica Recomendada
Substituir `ax.axvspan` por `ax.fill_between([xmin, xmax], y1, y2)` (que opera estritamente em coordenadas de dados $X$ e $Y$) e desenhar os 4 limiares ($x = 5{,}0; 6{,}5$ e $y = 0{,}30; 0{,}70$).

---

## 2.4 Investigação 4 — Robustez da Inferência Molecular

### 1. Análise de `featurize_dataframe()` (`src/chem_utils.py`)
- Na linha `211` de [`src/chem_utils.py`](src/chem_utils.py#L211), `featurize_dataframe()` executa `mol = Chem.MolFromSmiles(str(s))` **sem chamar `standardize_mol(mol)` nem `canonicalize_smiles(s)`**.
- **Consequências para entradas não-canônicas ou sais em lote:**
  1. Contra-íons (`.Cl`, `.[Na+]`) não são removidos se `featurize_dataframe()` for chamada diretamente sobre SMILES brutos, alterando `MolWt`, `MolLogP`, `TPSA` e bits de fingerprint.
  2. A ordem atômica de SMILES Kekulé vs. canônico gera ruído de ponto flutuante na 16ª casa decimal de `MolLogP` (`2.1241000000000003` vs. `2.1241` na istradefilina) **`[Reproduzido na Fase 2A]`**.
- **Impacto de regressão ao corrigir `featurize_dataframe()`:**
  - Nos arquivos `a2a_featurized.parquet` e `herg_featurized.parquet`: **`0,000000`** (todos os SMILES de treino já haviam passado por `canonicalize_smiles()` na etapa de curadoria).
  - Nas predições dos 5 controles (`pIC50` e `P_hERG`): **`0,0000`** **`[Reproduzido na Fase 2A]`**.

---

## 2.5 Investigação 5 — Domínio de Aplicabilidade (AD)

### Calibração Empírica no Conjunto de Teste Hold-Out (`n = 273`, `random_state=42`)
Calculando a similaridade máxima de Tanimoto ($S_{\max}$) com fingerprints ECFP4 (`1024 bits`) entre cada molécula de teste e o conjunto de treinamento (`1.091` moléculas) **`[Reproduzido na Fase 2A]`**:

| Faixa de Similaridade Máxima ($S_{\max}$) | $n$ (Teste) | MAE ($\text{pIC}_{50}$) | RMSE ($\text{pIC}_{50}$) | Diagnóstico de Cobertura |
| :--- | :--- | :--- | :--- | :--- |
| **$S_{\max} \ge 0{,}50$** (Dentro do Domínio) | `263` (`96,3%`) | **`0,410`** | **`0,570`** | Alta cobertura estrutural; erro alinhado à validação cruzada ($0{,}576$). |
| **$0{,}30 \le S_{\max} < 0{,}50$** (Domínio Limítrofe) | `5` (`1,8%`) | **`0,564`** | **`0,653`** | Cobertura moderada; aumento gradual da incerteza. |
| **$S_{\max} < 0{,}30$** (Fora do Domínio) | `5` (`1,8%`) | **`0,850`** | **`1,027`** | **Extrapolação severa** (erro dobra: $\text{RMSE} > 1{,}0\text{ log}$). |

### Envelope Físico-Químico do Treinamento $\text{A}_{2\text{A}}$ (`N = 1.364`) **`[Reproduzido na Fase 2A]`**
- `MolWt`: mín $= 180{,}17\text{ Da}$ ($Q_{1\%} = 258{,}81$, $Q_{99\%} = 716{,}82$, máx $= 878{,}14\text{ Da}$).
- `MolLogP`: mín $= -1{,}84$ ($Q_{1\%} = 0{,}51$, $Q_{99\%} = 5{,}77$, máx $= 7{,}80$).
- `TPSA`: mín $= 9{,}49\text{ \AA}^2$ ($Q_{1\%} = 46{,}61$, $Q_{99\%} = 201{,}15$, máx $= 268{,}68\text{ \AA}^2$).

---

## 2.6 Investigação 6 — Configuração do OpenBLAS no Windows

1. **Ordem de Importação Real **`[Reproduzido na Fase 2A]`**:**
   - Quando iniciado via `streamlit run app.py`, o executável `streamlit.exe` importa `numpy`, `pandas` e `pyarrow` durante o bootstrap do CLI **antes** de ler `app.py`. Assim, `os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")` dentro de `app.py` ocorre após a inicialização da DLL `libopenblas`.
   - Quando iniciado via `python script.py` ou `pytest`, `os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")` funciona se posicionado no topo de `src/__init__.py` e `app.py` antes de `import numpy`.
2. **Mitigação em Dupla Camada:**
   - Definir `os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")` e `os.environ.setdefault("OMP_NUM_THREADS", "1")` no topo absoluto de `src/__init__.py` e `app.py` + acionar `threadpoolctl.threadpool_limits(limits=1)` em tempo de execução.
   - Incluir um script lançador local (`run_app.bat` / `run_app.ps1`) que exporta `OPENBLAS_NUM_THREADS=1` no processo local antes de invocar `streamlit run app.py`.

---

# Parte III — Matriz Consolidada de Achados e Testes de Aceitação

| ID do achado | Evidência reproduzida | Conclusão técnica | Correção recomendada | Arquivos potencialmente afetados | Risco de regressão | Teste de aceitação |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **ACHADO-01**<br>*(Padronização em `featurize_dataframe`)* | Em [`src/chem_utils.py`](src/chem_utils.py#L211), `featurize_dataframe()` usa `Chem.MolFromSmiles` direto. Gera diferença de $3\times 10^{-16}$ em `MolLogP` para SMILES não-canônico da istradefilina e não remove sais em lotes brutos. Impacto nas predições atuais dos 5 controles e nos Parquets de treino: **`0,0000`** **`[Reproduzido na Fase 2A]`**. | Falha de implementação na função de featurização em lote, que ignora `standardize_mol()` e `canonicalize_smiles()`. | Usar `smiles_to_mol(str(s), standardize=True)` e converter para SMILES canônico antes de calcular descritores e Morgan bits em `featurize_dataframe()`, adicionando retorno estruturado de erros em lote. | [`src/chem_utils.py`](src/chem_utils.py) | **Nulo** (datasets de treino já estão 100% canônicos; predições dos 5 controles inalteradas). | Teste unitário verificando igualdade bit-a-bit (`assert_array_equal`) entre SMILES Kekulé, SMILES canônico e SMILES com sal (`.Cl`), mais reporte de SMILES inválido sem exceção. |
| **ACHADO-02**<br>*(Identidade, ID ChEMBL e endpoint da Istradefilina)* | O SMILES no código é autenticamente a istradefilina (`InChIKey=IQVRBWUUXZMOPW-PKNBQFBNSA-N`), cujo ID real nos dados é **`CHEMBL431770`** (e não `CHEMBL223405`). Ela está **dentro do treino** (`idx=1315`, `pIC50=5.2799`, oriundo de 2 ensaios funcionais `assay_type='F'` de inibição de cAMP induzido por NECA, `CHEMBL2384072`). O modelo de produção prevê `5.5053` (`5,51`) **`[Reproduzido na Fase 2A]`**. | Erro de rotulagem (`CHEMBL223405` no código; SMILES inválido em `DEVELOPMENT.md`) e erro de interpretação farmacológica (tratar $\text{pIC}_{50}$ funcional competitivo com NECA como $\text{p}K_i > 6{,}5$ e usar molécula *in-sample* como validação externa). | **Sem retreino:** Corrigir ID para `CHEMBL431770`, corrigir SMILES em `DEVELOPMENT.md`, e documentar na UI/notebooks que a referência experimental `IC50` no ChEMBL é funcional ($\text{pIC}_{50} = 5{,}28$, *in-sample*).<br>**Com retreino (opcional futuro):** Separar `assay_type == 'B'` de `'F'` e/ou ingerir `Ki`. | [`app.py`](app.py), [`DEVELOPMENT.md`](DEVELOPMENT.md), [`notebooks/05_model_a2a_training.ipynb`](notebooks/05_model_a2a_training.ipynb), [`notebooks/06_model_herg_training.ipynb`](notebooks/06_model_herg_training.ipynb), [`notebooks/07_multitarget_shap_evaluation.ipynb`](notebooks/07_multitarget_shap_evaluation.ipynb) | **Nulo** na correção documental e de metadados; não altera pesos dos modelos. | Verificação automatizada de que o SMILES dos controles é válido no RDKit, corresponde ao `CHEMBL431770` em `a2a_curated.csv` e exibe aviso de amostra *in-sample* ($\text{pIC}_{50\text{ exp}} = 5{,}28$). |
| **ACHADO-03**<br>*(Extrapolação fora do domínio para Taurina e L-Teanina)* | Taurina (`pred=5.4548`, $S_{\max}=0{,}117$, $\text{MolWt}=125{,}15\text{ Da}$) e L-Teanina (`pred=5.7962`, $S_{\max}=0{,}203$, $\text{MolWt}=174{,}20\text{ Da}$) estão abaixo do $\text{MolWt}_{\min}$ do treino ($180{,}17\text{ Da}$) e no percentil $< 1\%$ de similaridade. No teste hold-out, moléculas com $S_{\max} < 0{,}30$ têm $\text{RMSE} = 1{,}027$ vs. $0{,}570$ para $S_{\max} \ge 0{,}50$ **`[Reproduzido na Fase 2A]`**. | Limitação do domínio de aplicabilidade (AD) e viés de publicação do ChEMBL (`98,1%` das moléculas de treino têm $\text{pIC}_{50} \ge 5{,}0$; `base_value = 7.295`). | Implementar cálculo de Domínio de Aplicabilidade em `src/model_utils.py` e `app.py` (Tanimoto $S_{\max}$ contra `a2a_featurized.parquet` + envelope físico-químico), exibindo alerta técnico não-alarmista quando $S_{\max} < 0{,}30$ ou $\text{MolWt} < 180\text{ Da}$. | [`src/model_utils.py`](src/model_utils.py), [`app.py`](app.py) | **Nulo** nas predições numéricas (adiciona métrica diagnóstica $S_{\max}$ e status de cobertura sem alterar `predict()`). | Teste verificando que Istradefilina ($S_{\max\text{ excl}}=0{,}840$) recebe status "Dentro do Domínio", Cafeína ($S_{\max\text{ excl}}=0{,}389$) "Domínio Limítrofe" e Taurina/L-Teanina/Terfenadina ($S_{\max} < 0{,}30$) "Fora do Domínio". |
| **ACHADO-04**<br>*(Coordenadas de `axvspan` e 9 regiões no gráfico multiobjetivo)* | Com `ylim=(-0.05, 1.05)`, `axvspan(..., ymin=0.70, ymax=1.0)` sombreia $y_{\text{dados}} \in [0{,}72, 1{,}05]$ (erro de $+0{,}02$) e `ymax=0.30` sombreia $y_{\text{dados}} \in [-0{,}05, 0{,}28]$ (erro de $-0{,}02$). Além disso, `axvspan(3.0, 6.5)` rotula a faixa moderada ($5{,}0\text{–}6{,}5$) como "Baixa Potência" e omite as linhas $x=5{,}0$ e $y=0{,}30$ **`[Reproduzido na Fase 2A]`**. | Erro de implementação geométrica no Matplotlib (mistura de coordenadas de dados em $X$ e relativas em $Y$) e representação incompleta da grade $3 \times 3$. | Substituir `ax.axvspan` por `ax.fill_between([xmin, xmax], y1, y2)` em coordenadas de dados estritas e adicionar as linhas tracejadas secundárias em $x = 5{,}0$ e $y = 0{,}30$ para delimitar coerentemente as 9 regiões. | [`app.py`](app.py), [`notebooks/07_multitarget_shap_evaluation.ipynb`](notebooks/07_multitarget_shap_evaluation.ipynb) | **Nulo** (alteração estritamente visual/geométrica na camada de plotagem). | Inspecionar os objetos `PolyCollection`/`Line2D` do eixo Matplotlib via teste automatizado, confirmando vértices exatos em $y=0{,}70$, $y=0{,}30$, $x=5{,}0$ e $x=6{,}5$. |
| **ACHADO-05**<br>*(Ausência de teste estatístico e split por scaffolds)* | `notebooks/05` e `06` usam apenas 1 split aleatório (`random_state=42`) e `5-fold KFold` aleatório; em $\text{A}_{2\text{A}}$, a diferença de RMSE entre XGBoost (`0.5567`) e RF (`0.5621`) no hold-out é de `0.0054` sem teste pareado **`[Reproduzido na Fase 2A]`**. | Limitação metodológica de validação QSAR (séries congêneres análogas no treino e teste inflam métricas em split aleatório; na istradefilina há análogo com $S_{\max}=0{,}840$). | **Sem retreino:** Documentar a limitação de split aleatório e reportar a diferença XGB vs. RF como estatisticamente equivalente.<br>**Em reavaliação futura:** Implementar `ScaffoldSplit` (Bemis-Murcko) e teste de Wilcoxon pareado nos folds. | [`DEVELOPMENT.md`](DEVELOPMENT.md), [`src/model_utils.py`](src/model_utils.py) | **Nulo** se mantido como diagnóstico/documentação sem substituir `.joblib`. | Execução de função de avaliação comparativa em modo leitura confirmando cálculo de $p$-valor pareado nos 5 folds. |
| **ACHADO-06**<br>*(Probabilidades não calibradas em `herg_model`)* | `models/herg_model.joblib` é um `RandomForestClassifier(n_estimators=300, max_depth=20)` puro sem `CalibratedClassifierCV`. As probabilidades `predict_proba` são frequências brutas de folhas **`[Reproduzido na Fase 2A]`**. | Limitação estatística: frequências de árvores de decisão em Random Forest tendem a comprimir probabilidades longe de $0$ e $1$, não representando probabilidades frequentistas calibradas. | **Sem retreino:** Ajustar a terminologia na UI (`app.py`) de *"Probabilidade de Bloqueio"* para *"Escore Probabilístico RF ($P_{\text{hERG}}$ — não calibrado)"* e incluir Brier Score / curva de calibração diagnóstica. | [`app.py`](app.py), [`DEVELOPMENT.md`](DEVELOPMENT.md) | **Nulo** (preserva exatamente os mesmos valores de $P_{\text{hERG}}$ e limiares $0{,}30/0{,}70$). | Verificar na UI e testes que os valores de $P_{\text{hERG}}$ dos 5 controles permanecem inalterados e que o rótulo explicita escore RF não calibrado. |
| **ACHADO-07**<br>*(Desalinhamento documental: `PRD.md` vs. `README.md` vs. `DEVELOPMENT.md`)* | `PRD.md` cita `2.838` compostos $\text{A}_{2\text{A}}$ e `LightGBM`; `README.md` cita `XGBoost` para $\text{hERG}$; `DEVELOPMENT.md` tem SMILES inválido na linha 500 **`[Reproduzido na Fase 2A]`**. | Inconsistência documental entre especificações iniciais e artefatos finais (`1.364` $\text{A}_{2\text{A}}$ com XGBoost; `652` $\text{hERG}$ com Random Forest). | Sincronizar `README.md` e `DEVELOPMENT.md` com os metadados reais (`models/*_metadata.json`) e adicionar nota de histórico arquitetural no topo de `context/PRD.md`. | [`README.md`](README.md), [`DEVELOPMENT.md`](DEVELOPMENT.md), [`context/PRD.md`](context/PRD.md) | **Nulo** (apenas documentação). | Validação de que todos os números de amostras, modelos e SMILES citados na documentação coincidem com `models/*_metadata.json` e passam por `Chem.MolFromSmiles`. |
| **ACHADO-08**<br>*(Falha de alocação OpenBLAS no Windows e falta de suíte `tests/`)* | `streamlit.exe` importa `numpy` e `pandas` no bootstrap **antes** de executar `app.py`; scripts Python sem `OPENBLAS_NUM_THREADS=1` falham com `OpenBLAS error: Memory allocation still failed after 10 retries` **`[Reproduzido na Fase 2A]`**. | Problema de configuração de ambiente de execução no Windows (alocação excessiva de buffers de threads pelo OpenBLAS) e ausência de testes automatizados. | Adicionar `os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")` + `threadpoolctl.threadpool_limits(limits=1)` no topo de `src/__init__.py` e `app.py`, criar script lançador local (`run_app.bat`/`run_app.ps1`) e suíte `tests/`. | [`src/__init__.py`](src/__init__.py), [`app.py`](app.py), `run_app.bat`, `tests/` | **Nulo** (estabiliza a execução sem alterar resultados matemáticos). | Execução de `pytest` e inicialização do app sem erro de alocação OpenBLAS em sub-processos Windows. |

---

# Parte IV — Plano de Correção Controlada

## Grupo A — Correções Executáveis Sem Retreinar Modelos (Preservando `.joblib` e Limiares)
1. **Padronização Química em `featurize_dataframe()` (`ACHADO-01`):** Garantir que toda chamada a `featurize_dataframe()` passe por `standardize_mol()` + `canonicalize_smiles()`, removendo contra-íons automaticamente e reportando falhas por linha em processamentos em lote.
2. **Correção Geométrica do Gráfico Multiobjetivo (`ACHADO-04`):** Substituir `ax.axvspan` por `ax.fill_between` em coordenadas estritas de dados ($X, Y$) e traçar os 4 limiares ($x = 5{,}0; 6{,}5$ e $y = 0{,}30; 0{,}70$) para representar coerentemente as 9 regiões de decisão.
3. **Indicador de Domínio de Aplicabilidade (`ACHADO-03`):** Calcular a similaridade máxima de Tanimoto ($S_{\max}$) contra `a2a_featurized.parquet` e verificar os limites físico-químicos (`MolWt`, `MolLogP`, `TPSA`), exibindo aviso informativo não-alarmista quando a molécula estiver fora do espaço químico treinado ($S_{\max} < 0{,}30$ ou $\text{MolWt} < 180\text{ Da}$).
4. **Saneamento de Metadados de Controles e Documentação (`ACHADO-02`, `ACHADO-06`, `ACHADO-07`):**
   - Corrigir o identificador da istradefilina para `CHEMBL431770` e sinalizar na interface/documentação que Istradefilina ($\text{pIC}_{50\text{ exp}} = 5{,}28$, ensaio funcional `F`) e Cafeína ($\text{pIC}_{50\text{ exp}} = 4{,}85$, ensaio de ligação `B`) fazem parte do conjunto de treinamento de $\text{A}_{2\text{A}}$.
   - Corrigir o SMILES inválido na linha 500 de `DEVELOPMENT.md` e alinhar `README.md` com `models/*_metadata.json`.
   - Explicitar na interface que $P_{\text{hERG}}$ corresponde ao escore probabilístico Random Forest não calibrado.
5. **Estabilização do OpenBLAS e Criação de Testes Automatizados (`ACHADO-08`):** Configurar `os.environ.setdefault` + `threadpoolctl` em `src/__init__.py` e `app.py`, criar script de inicialização local (`run_app.bat`/`run_app.ps1`) e implementar a suíte `tests/` cobrindo todos os testes de aceitação.

## Grupo B — Melhorias que Exigiriam Nova Curadoria ou Retreinamento (Dependem de Aprovação Explícita)
1. **Separação de Ensaios de Ligação (`assay_type == 'B'`) vs. Funcionais (`'F'`) e Ingestão de $K_i$ (`ACHADO-02`):** Treinar modelos específicos por mecanismo de ensaio ($\text{p}K_i$ de deslocamento de radioligante vs. $\text{pIC}_{50}$ funcional celular).
2. **Inclusão de Inativos Verificados / Decoys (`ACHADO-03`):** Mitigar o viés de publicação do ChEMBL na região $\text{pIC}_{50} < 5{,}0$.
3. **Particionamento por Scaffolds (Bemis-Murcko) e Calibração Probabilística (`CalibratedClassifierCV`) (`ACHADO-05`, `ACHADO-06`):** Reavaliar a generalização *out-of-scaffold* e calibrar as probabilidades do classificador $\text{hERG}$.

---

# Parte V — Fase 3D-0: Auditoria de Leitura do Dataset hERG e do Domínio de Aplicabilidade

**Objetivo:** Resolver, com evidências do código e dos artefatos binários/tabulares no disco, a discrepância entre os **13.136** registros hERG e a menção a **652** moléculas na Parte I deste documento, determinando qual conjunto foi efetivamente utilizado para treinar [`models/herg_model.joblib`](models/herg_model.joblib), qual conjunto deve servir de referência para o Domínio de Aplicabilidade (AD) e o estado atual dos operadores dos limiares hERG (`0.30` e `0.70`).

---

## 5.1 Arquivos e Funções Inspecionados

### Código-fonte e Scripts de Construção
* [`src/data_utils.py`](src/data_utils.py):
  * `TDC_DATASETS` (linhas `37–41`) — Dicionário de mapeamento de IDs do Harvard Dataverse (`"herg": 4259588`, `"herg_karim": 6822246`, `"herg_central": 5740618`).
  * `download_tdc_dataset()` (linhas `48–85`) — Download e persistência bruta em `data/raw/tdc_<dataset_name>.csv`.
  * `canonicalize_smiles()` (linhas `93–115`) — Dessalinização (maior fragmento orgânico) e geração de SMILES canônico isomérico.
  * `curate_a2a_dataset()` (linhas `118–198`) — Curadoria ChEMBL (`CHEMBL251`, `IC50`, `=`, `nM`), conversão para `pIC50` e agregação por mediana (`std <= 1.0`).
  * `curate_herg_dataset()` (linhas `201–241`) — Curadoria binária do dataset hERG: padronização por `canonicalize_smiles()`, agrupamento por `canonical_smiles`, cálculo de `mean_label = mean(Y)` e retenção exclusiva de estruturas concordantes (`mean_label == 0.0` ou `mean_label == 1.0`).
* [`src/chem_utils.py`](src/chem_utils.py):
  * `featurize_smiles()` (linhas `140–185`) e `featurize_dataframe()` (linhas `193–253`) — Extração dos 6 descritores físico-químicos 2D e 1.024 bits Morgan/ECFP4 (`radius=2`, `nBits=1024`, `uint8`).
* [`src/model_utils.py`](src/model_utils.py):
  * `load_models()` (linhas `44–81`) e `predict_Safety_and_Potency()` (linhas `84–157`).
* [`app.py`](app.py):
  * Fluxo de predição, cartões semafóricos hERG (linhas `1551–1552`, `1714–1728`, `1814`) e `build_multiobjective_figure()` (linhas `1944–2035`) / diagnóstico de quadrantes (linhas `2067–2099`).
* Scripts auxiliares de geração dos notebooks e avaliação:
  * [`scratch/build_phase2_notebooks.py`](scratch/build_phase2_notebooks.py), [`scratch/test_herg_models.py`](scratch/test_herg_models.py) e [`scratch/build_herg_modeling_notebook.py`](scratch/build_herg_modeling_notebook.py).

### Notebooks, Artefatos, Metadados, Documentação e Testes
* Notebooks: [`notebooks/02_herg_curation.ipynb`](notebooks/02_herg_curation.ipynb), [`notebooks/04_herg_featurization_eda.ipynb`](notebooks/04_herg_featurization_eda.ipynb), [`notebooks/05_a2a_modeling.ipynb`](notebooks/05_a2a_modeling.ipynb), [`notebooks/06_herg_modeling.ipynb`](notebooks/06_herg_modeling.ipynb) e [`notebooks/07_multitarget_shap_evaluation.ipynb`](notebooks/07_multitarget_shap_evaluation.ipynb).
* Dados e Artefatos: `data/raw/tdc_herg_karim.csv`, `data/processed/herg_curated.csv`, `data/processed/herg_featurized.parquet`, `data/raw/chembl_a2a_raw.csv`, `data/processed/a2a_curated.csv`, `data/processed/a2a_featurized.parquet`, `data/processed/a2a_featurized.csv`, [`models/herg_model.joblib`](models/herg_model.joblib), [`models/herg_model_metadata.json`](models/herg_model_metadata.json), [`models/a2a_model.joblib`](models/a2a_model.joblib) e [`models/a2a_model_metadata.json`](models/a2a_model_metadata.json).
* Documentação e Testes: [`DEVELOPMENT.md`](DEVELOPMENT.md), [`context/PRD.md`](context/PRD.md), [`AUDITORIA.md`](AUDITORIA.md) e [`tests/test_controls_and_interpretation.py`](tests/test_controls_and_interpretation.py).

---

## 5.2 Totais Físicos e Definições de Contagem por Arquivo

Todos os arquivos de dados foram lidos diretamente do disco e inspecionados via `pandas` e `RDKit`:

### 5.2.1 Pipeline de Dados hERG

| Arquivo no Disco | Tamanho (Bytes) | `.shape` Físico | Colunas | Registros Totais | Estruturas Padronizadas Únicas (`canonical_smiles`) | Positivos (`1`) | Negativos (`0`) | Nulos / Inválidos |
| :--- | :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| `data/raw/tdc_herg_karim.csv` | `871.628` | `(13445, 3)` | `Drug_ID`, `Drug`, `Y` | `13.445` | `13.152` | `6.718` (`49,97%`) | `6.727` (`50,03%`) | `0` nulos / `0` SMILES inválidos |
| `data/processed/herg_curated.csv` | `875.685` | `(13136, 4)` | `Drug_ID`, `canonical_smiles`, `hERG_label`, `num_records` | `13.136` | `13.136` | `6.570` (`50,015%`) | `6.566` (`49,985%`) | `0` nulos / `0` duplicatas |
| `data/processed/herg_featurized.parquet` | `1.881.275` | `(13136, 1034)` | `4` metadados + `6` descritores 2D + `1.024` bits `fp_0..fp_1023` | `13.136` | `13.136` | `6.570` (`50,015%`) | `6.566` (`49,985%`) | `0` nulos / `0` duplicatas |

#### Contabilidade Exata da Deduplicação em `curate_herg_dataset()`:
1. **Registros brutos em `data/raw/tdc_herg_karim.csv`:** `13.445` linhas (com `13.445` strings `Drug` distintas antes da padronização).
2. **Padronização via `canonicalize_smiles()`:** Todas as `13.445` entradas geram moléculas RDKit válidas (`0` descartadas por erro de parse) e colapsam em **`13.152` estruturas padronizadas únicas** (`canonical_smiles`):
   * `12.867` estruturas aparecem apenas 1 vez (`12.867` linhas);
   * `277` estruturas aparecem 2 vezes (`554` linhas: `261` concordantes e `16` conflitantes);
   * `8` estruturas aparecem 3 vezes (`24` linhas: todas as `8` concordantes).
   * Total de registros extras devido a entradas repetidas da mesma estrutura canônica: $277 \times 1 + 8 \times 2 = 293$ linhas ($13.445 - 293 = 13.152$ estruturas únicas).
3. **Descarte de rótulos conflitantes (`0.0 < mean_label < 1.0`):** Exatamente **`16` estruturas únicas** (que correspondem a `32` linhas no CSV bruto, todas com `1` medição positiva e `1` medição negativa, i.e., `mean_label == 0.5`) são descartadas como inconclusivas.
4. **Resultado curado final (`data/processed/herg_curated.csv` e `data/processed/herg_featurized.parquet`):**
   * $13.152 - 16 = \mathbf{13.136}$ **estruturas padronizadas únicas** (`canonical_smiles`), que agregam $13.445 - 32 = 13.413$ medições originais (`sum(num_records) == 13413`).
   * A correspondência entre `herg_curated.csv` e `herg_featurized.parquet` nas colunas `canonical_smiles` e `hERG_label` é **100% idêntica linha a linha** (`(df_cur['canonical_smiles'] == df_feat['canonical_smiles']).all() == True`).

### 5.2.2 Pipeline de Dados A2A

| Arquivo no Disco | Tamanho (Bytes) | `.shape` Físico | Registros / Estruturas Únicas | Observações |
| :--- | :---: | :---: | :---: | :--- |
| `data/raw/chembl_a2a_raw.csv` | `1.531.696` | `(3050, 47)` | `3.050` bioatividades brutas | Alvo `CHEMBL251` (`IC50`, `=`, `nM`) |
| `data/processed/a2a_curated.csv` | `124.212` | `(1364, 5)` | `1.364` `canonical_smiles` únicos | Colunas: `molecule_chembl_id`, `canonical_smiles`, `pIC50_median`, `pIC50_std`, `num_records` |
| `data/processed/a2a_featurized.parquet` | `400.785` | `(1364, 1035)` | `1.364` `canonical_smiles` únicos | `5` metadados + `6` descritores + `1.024` bits `fp_0..fp_1023` |
| `data/processed/a2a_featurized.csv` | `2.905.331` | `(1364, 1035)` | `1.364` `canonical_smiles` únicos | Idêntico ao Parquet (`pIC50_median` médio = `7,294726`) |

---

## 5.3 Explicação Documentada da Discrepância entre `13.136` e `652`

A investigação do código, do histórico Git (`git log --all --stat`) e dos logs de transcrição da sessão esclarece integralmente a origem do número `652`:

1. **Inexistência de qualquer dataset de 652 moléculas no repositório ou histórico Git:**
   * Em [`src/data_utils.py`](src/data_utils.py#L37-L41), o dicionário `TDC_DATASETS` mapeia três datasets do Therapeutics Data Commons:
     ```python
     TDC_DATASETS = {
         "herg": 4259588,         # ~648 compostos (Wang et al.)
         "herg_karim": 6822246,   # ~13.4k compostos, balanceado (Karim et al.)
         "herg_central": 5740618, # ~300k compostos
     }
     ```
   * O notebook [`notebooks/02_herg_curation.ipynb`](notebooks/02_herg_curation.ipynb#L76) invocou exclusivamente `download_tdc_dataset("herg_karim")`. O dataset menor `"herg"` (`file_id = 4259588`, ~655 compostos brutos / ~652 curados) **jamais foi baixado, processado ou commitado** em nenhum commit do repositório (desde o commit inicial `f0d73a2` da Fase 1, `data/raw/tdc_herg_karim.csv` tem `13.445` linhas e `data/processed/herg_curated.csv` tem `13.136` linhas).
2. **Origem do número `652` na Parte I deste documento:**
   * Na execução original da auditoria de Fase 1, os totais reais (`13.445` brutos e `13.136` curados de `herg_karim`) foram lidos corretamente.
   * Entretanto, ao consolidar este documento na Etapa 5 (após compactação de contexto), a **Parte I** (linhas 38–39, 72, 85–86, 94) confundiu o dataset `herg_karim` com o benchmark clássico menor `herg` do TDC (~655 registros brutos / ~652 curados) e registrou erroneamente `652` moléculas e hiperparâmetros `n_estimators=300, max_depth=20, class_weight='balanced'`.
3. **Outras ocorrências literais de `"652"` no projeto:**
   * Em [`app.py`](app.py#L586) (linhas `586`, `713`, `809`) e [`DEVELOPMENT.md`](DEVELOPMENT.md#L366) (linhas `366`, `470`, `519`), o número `652` refere-se exclusivamente ao resíduo de aminoácido **`Tyr652`** do canal hERG (Kv11.1) e ao bit **`fp_652`** do fingerprint Morgan.

**Conclusão sobre a discrepância:** `652` é um erro material exclusivo do texto da Parte I deste documento. O único conjunto hERG existente no repositório e utilizado em todo o pipeline é `herg_karim`, contendo **13.136 moléculas curadas**.

---

## 5.4 Evidências sobre o Conjunto Usado no Treinamento de `models/herg_model.joblib`

A prova de que [`models/herg_model.joblib`](models/herg_model.joblib) foi treinado sobre as **13.136 moléculas** de `data/processed/herg_featurized.parquet` (e que [`models/a2a_model.joblib`](models/a2a_model.joblib) foi treinado sobre as **1.364 moléculas** de `data/processed/a2a_featurized.parquet`) é convergente em três níveis independentes:

### 5.4.1 Evidência no Código de Treinamento e Serialização
* Em [`notebooks/06_herg_modeling.ipynb`](notebooks/06_herg_modeling.ipynb#L204-L215) (gerado por [`scratch/build_herg_modeling_notebook.py`](scratch/build_herg_modeling_notebook.py#L60-L92)):
  1. Carrega `parquet_path = os.path.join('..', 'data', 'processed', 'herg_featurized.parquet')` (`13.136` linhas).
  2. Extrai `X = df[list(FEATURE_NAMES)].copy()` (`shape = (13136, 1030)`) e `y = df['hERG_label'].values` (`shape = (13136,)`).
  3. Executa `train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)` (linhas `272–277`), resultando em:
     * `X_train`: **`10.508` amostras** (`5.256` classe `1` / `5.252` classe `0`);
     * `X_test`: **`2.628` amostras** (`1.314` classe `1` / `1.314` classe `0`).
  4. Avalia `RandomForestClassifier(n_estimators=150, min_samples_split=4, random_state=42, n_jobs=2)` no holdout (`ROC-AUC = 0.9299`, `PR-AUC = 0.9333`, `Brier = 0.1118`).
  5. Nas linhas `1168–1180` de [`notebooks/06_herg_modeling.ipynb`](notebooks/06_herg_modeling.ipynb#L1168-L1180), instancia `final_model = RandomForestClassifier(n_estimators=150, min_samples_split=4, random_state=42, n_jobs=2)`, executa **`final_model.fit(X, y)` sobre 100% da matriz `X` (`13.136` amostras)** e salva em `models/herg_model.joblib` com `joblib.dump(final_model, model_path, compress=3)`.

### 5.4.2 Evidência nos Metadados JSON
* [`models/herg_model_metadata.json`](models/herg_model_metadata.json) registra:
  * `"algorithm": "Random Forest Classifier"`
  * `"hyperparameters": {"n_estimators": 150, "min_samples_split": 4, "random_state": 42, "n_jobs": 2}`
  * `"training_samples_total": 13136`
  * `"features_count": 1030`
  * `"performance_holdout_test_20pct": {"samples": 2628, "ROC_AUC": 0.9299293540965219, "Brier_Score": 0.11182534300877001, ...}`

### 5.4.3 Evidência Interna Direta nos Objetos Binários `.joblib`

A inspeção programática dos atributos internos de [`models/herg_model.joblib`](models/herg_model.joblib) e [`models/a2a_model.joblib`](models/a2a_model.joblib) confirma matematicamente os conjuntos utilizados:

| Atributo Interno Inspecionado | `models/herg_model.joblib` (`RandomForestClassifier`) | `models/a2a_model.joblib` (`XGBRegressor`) |
| :--- | :--- | :--- |
| Classe / Tipo | `sklearn.ensemble._forest.RandomForestClassifier` | `xgboost.sklearn.XGBRegressor` |
| `n_features_in_` | `1030` | `1030` |
| `feature_names_in_` | `['MolWt', 'MolLogP', 'TPSA', 'NumHDonors', 'NumHAcceptors', 'NumRotatableBonds', 'fp_0', ..., 'fp_1023']` | `['MolWt', 'MolLogP', 'TPSA', 'NumHDonors', 'NumHAcceptors', 'NumRotatableBonds', 'fp_0', ..., 'fp_1023']` |
| Número de árvores | `len(estimators_) == 150` | `num_boosted_rounds == 250` |
| Hiperparâmetros internos | `min_samples_split=4`, `max_depth=None`, `class_weight=None`, `bootstrap=True`, `random_state=42` | `learning_rate=0.05`, `max_depth=6`, `subsample=0.8`, `colsample_bytree=0.8`, `random_state=42` |
| **Tamanho exato de treino gravado na estrutura interna** | Em **todas as 150 árvores**, `est.tree_.weighted_n_node_samples[0]` no nó raiz (`node 0`) é **exatamente `13136.0`** (`min = 13136.0`, `max = 13136.0`). | O parâmetro `base_score` interno do booster é **`7.294726`**, que coincide na 6ª casa decimal com `df_a2a['pIC50_median'].mean()` = **`7.2947258`** das **`1.364` moléculas**. |
| Amostras únicas de bootstrap no nó raiz (`n_node_samples[0]`) | Média de **`8304.7`** amostras únicas por árvore ($13.136 \times (1 - e^{-1}) \approx 8303{,}5$). | — |
| Proporção de classes no nó raiz (`est.tree_.value[0, 0]`) | Média `[0.5000132, 0.4999868]` nas 150 árvores (idêntica ao balanço `6.566` / `6.570` de `herg_featurized.parquet`). | — |

> [!NOTE]
> **Recuperabilidade das moléculas apenas a partir do `.joblib`:** Nem `RandomForestClassifier` nem `XGBRegressor` armazenam as matrizes de entrada `X` ou as strings `canonical_smiles` dentro do arquivo `.joblib` — eles preservam apenas os limiares de corte das colunas (`feature_names_in_`) e as estatísticas agregadas dos nós (`weighted_n_node_samples`, `value`, `base_score`). Portanto, para calcular similaridade estrutural (Tanimoto) no Domínio de Aplicabilidade, é indispensável carregar os conjuntos de treinamento a partir dos arquivos persistidos em `data/processed/`.

---

## 5.5 Conjunto de Referência Recomendado para o Domínio de Aplicabilidade

Com base nas evidências acima, o Domínio de Aplicabilidade (AD) deve utilizar referências independentes por endpoint, refletindo exatamente as moléculas com as quais cada modelo final (`100% refit`) foi ajustado:

1. **Endpoint hERG (`models/herg_model.joblib`):**
   * **Conjunto de referência:** As **13.136 moléculas únicas** (`canonical_smiles`) de `data/processed/herg_featurized.parquet` (ou `data/processed/herg_curated.csv`, que contém exatamente as mesmas `13.136` moléculas na mesma ordem).
   * **Vantagem técnica de `data/processed/herg_featurized.parquet`:** Já contém as `13.136` estruturas padronizadas deduplicadas e a matriz pré-computada dos `1.024` bits Morgan/ECFP4 (`fp_0` a `fp_1023`, `uint8`), evitando recalcular fingerprints a partir de SMILES a cada inicialização.
2. **Endpoint A2A (`models/a2a_model.joblib`):**
   * **Conjunto de referência:** As **1.364 moléculas únicas** (`canonical_smiles`) de `data/processed/a2a_featurized.parquet` (ou `data/processed/a2a_curated.csv`).
   * Assim como no hERG, `data/processed/a2a_featurized.parquet` já possui os `1.024` bits Morgan/ECFP4 (`fp_0` a `fp_1023`, `uint8`) alinhados com as `1.364` moléculas de treinamento do regressor `XGBRegressor`.

---

## 5.6 Divergências Encontradas nos Operadores dos Limiares hERG (`0.30` e `0.70`)

A revisão de todo o repositório confrontou os operadores atualmente implementados com a especificação de referência:
* **Baixo:** `PhERG < 0.30` (equivalente no eixo complementar $y = 1 - P_{\text{hERG}}$ a $y > 0.70$)
* **Intermediário:** `0.30 <= PhERG < 0.70` (equivalente a $0.30 < y \le 0.70$)
* **Alto:** `PhERG >= 0.70` (equivalente a $y \le 0.30$)

### Resultado da Inspeção
* Na fronteira **`0.30`**, o código de classificação de cartões e quadrantes **já utiliza `prob_herg < 0.30`** para Baixo Risco e inclui `0.30` na faixa Intermediária (`>= 0.30` ou `y_cur <= 0.70`).
* Na fronteira **`0.70`**, porém, **há divergência sistemática**: o código, os metadados, os notebooks, a documentação e um teste de regressão utilizam `<= 0.70` para Intermediário e `> 0.70` para Alto (ou `y_cur < 0.30` no eixo complementar), em vez de `< 0.70` para Intermediário e `>= 0.70` (`y_cur <= 0.30`) para Alto.

A tabela abaixo lista **todos os arquivos e linhas exatas** onde ocorre a divergência na fronteira `0.70` (ou em rótulos/comentários associados):

| Arquivo | Linhas | Código / Texto Atual | Divergência em Relação ao Alvo (`0.30 <= PhERG < 0.70` e `PhERG >= 0.70`) |
| :--- | :---: | :--- | :--- |
| [`app.py`](app.py#L1551-L1552) | `1551–1552` | `- **Moderado (Alerta):** $0,30 \le P \le 0,70$`<br>`- **Alto (Tóxico):** $P > 0,70$` | Usa $\le 0,70$ no Moderado/Intermediário e $> 0,70$ no Alto (deveria ser $< 0,70$ e $\ge 0,70$). |
| [`app.py`](app.py#L1719-L1728) | `1719–1728` | `elif prob_herg <= 0.70:` (`"0,30 ≤ P_hERG ≤ 0,70"`) e `else:` (`"P_hERG > 0,70"`) | Caso `prob_herg == 0.70`, classifica como `"Risco Moderado"` em vez de `"Alto Risco"`. Deve usar `elif prob_herg < 0.70:` e rótulos `0,30 ≤ P_hERG < 0,70` e `P_hERG ≥ 0,70`. |
| [`app.py`](app.py#L1814) | `1814` | `herg_bar_color = "#00CC96" if prob_herg < 0.30 else ("#F59E0B" if prob_herg <= 0.70 else "#FF4B4B")` | Usa `prob_herg <= 0.70` para a cor amarela (deveria ser `prob_herg < 0.70`). |
| [`app.py`](app.py#L1975-L1983) | `1975`, `1983` | `label="Zona de Alto Risco hERG (P_hERG > 0,70)"` e `"Limiar Risco Alto hERG (P_hERG > 0,70)"` | Rótulos da legenda em `build_multiobjective_figure()` exibem `P_hERG > 0,70` em vez de `P_hERG ≥ 0,70`. |
| [`app.py`](app.py#L2074-L2097) | `2074`, `2079`, `2088`, `2091`, `2097` | `elif x_cur >= 6.5 and y_cur < 0.30:` (`$P_{\text{hERG}} > 0,70$`) e `elif x_cur < 6.5 and y_cur < 0.30:` (`$P_{\text{hERG}} > 0,70$`) e `$0,30 \le P_{\text{hERG}} \le 0,70$` | Como `y_cur = 1.0 - prob_herg`, `y_cur < 0.30` equivale a `prob_herg > 0.70`. Para `prob_herg >= 0.70`, a condição exata é `y_cur <= 0.30` (ou testar diretamente `prob_herg >= 0.70`). |
| [`DEVELOPMENT.md`](DEVELOPMENT.md#L581) | `581`, `622–623` | `$30\% \le P_{\text{hERG}} \le 70\%$` / `$0,30 \le P_{\text{hERG}} \le 0,70$` e `$P_{\text{hERG}} > 70\%$` | Documentação usa $\le 0,70$ e $> 0,70$ em vez de $< 0,70$ e $\ge 0,70$. |
| [`context/PRD.md`](context/PRD.md#L67-L68) | `67–68` | `$30\% \le P_{\text{hERG}} \le 70\%$` e `$P_{\text{hERG}} > 70\%$` | Documentação de requisitos usa $\le 70\%$ e $> 70\%$. |
| [`models/herg_model_metadata.json`](models/herg_model_metadata.json#L1077-L1078) | `1077–1078` | `"yellow_moderate_risk": "0.30 <= P(hERG) <= 0.70"`, `"red_high_risk": "P(hERG) > 0.70"` | Metadado JSON usa `<= 0.70` e `> 0.70`. |
| [`notebooks/06_herg_modeling.ipynb`](notebooks/06_herg_modeling.ipynb#L1116-L1118) | `1116–1118`, `1228–1229` | `if p > 0.70:` / `elif p >= 0.30:` e strings `"0.30 <= P(hERG) <= 0.70"` / `"P(hERG) > 0.70"` | Funções auxiliares e geração de metadados no notebook usam `p > 0.70`. |
| [`notebooks/07_multitarget_shap_evaluation.ipynb`](notebooks/07_multitarget_shap_evaluation.ipynb#L330-L332) | `330–332` | `if p > 0.70: return 'ALTO RISCO'` / `elif p >= 0.30: return 'ALERTA MODERADO'` | Função de triagem no notebook usa `p > 0.70` em vez de `p >= 0.70`. |
| [`tests/test_controls_and_interpretation.py`](tests/test_controls_and_interpretation.py#L248) | `248` | `assert "prob_herg <= 0.70" in app_text` | Asserção literal do teste verifica a string antiga `prob_herg <= 0.70`. |
| [`tests/test_multiobjective_plot.py`](tests/test_multiobjective_plot.py#L102) | `102` | Docstring/mensagem de `assert`: `"deve cobrir y >= 0.70 (1 - PhERG >= 0.70)"` | Apenas texto de comentário/mensagem (no Matplotlib `fill_between` a fronteira contínua `[0.70, 1.05]` é idêntica, mas a inequação estrita para `PhERG < 0.30` é `1 - PhERG > 0.70`). |

### 5.7 Resolução e Homologação das Fronteiras hERG na Fase 3C-1

Todas as divergências listadas na Seção 5.6 foram corrigidas na **Fase 3C-1**, adotando-se a convenção canônica em [`app.py`](app.py), [`DEVELOPMENT.md`](DEVELOPMENT.md), [`context/PRD.md`](context/PRD.md), [`models/herg_model_metadata.json`](models/herg_model_metadata.json), [`notebooks/06_herg_modeling.ipynb`](notebooks/06_herg_modeling.ipynb), [`notebooks/07_multitarget_shap_evaluation.ipynb`](notebooks/07_multitarget_shap_evaluation.ipynb) e na suíte `tests/` (`15 passed`):
* **Baixo Risco Operacional:** $P_{\text{hERG}} < 0{,}30$ ($y = 1 - P_{\text{hERG}} > 0{,}70$)
* **Risco Intermediário/Moderado Operacional:** $0{,}30 \le P_{\text{hERG}} < 0{,}70$ ($0{,}30 < y \le 0{,}70$)
* **Alto Risco Operacional:** $P_{\text{hERG}} \ge 0{,}70$ ($y \le 0{,}30$, com tolerância `np.isclose(..., atol=1e-12)` para evitar desvios de ponto flutuante IEEE 754 em $1{,}0 - 0{,}70$ e $1{,}0 - 0{,}30$, e validação explícita de fronteira em `0.299`, `0.300`, `0.699`, `0.700` e `0.701`).

---

# Parte VI — Fase 3D-1: Diagnóstico Exploratório do Domínio de Aplicabilidade

**Objetivo:** Realizar uma análise diagnóstica e estritamente somente leitura do Domínio de Aplicabilidade (AD) dos modelos $\text{A}_{2\text{A}}$ e $\text{hERG}$, utilizando exatamente os compostos e as representações moleculares (Morgan/ECFP4, raio 2, 1.024 bits) dos conjuntos de treinamento efetivamente usados nos artefatos de produção.

---

## 6.1 Arquivos e Dimensões Verificados (Auditoria de Disponibilidade e Paridade)

A inspeção de leitura confrontou diretamente os arquivos tabulares e binários em `data/processed/` e `models/` com o pipeline de quimioinformática definido em [`src/chem_utils.py`](src/chem_utils.py):

| Atributo Verificado | Endpoint $\text{A}_{2\text{A}}$ | Endpoint $\text{hERG}$ |
| :--- | :--- | :--- |
| **CSV Curado** | `data/processed/a2a_curated.csv` | `data/processed/herg_curated.csv` |
| **`.shape` do CSV Curado** | `(1364, 5)` | `(13136, 4)` |
| **Colunas do CSV Curado** | `molecule_chembl_id`, `canonical_smiles`, `pIC50`, `num_records`, `pIC50_std` | `Drug_ID`, `canonical_smiles`, `hERG_label`, `num_records` |
| **Parquet Featurizado** | `data/processed/a2a_featurized.parquet` | `data/processed/herg_featurized.parquet` |
| **`.shape` do Parquet** | `(1364, 1035)` (`5` metadados + `1.030` features) | `(13136, 1034)` (`4` metadados + `1.030` features) |
| **Modelo Serializado (`.joblib`)** | [`models/a2a_model.joblib`](models/a2a_model.joblib) (`n_features_in_ = 1030`, `base_score = 7.294726`) | [`models/herg_model.joblib`](models/herg_model.joblib) (`n_features_in_ = 1030`, `weighted_n_node_samples[0] = 13136.0`) |
| **Coluna de Estrutura Química** | `canonical_smiles` (gerada via `canonicalize_smiles()`: `LargestFragmentChooser` + `Chem.MolToSmiles(..., canonical=True, isomericSmiles=True)`) | `canonical_smiles` (mesma padronização via `canonicalize_smiles()`) |
| **Moléculas RDKit Válidas / Exclusões** | **`1.364 / 1.364`** (`0` inválidas, `0` exclusões) | **`13.136 / 13.136`** (`0` inválidas, `0` exclusões) |
| **Strings `canonical_smiles` Únicas** | **`1.364`** (`100%` idempotentes ao reaplicar `canonicalize_smiles()`) | **`13.136`** (`100%` idempotentes ao reaplicar `canonicalize_smiles()`) |
| **Paridade CSV Curado $\leftrightarrow$ Parquet** | **`100%`** (`canonical_smiles` e `pIC50` idênticos linha a linha) | **`100%`** (`canonical_smiles` e `hERG_label` idênticos linha a linha) |
| **Paridade Bits Recalculados $\leftrightarrow$ Parquet** | **`1.364 / 1.364`** (`100%` idênticos a `fp_0..fp_1023`) | **`13.136 / 13.136`** (`100%` idênticos a `fp_0..fp_1023`) |
| **Vetores `fp_0..fp_1023` Distintos** | **`1.277`** vetores únicos entre `1.364` moléculas | **`11.073`** vetores únicos entre `13.136` moléculas |
| **`InChIKey` Completos Distintos** | **`1.364`** | **`12.950`** (devido a `182` grupos tautoméricos/mesoméricos; ver Seção 6.5) |
| **Blocos de Conectividade 2D (`InChIKey` 1º bloco)** | **`1.278`** esqueletos 2D distintos | **`10.897`** esqueletos 2D distintos |
| **Bits Ativos por Molécula (`onBits`)** | Mín: `18` · Mediana: `58,0` · Média: `57,0` · Máx: `95` | Mín: `3` · Mediana: `56,0` · Média: `55,0` · Máx: `93` |

---

## 6.2 Método de Cálculo e Estratégia Computacional

1. **Representação Molecular Exata do Pipeline:**
   * Fingerprints circulares **Morgan / ECFP4** com `radius = 2` e `nBits = 1024` (definidos em [`src/chem_utils.py`](src/chem_utils.py#L93-L115) via `rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)`).
2. **Estratégia Vetorizada em C++ (`DataStructs.BulkTanimotoSimilarity`):**
   * As matrizes `fp_0..fp_1023` (`uint8`) dos Parquets foram convertidas em listas de objetos `DataStructs.ExplicitBitVect(1024)` do RDKit.
   * Para cada molécula $i \in \{0, \dots, N-1\}$, o vetor de similaridades em relação a todas as moléculas do conjunto foi calculado em C++ via `DataStructs.BulkTanimotoSimilarity(bitvects[i], bitvects)`, atribuindo `sims[i] = -1.0` para excluir estritamente a auto-comparação ($j \neq i$) sem alocar uma matriz densa $N \times N$ em memória.
3. **Desempenho e Consumo de Memória Medidos (`tracemalloc`):**
   * **Pico de memória RAM Python:** **`106,58 MB`** (muito abaixo do teto de `2 GB` de inferência / `8 GB` da máquina).
   * **Tempo de execução Leave-One-Out ($\text{A}_{2\text{A}}$, $1.364 \times 1.363$ pares):** **`2,76 segundos`**.
   * **Tempo de execução Leave-One-Out ($\text{hERG}$, $13.136 \times 13.135 \approx 1{,}725 \times 10^8$ pares):** **`302,63 segundos`** (~5 min para a matriz completa de 172,5 milhões de pares; para **uma única molécula de consulta** na inferência contra as `13.136` moléculas hERG, o tempo médio de `BulkTanimotoSimilarity` é de apenas **`23 milissegundos`**).

---

## 6.3 Estatísticas Completas da Similaridade Máxima ($S_{\max}$) para $\text{A}_{2\text{A}}$ e $\text{hERG}$

Como `canonicalize_smiles()` preserva estereoquímica (`isomericSmiles=True`), mas o gerador padrão de Morgan fingerprint opera em topologia 2D aquiral (`includeChirality=False`), estereoisômeros distintos presentes no mesmo dataset possuem `canonical_smiles` diferentes, porém **vetores de 1.024 bits idênticos ($S_{\text{Tanimoto}} = 1{,}0000$)**. Por transparência diagnóstica, a tabela abaixo reporta tanto o $S_{\max}$ **Leave-One-Out por índice ($j \neq i$)** quanto o $S_{\max}$ **excluindo vetores de bits idênticos ($S < 1{,}0$)**:

| Estatística de $S_{\max}$ | $\text{A}_{2\text{A}}$ — LOO por Índice ($j \neq i$) | $\text{A}_{2\text{A}}$ — Excluindo FP Idêntico ($S < 1{,}0$) | $\text{hERG}$ — LOO por Índice ($j \neq i$) | $\text{hERG}$ — Excluindo FP Idêntico ($S < 1{,}0$) |
| :--- | :---: | :---: | :---: | :---: |
| **Moléculas Válidas ($N$)** | `1.364` | `1.364` | `13.136` | `13.136` |
| **Mínimo (`Min`)** | `0,1642` | `0,1642` | `0,1000` | `0,1000` |
| **Percentil 1 (`P01`)** | `0,2447` | `0,2447` | `0,2880` | `0,2783` |
| **Percentil 5 (`P05`)** | **`0,5653`** | **`0,5653`** | **`0,3881`** | **`0,3651`** |
| **Percentil 10 (`P10`)** | **`0,6878`** | **`0,6757`** | **`0,5571`** | **`0,5000`** |
| **1º Quartil (`Q1 / P25`)** | `0,7692` | `0,7647` | `0,7377` | `0,6988` |
| **Mediana (`P50`)** | **`0,8219`** | **`0,8108`** | **`0,8308`** | **`0,7885`** |
| **Média (`Mean`)** | **`0,8118`** | **`0,7898`** | **`0,8121`** | **`0,7440`** |
| **Desvio-Padrão (`Std`)** | `0,1384` | `0,1230` | `0,1829` | `0,1553` |
| **3º Quartil (`Q3 / P75`)** | `0,8788` | `0,8571` | `1,0000` | `0,8448` |
| **Percentil 90 (`P90`)** | `1,0000` | `0,8919` | `1,0000` | `0,8889` |
| **Percentil 95 (`P95`)** | `1,0000` | `0,9434` | `1,0000` | `0,9105` |
| **Percentil 99 (`P99`)** | `1,0000` | `0,9654` | `1,0000` | `0,9717` |
| **Máximo (`Max`)** | `1,0000` | `0,9890` | `1,0000` | `0,9870` |
| **Casos com $S_{\max} = 1{,}0000$ ($j \neq i$)** | `160` (`11,73%`) | `0` (`0,00%`) | `3.855` (`29,35%`) | `0` (`0,00%`) |

### Distribuição por Faixas de Similaridade ($j \neq i$)

| Faixa de $S_{\max}$ | $\text{A}_{2\text{A}}$ ($N = 1.364$) | $\text{A}_{2\text{A}}$ (`%`) | $\text{hERG}$ ($N = 13.136$) | $\text{hERG}$ (`%`) |
| :---: | :---: | :---: | :---: | :---: |
| `[0,0; 0,1)` | `0` | `0,00%` | `0` | `0,00%` |
| `[0,1; 0,2)` | `3` | `0,22%` | `4` | `0,03%` |
| `[0,2; 0,3)` | `20` | `1,47%` | `182` | `1,39%` |
| `[0,3; 0,4)` | `16` | `1,17%` | `523` | `3,98%` |
| `[0,4; 0,5)` | `13` | `0,95%` | `398` | `3,03%` |
| `[0,5; 0,6)` | `26` | `1,91%` | `472` | `3,59%` |
| `[0,6; 0,7)` | `73` | `5,35%` | `978` | `7,45%` |
| `[0,7; 0,8)` | `375` | `27,49%` | `2.650` | `20,17%` |
| `[0,8; 0,9)` | `566` | `41,50%` | `3.397` | `25,86%` |
| `[0,9; 1,0]` | `272` | `19,94%` | `4.532` | `34,50%` |

### Exemplos de Extremos Estruturais Observados nos Conjuntos de Treinamento

* **Menor similaridade máxima em $\text{A}_{2\text{A}}$ (*singletons* estruturais):**
  * `idx=1243` (`CHEMBL1475`, `Cc1cc2cc3c(C)cc(=O)oc3c(C)c2o1`, derivado psoraleno/furocumarina, $\text{pIC}_{50} = 5{,}88$, `29` bits ativos): **$S_{\max} = 0{,}1642$** (vizinho mais próximo: `CHEMBL5824224`).
  * `idx=1296` (`CHEMBL64894`, `CN(C)c1ccc(C(=C2C=CC(=[N+](C)C)C=C2)c2ccc(N(C)C)cc2)cc1`, cátion cristal violeta, $\text{pIC}_{50} = 5{,}47$, `27` bits ativos): **$S_{\max} = 0{,}1842$**.
  * `idx=1352` (`CHEMBL44`, genisteína `O=c1c(-c2ccc(O)cc2)coc2cc(O)cc(O)c12`, isoflavona, $\text{pIC}_{50} = 4{,}76$, `32` bits ativos): **$S_{\max} = 0{,}1905$**.
* **Menor similaridade máxima em $\text{hERG}$:**
  * `idx=8994` (`Drug_ID=12232`, clorofórmio `ClC(Cl)Cl`, `hERG_label = 0`, apenas `4` bits ativos): **$S_{\max} = 0{,}1000$**.
  * `idx=12108` (`Drug_ID=12696`, clordecona/kepone `O=C1C2(Cl)C3(Cl)C4(Cl)C(Cl)(Cl)C5(Cl)C3(Cl)C1(Cl)C5(Cl)C24Cl`, `hERG_label = 0`, `15` bits ativos): **$S_{\max} = 0{,}1250$**.
  * `idx=13111` (`Drug_ID=11585`, benzeno `c1ccccc1`, `hERG_label = 0`, apenas `3` bits ativos): **$S_{\max} = 0{,}1875$**.
* **Casos de $S_{\max} = 1{,}0000$ ($j \neq i$) e Próximos de Duplicação Estrutural:**
  * Em $\text{A}_{2\text{A}}$, `160` moléculas (`87` pares/grupos) possuem $S_{\max} = 1{,}0000$ com outra linha do dataset: `86` desses casos são **estereoisômeros ou pares racêmico/enantiômero** (mesmo primeiro bloco de `InChIKey`, como `CHEMBL5826102` [$\text{pIC}_{50}=10{,}00$] vs. `CHEMBL5834574` [$\text{pIC}_{50}=9{,}70$], ou `CHEMBL5951900` [$\text{pIC}_{50}=9{,}70$] vs. diastereoisômero `CHEMBL5766365` [$\text{pIC}_{50}=8{,}54$]), e `1` caso decorre de homologação de cadeia/anel que não cria novos ambientes circulares de raio 2.
  * Em $\text{hERG}$, `3.855` moléculas possuem $S_{\max} = 1{,}0000$ com outra linha do dataset, refletindo: (i) pares de enantiômeros/diastereoisômeros e racematos (`10.897` conectividades 2D únicas entre `13.136` SMILES isoméricos) e (ii) homólogos de anel/cadeia alifática saturada indistinguíveis em ECFP4 de raio 2 (ex.: `Drug_ID=11928` com anel azepano `C(CN1CCCCCC1)=...` vs. `Drug_ID=2872` com anel piperidina `C(CN1CCCCC1)=...`, ambos com os mesmos bits ECFP4 em 1.024 dimensões).

---

## 6.4 Resultados dos Cinco Compostos de Controle por Endpoint

Cada controle foi submetido a `canonicalize_smiles()` antes da extração dos 1.024 bits Morgan/ECFP4:

| Composto Controle | SMILES Canônico Padronizado (`onBits`) | Pertence ao Treino $\text{A}_{2\text{A}}$? | $S_{\max}$ vs. $\text{A}_{2\text{A}}$ (excl. auto) e Vizinho Mais Próximo | Pertence ao Treino $\text{hERG}$? | $S_{\max}$ vs. $\text{hERG}$ (excl. auto) e Vizinho Mais Próximo |
| :--- | :--- | :---: | :--- | :---: | :--- |
| **Istradefilina**<br>(`CHEMBL431770`) | `CCn1c(=O)c2c(nc(/C=C/c3ccc(OC)c(OC)c3)n2C)n(CC)c1=O`<br>(`46` bits) | **Sim**<br>(`idx=1315`, $\text{pIC}_{50\text{ exp}} = 5{,}28$) | **`0,8400`** *(Média Top-5: `0,6632`)*<br>Vizinho: `CHEMBL27508` (`idx=116`, análogo 1,3-dipropilxantina, $\text{pIC}_{50} = 9{,}00$) | **Não** | **`0,4000`** *(Média Top-5: `0,3544`)*<br>Vizinho: `Drug_ID=665` (`idx=4910`, `hERG_label = 1`) |
| **Cafeína**<br>(`CHEMBL113`) | `Cn1c(=O)c2c(ncn2C)n(C)c1=O`<br>(`24` bits) | **Sim**<br>(`idx=1347`, $\text{pIC}_{50\text{ exp}} = 4{,}85$) | **`0,3889`** *(Média Top-5: `0,2797`)*<br>Vizinho: `CHEMBL1355736` (`idx=1344`, teofilina `Cn1c(=O)c2nc[nH]c2n(C)c1=O`, $\text{pIC}_{50} = 4{,}92$) | **Não** | **`0,5152`** *(Média Top-5: `0,3243`)*<br>Vizinho: `Drug_ID=8191` (`idx=9350`, teobromina `Cn1cnc2c1c(=O)[nH]c(=O)n2C`, `hERG_label = 0`) |
| **Taurina**<br>(`Endógeno`) | `NCCS(=O)(=O)O`<br>(`14` bits) | **Não** | **`0,1167`** *(Média Top-5: `0,0975`)*<br>Vizinho: `CHEMBL5958542` (`idx=1000`, $\text{pIC}_{50} = 6{,}52$) | **Não** | **`0,1905`** *(Média Top-5: `0,1567`)*<br>Vizinho: `Drug_ID=1042` (`idx=2523`, `hERG_label = 1`) |
| **L-Teanina**<br>(`Nutracêutico`) | `CCNC(=O)CCC(N)C(=O)O`<br>(`25` bits) | **Não** | **`0,2027`** *(Média Top-5: `0,1946`)*<br>Vizinho: `CHEMBL5994559` (`idx=697`, $\text{pIC}_{50} = 7{,}12$) | **Não** | **`0,3261`** *(Média Top-5: `0,3254`)*<br>Vizinho: `Drug_ID=9537` (`idx=7655`, `hERG_label = 0`) |
| **Terfenadina**<br>(`CHEMBL22`) | `CC(C)(C)c1ccc(C(O)CCCN2CCC(C(O)(c3ccccc3)c3ccccc3)CC2)cc1`<br>(`43` bits) | **Não** | **`0,2273`** *(Média Top-5: `0,2232`)*<br>Vizinho: `CHEMBL270378` (`idx=645`, $\text{pIC}_{50} = 7{,}22$) | **Não por SMILES isomérico**, mas **Sim em conectividade 2D e vetor 1030-D** *(ver Seção 6.5)* | **`1,0000`** vs. `idx=682` (`Drug_ID=5703`, `(R)`-terfenadina, `hERG_label = 1`)<br>**`0,9111`** (excl. `idx=682`) vs. `idx=792` (`Drug_ID=2427`, hidroxi-terfenadina, `hERG_label = 1`) |

---

## 6.5 Anomalias Descobertas, Limitações Metodológicas e Implicações para o AD

### 6.5.1 Anomalias Estruturais e de Representação Identificadas na Auditoria
1. **Caso Crítico da Terfenadina no Dataset hERG (Estereoisomeria vs. Espaço de Features 2D Aquiral):**
   * O controle submetido para a Terfenadina (`CC(C)(C)c1ccc(C(O)CCCN2CCC(C(O)(c3ccccc3)c3ccccc3)CC2)cc1`) não especifica configuração absoluta no carbono carbinólico secundário (`C(O)`), razão pela qual não ocorre correspondência exata de string `canonical_smiles` em `herg_curated.csv`.
   * Contudo, a linha `idx=682` (`Drug_ID=5703`, `hERG_label = 1`, `num_records = 2`) de `herg_curated.csv` é o enantiômero **`(R)`-terfenadina** (`CC(C)(C)c1ccc([C@@H](O)CCCN2CCC(C(O)(c3ccccc3)c3ccccc3)CC2)cc1`), que possui exatamente o mesmo bloco de conectividade 2D (`GUGOEEXESWIERI`) e **exatamente o mesmo vetor de 1.030 atributos (`ΔX = 0`, `S_Tanimoto = 1.0000`)**, pois tanto os 6 descritores 2D quanto o Morgan fingerprint (`includeChirality=False`) são invariantes à quiralidade.
   * Além disso, os vizinhos subsequentes da Terfenadina em `herg_curated.csv` ilustram um clássico *activity cliff* metabólico: `idx=792` (`Drug_ID=2427`, $S = 0{,}9111$, `hERG_label = 1`) é um análogo hidroxilado bloqueador, enquanto `idx=12751` e `idx=12720` ($S = 0{,}8200$, `hERG_label = 0`) são os dois enantiômeros da **fexofenadina** (o metabólito carboxílico não-cardiotóxico da terfenadina).
2. **Tautômeros e Formas Mesoméricas Distintas em `herg_curated.csv`:**
   * Embora todas as `13.136` strings `canonical_smiles` em `herg_curated.csv` sejam únicas e idempotentes sob `canonicalize_smiles()`, existem **`182` pares/grupos (`368` linhas)** que compartilham o mesmo `InChIKey` padrão completo (`12.950` `InChIKey` únicos). Isso ocorre porque `standardize_mol()` executa dessalinização (`LargestFragmentChooser`) e sanitização, sem canonização de tautômeros (ex.: 2-hidroxipiridina `c3cccnc3O` na linha `9811` vs. 2-piridona `c3ccc[nH]c3=O` na linha `9814`) ou dipolos iônicos (`[P+]([O-])` vs. `P(=O)`).

### 6.5.2 Limitações do Indicador de Similaridade Máxima de Tanimoto ($S_{\max}$)
1. **Indicador Estrutural Parcial (Não Equivalente a Confiabilidade Preditiva):**
   * A similaridade máxima de Tanimoto em ECFP4 de 1.024 bits mede apenas a sobreposição de fragmentos circulares 2D de raio $\le 2$ contra o vizinho mais próximo do conjunto de treinamento. Ela **não** captura propriedades físico-químicas globais contínuas (`MolWt`, `MolLogP`, `TPSA`), efeitos conformacionais 3D, quiralidade (já que o fingerprint do pipeline é aquiral) ou densidade local da vizinhança (um único ponto isolado próximo difere de uma série congênere densa).
2. **Inexistência de Equivalência Automática entre Alto $S_{\max}$ e "Predição Confiável":**
   * Mesmo quando $S_{\max}$ é elevado, *activity cliffs* farmacológicos podem inverter a atividade biológica com pequenas modificações funcionais — como demonstrado no próprio conjunto hERG entre a Terfenadina (`hERG_label = 1`) e seu metabólito ácido Fexofenadina (`hERG_label = 0`), que compartilham $S_{\text{Tanimoto}} = 0{,}8200$, ou em $\text{A}_{2\text{A}}$ entre a Istradefilina (`CHEMBL431770`, ensaio funcional `F`, $\text{pIC}_{50} = 5{,}28$) e seu análogo propílico `CHEMBL27508` ($\text{pIC}_{50} = 9{,}00$, $S_{\text{Tanimoto}} = 0{,}8400$).
3. **Viés de Séries Congêneres nas Estatísticas Leave-One-Out do Conjunto de Treinamento:**
   * As distribuições Leave-One-Out de $\text{A}_{2\text{A}}$ (mediana $S_{\max} = 0{,}8219$) e $\text{hERG}$ (mediana $S_{\max} = 0{,}8308$) são fortemente influenciadas pela presença de séries químicas análogas e estereoisômeros originários das mesmas publicações/campanhas de SAR no ChEMBL e no TDC. Portanto, **as estatísticas internas do conjunto de treinamento não representam uma validação independente de moléculas externas** e não devem ser usadas isoladamente para impor cortes rígidos de aprovação/rejeição.
4. **Sensibilidade ao Número de Bits Ativos (`onBits`) em Moléculas Muito Pequenas:**
   * Moléculas de baixo peso molecular e poucos fragmentos circulares ativos — como a Taurina (`14` bits ativos, $S_{\max\text{ A2A}} = 0{,}1167$, $S_{\max\text{ hERG}} = 0{,}1905$) ou solventes/fragmentos mínimos presentes no próprio treino hERG como clorofórmio (`4` bits ativos, $S_{\max} = 0{,}1000$) e benzeno (`3` bits ativos, $S_{\max} = 0{,}1875$) — apresentam estruturalmente baixa interseção binária de Tanimoto com moléculas *drug-like* maiores (`mediana = 56–58` bits ativos).

---

# Parte VII — Fases 3D-3 e 3D-3A: Implementação Controlada do Motor de Domínio de Aplicabilidade (AD) e Verificação de Desempenho

## 7.1 Arquitetura e Engenharia de Recursos
Em cumprimento às recomendações da Fase 3D-1 e à especificação técnica da Fase 3D-2, foi implementado o motor de diagnóstico de cobertura estrutural e proveniência molecular no módulo [`src/ad_utils.py`](src/ad_utils.py), integrado à Seção 16 de [`app.py`](app.py):

1. **Estrutura de Referência em Memória (`EndpointADReference`):**
   - Carrega as coleções curadas completas ($1.364$ amostras para $\text{A}_{2\text{A}}$ e $13.136$ amostras para $\text{hERG}$).
   - Pré-instancia objetos `DataStructs.ExplicitBitVect` (1.024 bits) para execução ultrarrápida de `DataStructs.BulkTanimotoSimilarity`.
   - Indexa o vetor completo de 1.030 features via digest compacto de 16 bytes (`hashlib.blake2b`), permitindo checagem $O(1)$ de identidade informacional ($\Delta X = 0$).
2. **Hierarquia Estrita de 4 Níveis de Proveniência Molecular (Quatro Casos Implementados):**
   - **Nível 1 (Correspondência de Estrutura):** SMILES canônico padronizado idêntico a um registro de treinamento.
   - **Nível 2 (Correspondência de Features):** Vetor completo de 1.030 atributos idêntico ($\Delta X = 0$), apesar de divergência no SMILES (ex.: $(R)$-terfenadina vs. forma racêmica em representação 2D).
   - **Nível 3 (Fingerprint Idêntico):** Fingerprint idêntico (1.024 bits Morgan ECFP4, $S_{\text{Tanimoto}} = 1{,}0000$), mas descritores físico-químicos divergentes (colisão de bits).
   - **Nível 4 (Sem Correspondência Exata Conhecida):** Nenhuma correspondência exata conhecida, com apresentação contínua de similaridade máxima $S_{\max}$ e identificação do vizinho mais próximo, sem imposição de cortes arbitrários.
3. **Cálculo de Similaridade e Auto-Exclusão Seletiva:**
   - $S_{\max\text{ all}}$ afere a similaridade máxima contra todo o conjunto de treino.
   - $S_{\max\text{ excluding match}}$ exclui exclusivamente correspondências dos dois primeiros casos (Níveis 1 e 2), não do terceiro (Nível 3), preservando no Nível 3 o composto com fingerprint idêntico como vizinho mais próximo devido à divergência em suas propriedades físico-químicas.
   - Preservação integral do vizinho mais próximo com ID de origem e atividade experimental verificada.
4. **Tratamento Rigoroso de Falhas:**
   - Diferenciação explícita de `invalid_structure`, `standardization_error`, `missing_fingerprint`, `zero_on_bits` e `technical_error`.
   - Caso o indicador não possa ser calculado ou tenha bits zerados ($onBits = 0$), retorna `None`, **nunca atribuindo $S_{\max} = 0$ de forma arbitrária**.

## 7.2 Governança de Memória e Benchmark de Desempenho

| Métrica Avaliada | Alvo $\text{A}_{2\text{A}}$ ($1.364$ amostras) | Alvo $\text{hERG}$ ($13.136$ amostras) | Impacto / Metodologia |
| :--- | :---: | :---: | :--- |
| **Carga Inicial a Frio (Cold Load)** | **$112{,}12\text{ ms}$** | **$423{,}81\text{ ms}$** | Tempo de parsing do Parquet e instanciação dos EBVs. |
| **Latência de Consulta (Cache Aquecido, 100 reps)** | **Mediana: $4{,}24\text{ ms}$** (P95: $5{,}50\text{ ms}$) | **Mediana: $9{,}10\text{ ms}$** (P95: $10{,}97\text{ ms}$) | Execução vetorizada em C++ via `BulkTanimotoSimilarity`. |
| **Heap Python (`tracemalloc`)** | $\approx 1{,}05\text{ MB}$ | $\approx 7{,}27\text{ MB}$ | **Total: $\approx 8{,}32\text{ MB}$** de estruturas Python gerenciadas. |
| **Resident Set Size do Processo (`psutil`)** | $+24{,}27\text{ MB}$ | $+69{,}12\text{ MB}$ | **Total delta: $+93{,}39\text{ MB}$** (inclui buffers nativos C++ e PyArrow). |
| **Estabilidade de Memória Pós-100 Consultas** | Estável | Estável | Variação de $-2{,}18\text{ MB}$ pós-queries. Não foi observado crescimento persistente de RSS após 100 consultas no protocolo experimental executado (comportamento sob concorrência multiusuário contínua requer monitoramento em produção). |

## 7.3 Comprovação de Imutabilidade Byte-a-Byte (Hashes SHA-256)

A tabela abaixo comprova a integridade criptográfica byte-a-byte dos arquivos serializados de modelos e conjuntos tabulares processados listados, cujos hashes SHA-256 permanecem estritamente idênticos aos registrados nas auditorias anteriores (a integridade do código funcional e das regras de negócio é verificada de forma independente por inspeção de diff e testes automatizados):

| Arquivo Protegido | Hash SHA-256 de Referência | Status de Integridade |
| :--- | :--- | :---: |
| `models/a2a_model.joblib` | `1907c7241edf64ffae3aa5c59a497037d427b1ead6b83abdf15abbfe3b2a4787` | **Inalterado** |
| `models/herg_model.joblib` | `fcedbecb89843870fe1b7da9c501eb62b8e9e6a0a254a9ed8139c17d6ad154e6` | **Inalterado** |
| `data/processed/a2a_curated.csv` | `88354901075bc9058b6ae6b5ad8920a4a710f10b2efd32abd8ab02541b977a3e` | **Inalterado** |
| `data/processed/herg_curated.csv` | `df7be71479e64c05f735b17b35a6fd888cc6ca869615e9bde50f489c715b3798` | **Inalterado** |
| `data/processed/a2a_featurized.parquet` | `ac006abb90b59ac74c6d3a14cad25ceee62c58cee44fa263a8a69f90928b31b7` | **Inalterado** |
| `data/processed/herg_featurized.parquet` | `612e115eba6e47788adb92a2edd067e89d5af9f5095dba0dfcad6fc817d7eb92` | **Inalterado** |
| `data/processed/a2a_featurized.csv` | `857b1c083df46f88750decbf43085ce4854eca8204b0aef83744ffbf81c43575` | **Inalterado** |
