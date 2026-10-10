# PRD v3

# Product Requirements Document (PRD)

## 1. Visão Geral e Problema

- **Nome do Produto:** MultiTarget BioPredict (Preditor Multiobjetivo de Afinidade $A_{2A}$ e Segurança Cardíaca hERG).
- **Contexto:** No desenvolvimento inicial de novos candidatos a bioativos e fármacos, a síntese laboratorial e os ensaios biológicos *in vitro* demandam semanas de bancada e alto custo financeiro. Além de buscar eficácia em um alvo terapêutico (como o receptor de adenosina $A_{2A}$, responsável pela modulação do cansaço e do foco), a indústria farmacêutica enfrenta falhas clínicas tardias decorrentes de toxicidade silenciosa — em especial a arritmia cardíaca induzida pelo bloqueio do canal de potássio hERG (*prolongamento do intervalo QT*).
- **Proposta de Valor:** Uma plataforma preditiva que realiza triagem virtual multiobjetivo em segundos a partir de representações químicas 2D, avaliando simultaneamente o potencial de eficácia/ancoragem ($A_{2A}$) e o risco toxicológico cardíaco (hERG).

## 2. Personas e Casos de Uso

### Personas Primárias

- **Cientista de Dados / Avaliador Acadêmico:** Avaliar a robustez estatística ($R^2$, RMSE, ROC-AUC), comparar o desempenho entre arquiteturas e auditar o impacto dos descritores via interpretabilidade (SHAP).
- **Químico Medicinal / Formulador:** Desenhar ou submeter moléculas (via SMILES) para obter triagem virtual instantânea de potência inibitória ($A_{2A}$) e indicativo prévio de cardiotoxicidade (hERG) antes do investimento em síntese.
- **Gestor de P&D / Analista de Negócios:** Priorizar candidatos a fármacos com alto potencial de eficácia e baixo risco regulatório/toxicológico, reduzindo o custo financeiro decorrente de reprovações na fase pré-clínica.

### Casos de Uso Centrais

- **UC-01 (Predição em Tempo Real):** O usuário desenha a molécula ou cola o SMILES de um composto conhecido (ex.: Cafeína, Istradefilina) ou inédito e recebe a estimativa quantitativa de $pIC_{50}$ para $A_{2A}$ e a probabilidade de risco cardíaco (hERG).
- **UC-02 (Auditoria e Interpretabilidade Otimizada):** O usuário consulta a decomposição dos fragmentos químicos e propriedades físico-químicas de maior impacto que influenciaram a predição para entender os grupos funcionais responsáveis pelo efeito.
- **UC-03 (Verificação de Domínio de Aplicabilidade):** O sistema avisa o usuário se a molécula submetida for excessivamente divergente da distribuição química vista no treinamento.

## 3. Escopo e Requisitos Funcionais

### RF-01: Ingestão e Processamento Químico

- A aplicação deve aceitar entrada via código `SMILES` estruturado ou desenho gráfico molecular interativo utilizando o componente integrado **`streamlit-ketcher`**.
- O backend deve higienizar o SMILES via RDKit (conversão de MolBlock/SMILES, remoção de sais, neutralização de cargas quando aplicável e geração de SMILES canônico).
- O pipeline deve calcular automaticamente os vetores de atributos:
    - *Morgan Fingerprints* (ECFP4 com raio 2, 1024 bits) forçando tipagem explícita em matrizes `numpy.uint8` (redução de 8x no consumo de RAM frente a matrizes float).
    - Descritores físico-químicos 2D fundamentais: MolWt, MolLogP, TPSA, NumHDonors, NumHAcceptors e NumRotatableBonds.

### RF-02: Pipeline de Modelagem Preditiva Local

| **Componente** | **Modelo 1: Eficácia (**$A_{2A}$**)** | **Modelo 2: Cardiotoxicidade (hERG)** |
| --- | --- | --- |
| **Tipo de Tarefa** | Regressão Contínua | Classificação Binária com Saída Probabilística Calibrada |
| **Variável Alvo** | Potência de inibição/ligação ($pIC_{50} = -\log_{10}(IC_{50} \times 10^{-9})$) | Probabilidade contínua $[0, 1]$ de ser um bloqueador hERG ($P_{\text{hERG}}$) |
| **Fonte dos Dados** | ChEMBL (`CHEMBL251` - Adenosine receptor $A_{2A}$) com $IC_{50}$ e relação `=` | Repositório Therapeutics Data Commons (TDC) via API do Harvard Dataverse **(excluída a biblioteca `PyTDC` para evitar bloat de dependências incompatíveis no Windows)**. |
| **Algoritmos Avaliados** | Ridge, Random Forest, XGBoost e MLP Regressor | Logistic Regression, Random Forest, XGBoost e MLP Classifier |
| **Métricas de Aceite** | $R^2 \ge 0{,}65$ e $\text{RMSE} \le 0{,}80$ | $\text{ROC-AUC} \ge 0{,}80$ e $\text{F1-Score} \ge 0{,}75$ |

### RF-03: Camada de Explicabilidade Otimizada (XAI)

- O sistema deve calcular valores de SHAP (*TreeExplainer* para os modelos de ensemble) permitindo visualizar o impacto dos principais bits e propriedades na predição final de $A_{2A}$ e hERG.
- **Restrição de Hardware:** Para otimização de memória e renderização fluida, a geração gráfica do SHAP deve ser restrita estritamente às **Top 10 propriedades/bits de maior impacto** na inferência atual.

### RF-04: Interface de Decisão e Semáforos Independentes

A interface gráfica exibirá dois indicadores visuais autônomos, permitindo ao usuário analisar a relação risco-benefício sem filtros combinados opacos:

#### 1. Indicador de Eficácia ($A_{2A}$)

Baseado exclusivamente na potência de inibição/ligação estimada pelo modelo de regressão ($pIC_{50}$):

- 🟢 **Alta Potência:** $pIC_{50} \ge 6{,}5$ (Composto altamente promissor para o alvo terapêutico).
- 🟡 **Potência Moderada:** $5{,}0 \le pIC_{50} < 6{,}5$ (Potência intermediária; pode exigir otimização de fragmentos).
- 🔴 **Baixa/Nenhuma Potência:** $pIC_{50} < 5{,}0$ (Composto considerado inativo ou fracamente ativo para o alvo).

#### 2. Indicador de Segurança Cardíaca (hERG)

Baseado exclusivamente na probabilidade contínua calculada pelo modelo de classificação ($P_{\text{hERG}}$):

- 🟢 **Risco Baixo (Seguro):** $P_{\text{hERG}} < 30\%$ (Baixa probabilidade de bloqueio do canal de potássio).
- 🟡 **Risco Moderado (Atenção):** $30\% \le P_{\text{hERG}} \le 70\%$ (Zona cinzenta; o composto requer monitoramento de toxicidade estrutural).
- 🔴 **Risco Alto (Cardiotóxico):** $P_{\text{hERG}} > 70\%$ (Alta probabilidade de indução de prolongamento do intervalo QT / arritmia ventricular).

*Nota de Demonstração:* Os botões de atalho pré-configurados (Cafeína, Istradefilina, Terfenadina e Taurina) servirão para demonstrar e validar o comportamento independente de ambos os medidores na interface.

## 4. Requisitos Não Funcionais

- **Arquitetura 100% Local (Antigravity):** Todo o ciclo de vida do projeto — ingestão de dados, pré-processamento químico, treino supervisionado, validação cruzada 5-fold, serialização e inferência com Streamlit — deve ser realizado estritamente no ambiente local gerenciado pelo Antigravity, sem uso de Google Colab ou serviços em nuvem.
- **Orçamento de Recursos e Memória:** O consumo de pico de memória RAM durante o treinamento não deve ultrapassar 5 GB, e durante a inferência da aplicação deve se manter abaixo de 2 GB, operando com total fluidez em computadores com 8 GB de RAM e processador Ryzen 7. Para garantir esse teto, processos de Validação Cruzada não devem exceder `n_jobs=2`.
- **Tempo de Resposta de Inferência:** Inferência preditiva completa (cálculo de descritores + predição dos 2 modelos + visualização SHAP Top 10) inferior a 3 segundos por molécula na interface local.
- **Serialização de Modelos:** Os artefatos finais treinados localmente devem ser serializados em arquivos binários compactados `.joblib` (`compress=3`), garantindo baixo consumo de disco e carregamento instantâneo no aplicativo.
- **Reprodutibilidade Científica:** Fixação de sementes aleatórias (`random_state=42`) para todas as etapas estocásticas de treinamento local.
- **Portabilidade:** Gerenciamento de dependências e ambiente virtual unificado com versões travadas em `requirements.txt` (`rdkit`, `streamlit-ketcher`, `scikit-learn`, `xgboost`, `shap`, `joblib` e `streamlit`).
- **Consistência de Licenciamento:** Utilização restrita de bases públicas de acesso aberto sob termos acadêmicos e não comerciais (ChEMBL / TDC / PubChem).

## 5. Arquitetura de Dados e Pipeline Local

```
       [ AMBIENTE LOCAL (Antigravity - Workspace Unificado) ]
       Bases Locais (ChEMBL / TDC) ──► Pipeline RDKit (Local)
                                              │
                                              ▼
                              Treinamento e Validação 5-Fold
                               (A2A Regressão + hERG Classif.)
                                              │
                                              ▼
                               Artefatos Serializados (.joblib)
                                              │
                                              ▼
       [ CAMADA DE APLICAÇÃO LOCAL (Streamlit - <= 8 GB RAM) ]
                   Carregamento dos Arquivos .joblib
                                  │
        Entrada do Usuário via [ streamlit-ketcher ] ou SMILES
                                  │
                                  ▼
                    [ Módulo RDKit (Python) ]
               Higienização + Vetorização Numérica
             (Morgan Fingerprint 1024b + Descritores)
                                  │
                 ┌────────────────┴────────────────┐
                 ▼                                 ▼
        [ Modelo 1: A2A ]                 [ Modelo 2: hERG ]
        (XGBoost .joblib)                 (R. Forest .joblib)
                 │                                 │
               pIC50                       Prob. Risco [0, 1]
                 └────────────────┬────────────────┘
                                  ▼
                     [ Motor de Decisão Nuanceado ]
                      Cálculo SHAP (Top 10 Features)
                                  │
                                  ▼
                    [ Saída Visual no Streamlit ]
                  (2 Semáforos + Gráficos XAI Top 10)
```

## 6. Critérios de Sucesso e Demonstração

- **Acurácia Preditiva e Generalização Local:** Validação cruzada estratificada em 5 folds executada com sucesso no ambiente Antigravity, comprovando ausência de vazamento de dados (*data leakage*) e cumprimento das métricas estipuladas na Seção 3.
- **Estabilidade e Eficiência no Antigravity:** Execução completa do pipeline de treino e inicialização da interface Streamlit local sem registrar estouro de memória (*Out of Memory*) no perfil de máquina de 8 GB de RAM.
- **Validação dos Controles Farmacológicos (Smoke Tests):**
    - Identificação correta da **Istradefilina** como composto ativo ($A_{2A}$) e cardioprotetor (Risco Baixo).
    - Identificação correta da **Terfenadina** como composto cardiotóxico de Alto Risco (Semáforo Vermelho automático).
    - Identificação da **Taurina** e **L-Teanina** como compostos biologicamente inativos para o receptor $A_{2A}$ e seguros para o canal hERG.
- **Entregável Técnico:** Repositório local unificado contendo os scripts de extração/treinamento documentados, os arquivos compactados `.joblib` e o script `app.py` executável via comando local no Antigravity (`streamlit run app.py`).