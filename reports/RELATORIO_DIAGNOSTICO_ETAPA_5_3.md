# RELATÓRIO DE DIAGNÓSTICO TÉCNICO — ETAPA 5.3.0
## Comparação Multimolecular — Projeto MultiTarget BioPredict

**Projeto:** MultiTarget BioPredict  
**Fase:** Diagnóstico Técnico da Subetapa 5.3 (Comparação Multimolecular)  
**Data:** 31 de Outubro de 2026  
**Branch Git:** `feat/ajustes-funcionalidades`  
**Natureza da Atividade:** Estritamente diagnóstica e de planejamento (sem alteração de código ou dados)  
**Status dos 8 Hashes Protegidos:** 100% Verificados e Intactos  

---

## 1. ESTRUTURA REAL DOS DADOS DISPONÍVEIS NO HISTÓRICO

A auditoria da implementação da Etapa 5.2 confirmou que `st.session_state["analysis_history"]` armazena uma lista ordenada (FIFO) com até 10 dicionários de análise independentes. A inspeção do código-fonte em [src/ui/history_manager.py](file:///c:/Users/Raphael/Documents/BioPredicta/src/ui/history_manager.py) e [src/ui/pages/p2_nova_analise.py](file:///c:/Users/Raphael/Documents/BioPredicta/src/ui/pages/p2_nova_analise.py) revelou o seguinte esquema de dados para cada registro:

```python
record = {
    # 1. Identificação Estrutural
    "canonical_smiles": str,        # Chave primária única (SMILES canônico Daylight)
    "timestamp": str,               # Carimbo ISO 8601 da inferência
    "formula": str,                 # Fórmula bruta (ex: "C8H10N4O2")
    "mol_wt": float,                # Massa molecular exata (Da)
    "mol": None,                    # Omitido em repouso para economizar heap

    # 2. Descritores Físico-Químicos 2D (RDKit)
    "descriptors": {
        "MolWt": float,             # Peso molecular (g/mol)
        "MolLogP": float,           # Coeficiente de partição octanol/água (Wildman-Crippen)
        "NumHDonors": int,          # Doadores de ligação de H (Lipinski)
        "NumHAcceptors": int,       # Aceptores de ligação de H (Lipinski)
        "TPSA": float,              # Área de superfície polar topológica (Å²)
        "NumRotatableBonds": int,   # Ligações simples rotacionáveis (Veber)
    },

    # 3. Proveniência e Catálogos Externos
    "provenance": {
        "compound_name": str,       # Nome usual / ChEMBL ID
        "chembl_id": str,           # Identificador ChEMBL (quando disponível)
        "is_in_a2a_curated": bool,  # Status no dataset de calibração A2A
        "is_in_herg_curated": bool, # Status no dataset de calibração hERG
    },
    "pubchem": {
        "cid": int,                 # Identificador PubChem CID
        "title": str,               # Nome comercial / denominação comum oficial
        "iupac_name": str,          # Nomenclatura sistemática IUPAC
    },

    # 4. Predição Farmacológica A2A (XGBoost Regressor)
    "a2a": {
        "pred_pIC50": float,        # Potência predita contínua (-log10 IC50 M)
        "pred_IC50_nM": float,      # Concentração inibitória molar estimada (10^(9 - pIC50))
        "classification": {
            "label": str,           # "Alta Potência", "Potência Moderada" ou "Baixa Potência"
            "color": str,           # Token cromático do Design System
        },
        "ad": {                     # Diagnóstico de Domínio de Aplicabilidade A2A
            "status": str,          # "ok" ou "technical_error"
            "provenance_level": int,# Nível 1 a 4
            "provenance_desc": str, # Descrição textual do nível
            "s_max_all": float,     # Similaridade Tanimoto máxima geral
            "s_max_excluding_match": float, # Similaridade máxima excluindo auto-match
            "nearest_neighbor": dict,       # ID e similaridade do composto mais próximo
            "loo_median_similarity": float, # Limiar de referência LOO (0.8219)
        },
        "shap_values": np.ndarray,  # Tensor 1030-D de contribuições locais
        "base_value": float,        # Valor esperado de linha de base (5.48)
    },

    # 5. Predição Antitóxica hERG (Random Forest Classifier)
    "herg": {
        "prob_herg": float,         # Escore de saída do classificador (fração classe 1)
        "safety_index": float,      # Índice complementar derivado (1.0 - prob_herg)
        "complementary_index": float,
        "classification": {
            "label": str,           # "Baixo Risco", "Risco Moderado" ou "Alto Risco"
            "color": str,           # Token cromático
        },
        "ad": {                     # Diagnóstico de Domínio de Aplicabilidade hERG (Independente)
            "status": str,
            "provenance_level": int,
            "s_max_all": float,
            "nearest_neighbor": dict,
            "loo_median_similarity": float,
        },
        "shap_values": np.ndarray,  # Tensor 1030-D para classe positiva
        "base_value": float,        # Linha de base da classe 1 (0.50)
    },

    # 6. Posicionamento Cartesiano Multiobjetivo
    "multiobjective": {
        "quadrant": {
            "quad_title": str,      # Ex: "Quadrante 1 — Zona Ideal"
            "quad_badge_cls": str,  # "semaphore-green", "semaphore-red", etc.
        },
        "x": float,                 # Coordenada X (pIC50 A2A)
        "y": float,                 # Coordenada Y (1 - P_hERG)
    },

    # 7. Vetor de Features Completo
    "features": pd.DataFrame,       # 1 linha x 1030 atributos (6 físico-químicos + 1024 bits)
}
```

### Recuperação Não-Destrutiva dos Registros
- **Acesso Seguro:** A função `get_history()` retorna os registros em sessão sem interagir com `st.session_state["current_analysis"]`.
- **Isolamento de Seleção:** As moléculas escolhidas para comparação são recuperadas por filtragem simples (`[rec for rec in history if rec["canonical_smiles"] in selected_smiles]`), garantindo que a análise ativa permaneça intacta.
- **Reconstrução RDKit On-Demand:** A estrutura molecular nativa é instanciada sob demanda via `smiles_to_mol(rec["canonical_smiles"])`, dispensando a retenção de ponteiros pesados em repouso.

---

## 2. VIABILIDADE TÉCNICA DAS COMPARAÇÕES

### A. Comparação Farmacológica e Multiobjetivo
- **Viabilidade:** **100% Viável.**
- **Métricas:**
  - $A_{2A}$: $pIC_{50}$ contínuo, $IC_{50}$ formatado (nM/µM via `format_ic50`), classe canônica (Baixa $< 5.0$, Moderada $5.0 \le pIC_{50} < 6.5$, Alta $\ge 6.5$).
  - hERG: Escore do classificador $P(\text{hERG}=1)$, índice complementar $1 - P$, classe canônica (Baixo $P < 0.30$, Moderado $0.30 \le P \le 0.70$, Alto $P > 0.70$).
  - Quadrante Multiobjetivo: Quadrante 1 (Zona Ideal), 2 (Baixo Interesse), 3 (Descarte) ou 4 (Alerta Crítico).
- **Conformidade Científica:** Os limiares de corte ($5.0$, $6.5$, $0.30$, $0.70$) são rigorosamente idênticos aos definidos nas Etapas 2 e 3. Nenhuma métrica sintética ou score ponderado arbitrário é gerado.

### B. Domínio de Aplicabilidade (AD) Desacoplado
- **Viabilidade:** **100% Viável.**
- **Apresentação Independente:**
  - Coluna/painel $A_{2A}$ AD: Nível de proveniência (1 a 4), $S_{\max}$ e vizinho mais próximo no dataset de 1.364 compostos.
  - Coluna/painel hERG AD: Nível de proveniência (1 a 4), $S_{\max}$ e vizinho mais próximo no dataset de 13.136 compostos.
- **Rigor Metodológico:** Ambas as dimensões são expostas lado a lado sem mesclagem ou médias que possam mascarar extrapolações parciais.

### C. Comparação Estrutural e Matriz de Tanimoto (ECFP4)
- **Viabilidade:** **100% Viável e de Altíssimo Valor Científico.**
- **Implementação:**
  - A partir dos SMILES canônicos, reconstrói-se o objeto `Chem.Mol` com `smiles_to_mol()`.
  - Gera-se o fingerprint circular Morgan via `AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)`.
  - Calcula-se a matriz de similaridade $N \times N$ ($2 \le N \le 5$) via `DataStructs.TanimotoSimilarity()`.
- **Desempenho:** A computação para até 5 moléculas requer no máximo 10 comparações par-a-par ($O(N^2)$), executando em menos de 2 milissegundos sem sobrecarga no servidor.
- **Visualização:** Tabela simétrica colorida em degradê de ciano (`#00D2BE`) indicando proximidade morfológica direta entre os candidatos.

### D. Comparação Físico-Química e Cálculo de Deltas ($\Delta$)
- **Viabilidade:** **100% Viável.**
- **Propriedades Avaliadas:** Massa Molecular ($MolWt$), Lipofilicidade ($MolLogP$), Doadores de H ($NumHDonors$), Aceptores de H ($NumHAcceptors$), Superfície Polar ($TPSA$) e Ligações Rotacionáveis ($NumRotatableBonds$).
- **Modo Relativo (Deltas Frente à Referência):**
  - O usuário seleciona um composto como "Molécula de Referência" (ex.: Istradefilina ou a molécula ativa).
  - Para cada outra molécula, calcula-se:
    $$\Delta pIC_{50} = pIC_{50}^{(i)} - pIC_{50}^{(\text{ref})}$$
    $$\Delta P_{\text{hERG}} = P_{\text{hERG}}^{(i)} - P_{\text{hERG}}^{(\text{ref})}$$
    $$\Delta MolWt = MolWt^{(i)} - MolWt^{(\text{ref})}$$
    $$\Delta LogP = LogP^{(i)} - LogP^{(\text{ref})}$$
  - A interface realça ganhos de potência ($\Delta pIC_{50} > 0$ em verde) e variações no escore hERG.

### E. Gráfico Cartesiano Multiobjetivo Multiponto
- **Viabilidade:** **100% Viável.**
- **Análise do Componente Atual ([src/ui/components.py#L91-L170](file:///c:/Users/Raphael/Documents/BioPredicta/src/ui/components.py#L91-L170)):**
  - A função original `build_multiobjective_figure(x_cur, y_cur, active_name)` deve ser estritamente preservada para não quebrar `p3_resultados.py` nem os testes regressivos AST.
- **Componente Complementar Proposto:** `build_comparative_multiobjective_figure(molecules_list, ref_smiles=None)`:
  - Mantém idêntica geometria: eixos ($X \in [3.5, 11.5]$, $Y \in [-0.05, 1.05]$), linhas limiares (5.0, 6.5, 0.30, 0.70) e sombreamento dos 4 quadrantes.
  - Plota múltiplos marcadores cromáticos diferenciados (paleta Dark Mode: Ciano `#00D2BE`, Âmbar `#F59E0B`, Púrpura `#A855F7`, Azul Céu `#38BDF8`, Rosa `#EC4899`).
  - Formas geométricas e índices numéricos (`#1`, `#2`, `#3`) para garantir acessibilidade a daltônicos.
  - Rótulos com deslocamento inteligente e legenda padronizada evitando sobreposições de texto.

---

## 3. PROPOSTA DE INTERFACE E ORGANIZAÇÃO VISUAL

A experiência da página [src/ui/pages/p5_comparacao.py](file:///c:/Users/Raphael/Documents/BioPredicta/src/ui/pages/p5_comparacao.py) será estruturada em blocos verticais lógicos:

```
┌────────────────────────────────────────────────────────────────────────┐
│  ⚖️ COMPARAÇÃO MULTIMOLECULAR                                          │
│  Avaliação visual, quantitativa e estrutural de 2 a 5 candidatos       │
├────────────────────────────────────────────────────────────────────────┤
│ 1. PAINEL DE CONTROLE E SELEÇÃO                                        │
│    • Seletor Múltiplo (2 a 5 moléculas do histórico)                   │
│    • Seletor da "Molécula de Referência" (para cálculo de deltas)      │
│    • Botão de Ação Rápida: "Carregar Controles" caso histórico < 2    │
├────────────────────────────────────────────────────────────────────────┤
│ 2. POSICIONAMENTO MULTIOBJETIVO CONJUNTO (Matriz Cartesiana 2D)        │
│    • Gráfico comparativo multiponto Matplotlib (Dark Theme)            │
│    • Dispersão nos 4 quadrantes operacionais com marcadores coloridos  │
│    • Destaque visual da Molécula de Referência                         │
├────────────────────────────────────────────────────────────────────────┤
│ 3. QUADRO SINÓPTICO FARMACOLÓGICO E AD (Grade Lado a Lado)             │
│    ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ │
│    │ Molécula #1  │ │ Molécula #2  │ │ Molécula #3  │ │ ...          │ │
│    │ Miniatura 2D │ │ Miniatura 2D │ │ Miniatura 2D │ │              │ │
│    │ A2A: 7.85 🟢 │ │ A2A: 5.25 🟡 │ │ A2A: 8.10 🟢 │ │              │ │
│    │ hERG: 0.18 🟢│ │ hERG: 0.13 🟢│ │ hERG: 0.75 🔴│ │              │ │
│    │ Q1: Ideal    │ │ Q2: Modesto  │ │ Q4: Crítico  │ │              │ │
│    │ AD: Nível 1  │ │ AD: Nível 1  │ │ AD: Nível 3  │ │              │ │
│    └──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘ │
├────────────────────────────────────────────────────────────────────────┤
│ 4. MATRIZ DE SIMILARIDADE ESTRUTURAL (Tanimoto ECFP4)                  │
│    • Tabela de distâncias químicas pareadas entre as selecionadas      │
├────────────────────────────────────────────────────────────────────────┤
│ 5. PERFIL FÍSICO-QUÍMICO COMPARATIVO E TABELA DE DELTAS (Δ)           │
│    • Tabela comparativa dos 6 descritores RDKit vs. Regras de Lipinski │
│    • Diferenças quantitativas (Δ pIC50, Δ Escore hERG, Δ MolWt, etc.)  │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 4. MAPEAMENTO DE ARQUIVOS E FUNÇÕES ENVOLVIDOS

| Arquivo | Natureza | Responsabilidade Técnica |
| :--- | :---: | :--- |
| `src/ui/components.py` | **Extensão** | Adição de `build_comparative_multiobjective_figure(molecules_list, ref_smiles=None)` sem alterar a existente `build_multiobjective_figure()`. |
| `src/ui/pages/p5_comparacao.py` | **Novo** | Módulo principal de renderização comparativa: seleção de 2 a 5 compostos, tabela sinóptica, matriz de Tanimoto e tabela de deltas. |
| `src/ui/history_manager.py` | **Reutilização** | Consumo seguro de `get_history()`, `get_history_summary_list()` e `set_active_molecule()`. |
| `app.py` | **Integração de Rota** | Registro de `st.Page(page_comparacao, title="Comparação", icon="⚖️", url_path="comparacao")` em `pages_navigation["ANÁLISE"]`. |
| `tests/test_ui_p5_comparacao.py` | **Novo** | Suíte de testes automatizados para a experiência multimolecular. |

---

## 5. ANÁLISE DE RISCOS TÉCNICOS E CIENTÍFICOS

| Risco Mapeado | Natureza | Gravidade | Estratégia de Mitigação |
| :--- | :---: | :---: | :--- |
| **Histórico com Menos de 2 Moléculas** | Usabilidade | Média | Implementar *Empty State* didático: quando `len(history) < 2`, a página exibe aviso instrutivo com botão de 1-clique para avaliar controles de referência (ex: Cafeína + Istradefilina). |
| **Sobreposição Gráfica de Rótulos no Scatter 2D** | Gráfica | Média | Se dois compostos possuírem coordenadas muito próximas (ex.: $\Delta x < 0.2$ e $\Delta y < 0.05$), aplicar offset direcional automático alternado nos rótulos e recorrer à legenda lateral. |
| **Comparação Ingênua de Predições em Níveis Distintos de AD** | Científica | **Alta** | Nunca apresentar tabelas ordenadas puramente por $pIC_{50}$ sem o selo de AD em evidência. Inserir nota metodológica advertindo que predições em Nível 4 (Extrapolação) têm menor confiabilidade comparativa. |
| **Confusão com a Molécula Ativa no Painel** | UX | Baixa | A seleção temporária para comparação na tela *Comparação* **não altera** `st.session_state["current_analysis"]`. Adicionar botão opcional *"Tornar Ativa"* caso o usuário deseje carregar um dos candidatos comparados nas abas de Interpretação. |
| **Degradação de Memória por Geração Excessiva de Figuras** | Técnica | Baixa | Fechar figuras Matplotlib explicitamente com `plt.close(fig)` após renderização no Streamlit. |

---

## 6. ESTRATÉGIA DE TESTES AUTOMATIZADOS

A implementação será acompanhada pelo arquivo `tests/test_ui_p5_comparacao.py`, cobrindo:
1. **Estado Vazio e Limites de Seleção:**
   - Comportamento nominal com 0 ou 1 molécula no histórico.
   - Validação de seleção restrita a $[2, 5]$ moléculas.
2. **Cálculo da Matriz de Tanimoto:**
   - Determinismo, simetria ($M = M^T$) e diagonal unitária ($M_{ii} = 1.0$).
3. **Cálculo de Deltas Farmacológicos e Físico-Químicos:**
   - Conferência de $\Delta pIC_{50}$, $\Delta P_{\text{hERG}}$ e descritores em relação à molécula de referência.
4. **Renderização do Gráfico Multiponto:**
   - Geração sem erros de Matplotlib para 2, 3, 4 e 5 pontos.
   - Verificação da preservação exata dos limiares canônicos (5.0, 6.5, 0.30, 0.70).
5. **Não-Regressão e Invariância:**
   - Execução integral da suíte de 115 testes funcionais pré-existentes.
   - Verificação criptográfica de 100% dos 8 hashes SHA-256.

---

## 7. PLANO DE IMPLEMENTAÇÃO EM INCREMENTOS PEQUENOS

Recomenda-se dividir a execução da Etapa 5.3 nos seguintes passos graduais:

- **Passo 1 (Componentes Químicos e Gráficos):**
  - Implementar em `src/ui/components.py` a função `build_comparative_multiobjective_figure()`.
  - Implementar funções auxiliares de cálculo de matriz Tanimoto e deltas.
  - Testes unitários focados na matemática do gráfico e dos descritores.
- **Passo 2 (Página `p5_comparacao.py` e Rota `app.py`):**
  - Construir a interface completa de `p5_comparacao.py` (seleção, gráfico, grade e tabelas).
  - Integrar a rota no menu "ANÁLISE" de `app.py`.
- **Passo 3 (Validação e Homologação):**
  - Execução da suíte completa de testes (`pytest`).
  - Verificação de hashes protegidos e elaboração do relatório de encerramento da Etapa 5.3.

---

### CONCLUSÃO DO DIAGNÓSTICO
O diagnóstico atesta que a base de dados em sessão da Etapa 5.2 contém todos os atributos necessários para viabilizar a comparação multimolecular com extremo rigor científico, sem necessidade de recalcular inferências ou onerar o ambiente local.

*Nenhum arquivo de código foi modificado. A execução está interrompida aguardando suas diretrizes e aprovação deste planejamento para iniciarmos a implementação.*

