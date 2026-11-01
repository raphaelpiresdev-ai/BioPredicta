# RELATÓRIO TÉCNICO — FASE 5.1 (PASSO 1.1)
## Revisão Corretiva do Tutorial de Utilização da Plataforma

**Projeto:** MultiTarget BioPredict  
**Fase:** Fase 5.1 — Passo 1.1: Revisão Corretiva do Tutorial de Utilização  
**Data:** 31 de Outubro de 2026  
**Branch Git:** `feat/ajustes-funcionalidades`  
**Status da Suíte de Testes:** 159 testes aprovados (100% de sucesso, 0 falhas, 1 desmarcado/benchmark)  
**Status dos Hashes Protegidos:** 8/8 íntegros e verificados criptograficamente via SHA-256  

---

## 1. RESUMO EXECUTIVO

O presente relatório consolida a execução da **Fase 5.1 — Passo 1.1**, referente à **revisão corretiva pontual do Tutorial de Utilização da Plataforma** ([`src/ui/pages/p6_como_utilizar.py`](file:///c:/Users/Raphael/Documents/BioPredicta/src/ui/pages/p6_como_utilizar.py)).

A revisão foi conduzida com estrita observância ao princípio de intervenção cirúrgica e não-destrutiva:
1. **Preservação Arquitetural:** Nenhuma rota, função de renderização ou componente do Design System foi refatorado ou removido.
2. **Estrutura em Três Abas:** Foram mantidas as abas **🚀 Início Rápido (3 Minutos)**, **📖 Manual Completo (7 Módulos)** e **🛡️ Diretrizes e Boas Práticas**.
3. **Escopo Delimitado:** O trabalho restringiu-se exclusivamente aos 6 pontos corretivos solicitados, sem avançar para outras páginas planejadas da Central de Ajuda (`fundamentos`, `governanca` ou `sobre`).

---

## 2. DETALHAMENTO DAS CORREÇÕES IMPLEMENTADAS

### Correção 1 — Abordagem Híbrida de Conteúdo nos 7 Módulos

- **Diagnóstico:** A versão anterior do tutorial apresentava todo o detalhamento técnico e quantitativo em bloco único dentro dos módulos, sobrecarregando o usuário iniciante que busca apenas compreender o fluxo operacional.
- **Implementação:** Foi implementada uma **arquitetura de apresentação em duas camadas** em cada um dos 7 módulos:
  1. **Camada Essencial Imediata (`#### Apresentação Essencial`):** Explicação direta, acessível e didática sobre a utilidade prática do recurso, passos operacionais e significados dos botões e saídas.
  2. **Camada de Aprofundamento Técnico (`st.expander("🔬 Aprofundamento Técnico...", expanded=False)`):** Detalhes matemáticos, parâmetros de hiperconfiguração dos modelos (XGBoost e Random Forest), tratamento vetorial RDKit, formulação aditiva local de SHAP e equacionamento cartesiano.
- **Resultado:** Acessibilidade didática sem qualquer perda de profundidade científica para usuários especialistas.

---

### Correção 2 — Tabela Única Visual Consolidada de Limiares Operacionais

- **Diagnóstico:** Os limiares de decisão de $A_{2A}$ e hERG estavam dispersos em listas textuais em seções distintas, dificultando a consulta rápida e a memorização das faixas de corte.
- **Implementação:** No **Módulo 4**, foi criada uma **Tabela Unificada de Classificação e Limiares Operacionais**, integrando ambos os modelos em formato tabular de alto contraste com conversões numéricas diretas:

| Alvo | Classificação | Limite Operacional | Referência Numérica / Conversão |
| :--- | :--- | :--- | :--- |
| **A2A** | Alta potência predita | $pIC_{50} \ge 6{,}5$ (pIC₅₀ ≥ 6,5) | Faixa nanomolar ($IC_{50} \le 316\text{ nM}$) |
| **A2A** | Potência predita moderada | $5{,}0 \le pIC_{50} < 6{,}5$ (5,0 ≤ pIC₅₀ < 6,5) | Faixa micromolar ($316\text{ nM} < IC_{50} \le 10\ \mu\text{M}$) |
| **A2A** | Baixa potência/inativo | $pIC_{50} < 5{,}0$ (pIC₅₀ < 5,0) | Faixa sub-micromolar/inativa ($IC_{50} > 10\ \mu\text{M}$) |
| **hERG** | Baixo escore de classificação | $P < 0{,}30$ (P(hERG=1) < 30%) | Saída positiva $< 30\%$ ($1 - P > 0{,}70$) |
| **hERG** | Escore moderado | $0{,}30 \le P \le 0{,}70$ (30% até 70%, inclusive) | Saída positiva entre $30\%$ e $70\%$, inclusive ($0{,}30 \le 1 - P \le 0{,}70$) |
| **hERG** | Alto escore de classificação | $P > 0{,}70$ (P(hERG=1) > 70%) | Saída positiva $> 70\%$ ($1 - P < 0{,}30$) |

- **Resultado:** Eliminação de quaisquer ambiguidades nos limites de transição e padronização das faixas em conformidade com as regras de negócio do sistema.

---

### Correção 3 — Descrição Canônica do Nível 4 do Domínio de Aplicabilidade (AD)

- **Diagnóstico:** O texto anterior continha expressões de julgamento de valor sobre validade ("o Nível 4 não invalida o resultado", "não significa que a predição é inválida"), extrapolando a função diagnóstica estrita do AD.
- **Implementação:** A redação foi ajustada no **Módulo 4** para aderir estritamente à definição computacional:
  > **Nível 4 — Não Encontrado na Base:** indica que não foi encontrada uma correspondência exata nos critérios de proveniência utilizados pelo sistema. Esse resultado deve ser considerado na interpretação da análise, sem ser confundido com erro de processamento (o Nível 4 não é erro de software ou falha de inferência).
- **Justificativa:** O Nível 4 representa puramente a **ausência de correspondência exata** no espaço estrutural e informacional do treino, cabendo ao usuário julgar o grau de confiança à luz da similaridade contínua ($S_{\max}$).

---

### Correção 4 — Harmonização Terminológica Científica e Farmacológica

- **Diagnóstico:** Uso de termos que podiam induzir a interpretações clínicas ou farmacológicas precipitadas (ex.: "eficácia terapêutica", "probabilidade clínica de cardiotoxicidade", "perfil seguro").
- **Implementação:**
  1. *Substituição de "eficácia terapêutica":* Padronizado para **"potência predita frente ao receptor $A_{2A}$"** ou **"afinidade inibitória predita"**.
  2. *Substituição de "probabilidade clínica":* Padronizado para **"escore de classificação hERG"** ou **"fração de votos das árvores do classificador Random Forest"**, com alerta explícito de que não constitui probabilidade clínica calibrada de arritmia em humanos.
  3. *Substituição de "perfil seguro":* Padronizado para **"perfil computacional favorável"** no Quadrante 1.
  4. *Ressalvas de não-substituição:* Reforço de que os resultados são estritamente *in silico* e não substituem ensaios experimentais *in vitro* (patch clamp), ensaios de segurança regulatória (GLP) ou avaliação clínica.

---

### Correção 5 — Contextualização da Métrica $S_{\max}$ e Referência a $< 0{,}30$

- **Investigação do Código Real:** A inspeção em [`src/ad_utils.py`](file:///c:/Users/Raphael/Documents/BioPredicta/src/ad_utils.py) confirmou que a métrica $S_{\max}$ é calculada de forma puramente contínua ($[0, 1]$) através de `BulkTanimotoSimilarity`, **sem qualquer limiar binário fixo ou ponto de corte rígido codificado no motor de AD**.
- **Origem da Referência $< 0{,}30$:** Trata-se de uma observação empírica decorrente da fase de validação e calibração (`README.md` e Relatórios de Auditoria), na qual foi observado que moléculas ultradiscrepantes do perfil quimioterápico médio (como Taurina e L-Teanina) apresentavam $S_{\max} < 0{,}30$ e erro de predição substancialmente superior ($\text{RMSE} \approx 1{,}03$ vs $0{,}57$).
- **Implementação:** Na **Aba 3 (Diretrizes e Boas Práticas)**, o item foi redigido esclarecendo que o sistema opera com métrica contínua e que a faixa $< 0{,}30$ é uma referência observacional exploratória indicativa de **regime de extrapolação matemática acentuada**, que recomenda prudência analítica sem bloquear o processamento.

---

### Correção 6 — Revisão das Exportações em `src/ui/pages/__init__.py`

- **Diagnóstico:** O arquivo [`src/ui/pages/__init__.py`](file:///c:/Users/Raphael/Documents/BioPredicta/src/ui/pages/__init__.py) havia recebido incidentalmente a importação de `render_comparacao` durante intervenções anteriores.
- **Implementação:** A importação e exportação em `__all__` foram revertidas, mantendo apenas:
  ```python
  __all__ = [
      "render_nova_analise",
      "render_resultados",
      "render_interpretacao",
      "render_como_utilizar",
  ]
  ```
- **Resultado:** Isolamento modular preservado, sem vazamento indevido da interface de comparação através do pacote agregador de páginas.

---

## 3. VERIFICAÇÃO AUTOMATIZADA E TESTES

### 3.1. Suíte Específica da Página de Tutorial

A suíte [`tests/test_ui_p6_como_utilizar.py`](file:///c:/Users/Raphael/Documents/BioPredicta/tests/test_ui_p6_como_utilizar.py) foi atualizada para cobrir detalhadamente todas as exigências do Passo 1 e as 6 correções do Passo 1.1:

```bash
.\.venv\Scripts\python.exe -m pytest tests/test_ui_p6_como_utilizar.py -v
```

**Resultado:**
- 16 testes executados e **16 aprovados (100% de sucesso)** em 7,84 segundos.
- 0 falhas, 0 erros e 0 advertências de escape de strings (`SyntaxWarning` completamente eliminados através de `r"""`).

### 3.2. Suíte Geral de Regressão do Projeto

A suíte completa de testes do MultiTarget BioPredict foi executada para garantir que nenhuma regressão foi introduzida:

```bash
.\.venv\Scripts\python.exe -m pytest tests -m "not benchmark" -q
```

**Resultado:**
- **159 testes aprovados**, 1 desmarcado (benchmark) em 58,49 segundos.
- 100% de conformidade com todos os módulos de quimioinformática, modelos ML, utilitários de AD e componentes de interface.

---

## 4. INTEGRIDADE CRIPTOGRÁFICA DOS ARTEFATOS PROTEGIDOS

Os 8 arquivos críticos do projeto foram verificados via hash SHA-256 e permanecem **100% inalterados e íntegros**:

| Arquivo Protegido | Hash SHA-256 Esperado e Obtido | Status |
| :--- | :--- | :--- |
| `models/a2a_model.joblib` | `1907C7241EDF64FFAE3AA5C59A497037D427B1EAD6B83ABDF15ABBFE3B2A4787` | ✅ ÍNTEGRO |
| `models/a2a_model_metadata.json` | `DA3E05936977254E08C34DAB6B21009C3DA2C632647E062522662BD47A1D5F8F` | ✅ ÍNTEGRO |
| `models/herg_model.joblib` | `FCEDBECB89843870FE1B7DA9C501EB62B8E9E6A0A254A9ED8139C17D6AD154E6` | ✅ ÍNTEGRO |
| `models/herg_model_metadata.json` | `781F4B07B68EEEFF8F7FE245AA7513073811BB78819FE16996902C4D3146115F` | ✅ ÍNTEGRO |
| `data/processed/a2a_curated.csv` | `88354901075BC9058B6AE6B5AD8920A4A710F10B2EFD32ABD8AB02541B977A3E` | ✅ ÍNTEGRO |
| `data/processed/herg_curated.csv` | `DF7BE71479E64C05F735B17B35A6FD888CC6CA869615E9BDE50F489C715B3798` | ✅ ÍNTEGRO |
| `data/processed/a2a_featurized.parquet` | `AC006ABB90B59AC74C6D3A14CAD25CEEE62C58CEE44FA263A8A69F90928B31B7` | ✅ ÍNTEGRO |
| `data/processed/herg_featurized.parquet` | `612E115EBA6E47788ADB92A2EDD067E89D5AF9F5095DBA0DFCAD6FC817D7EB92` | ✅ ÍNTEGRO |

---

## 5. LIMITAÇÕES CONHECIDAS E FRONTEIRAS DE ESCOPO

1. **Páginas Adicionais da Central de Ajuda:** Conforme as diretrizes, a implementação parou estritamente no Passo 1.1 da Fase 5.1. As rotas para Fundamentos Científicos (`fundamentos`), Governança e Validação (`governanca`) e Sobre o Projeto (`sobre`) permanecem com seus respectivos placeholders até a autorização de suas fases específicas.
2. **Commit Git:** Nenhuma alteração foi commitada no Git local ou remoto nesta etapa.

---

## 6. CONCLUSÃO

A revisão corretiva da **Fase 5.1 — Passo 1.1** foi concluída com sucesso total. Todas as 6 correções foram aplicadas com precisão científica e rigor arquitetural, atestadas por 16 testes unitários específicos e 159 testes do ecossistema, preservando integralmente o Design System e os modelos de produção.

A plataforma está pronta para avaliação pelo usuário.

