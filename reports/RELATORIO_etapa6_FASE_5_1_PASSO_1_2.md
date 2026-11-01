# RELATÓRIO TÉCNICO — FASE 5.1 (PASSO 1.2)
## Ajustes Finais de Precisão Científica e Terminológica do Tutorial de Utilização

**Projeto:** MultiTarget BioPredict  
**Fase:** Fase 5.1 — Passo 1.2: Ajustes Finais de Precisão Científica e Terminológica  
**Data:** 31 de Outubro de 2026  
**Branch Git:** `feat/ajustes-funcionalidades`  
**Status da Suíte de Testes:** 159 testes aprovados (100% de sucesso, 0 falhas, 1 desmarcado/benchmark)  
**Status dos Hashes Protegidos:** 8/8 íntegros e verificados criptograficamente via SHA-256  

---

## 1. RESUMO EXECUTIVO

O presente relatório documenta a conclusão da **Fase 5.1 — Passo 1.2**, referente aos **ajustes finais de precisão científica e terminológica do Tutorial de Utilização da Plataforma**, implementado em [`src/ui/pages/p6_como_utilizar.py`](file:///c:/Users/Raphael/Documents/BioPredicta/src/ui/pages/p6_como_utilizar.py).

Esta etapa consistiu em uma intervenção cirúrgica e estritamente textual, mantendo:
- A organização em três abas funcionais (**Início Rápido**, **Manual Completo** e **Diretrizes e Boas Práticas**);
- A integridade dos sete módulos sequenciais;
- A abordagem híbrida em duas camadas (apresentação essencial imediata e aprofundamento técnico sob demanda);
- A identidade visual do Design System, componentes e navegação em `app.py`;
- O congelamento de todos os modelos de Machine Learning, pipelines de quimioinformática e datasets.

---

## 2. DETALHAMENTO DOS QUATRO AJUSTES REALIZADOS

### Ajuste 1 — Descrição da Faixa de Baixa Potência A2A (Módulo 4)

- **Diagnóstico:** Na tabela unificada de limiares do Módulo 4, a terceira linha de classificação de $A_{2A}$ apresentava a descrição textual `Faixa sub-micromolar/inativa ($IC_{50} > 10\ \mu\text{M}$)`. Em farmacologia quantitativa, a concentração de $10\ \mu\text{M}$ ou superior situa-se na escala micromolar (ou milimolar), e nunca "sub-micromolar" (que corresponderia a frações de micromolar, i.e., nanomolar).
- **Ajuste Realizado:** A descrição textual foi corrigida para:
  $$\text{“Faixa micromolar ou superior (} IC_{50} > 10\ \mu\text{M} \text{).”}$$
- **Texto Consolidado na Tabela:**
  `| **A2A** | Baixa potência/inativo | $pIC_{50} < 5{,}0$ (pIC₅₀ < 5,0) | Faixa micromolar ou superior ($IC_{50} > 10\ \mu\text{M}$) |`
- **Impacto:** Preservação estrita dos limites numéricos operacionais ($pIC_{50} < 5{,}0$ e $IC_{50} > 10\ \mu\text{M}$) com absoluta exatidão terminológica físico-química.

---

### Ajuste 2 — Terminologia de Potência e Afinidade frente ao Receptor A2A

- **Diagnóstico:** Foram identificadas menções à expressão "afinidade inibitória predita" e à "afinidade global de ligação". Como o modelo de regressão para $A_{2A}$ foi treinado com dados experimentais de $IC_{50}$ (medida funcional de concentração inibitória semi-máxima e dependente do ensaio, descrita pela relação de Cheng-Prusoff), tratá-lo como mensurador direto de afinidade termodinâmica ($K_i$ ou $\Delta G^\circ$) constituía uma imprecisão farmacológica.
- **Ajustes Realizados:**
  1. *Módulo 1 (Apresentação Essencial):* Atualizado para:  
     `1. 🎯 **Potência predita frente ao receptor $A_{2A}$ ($ADORA2A$):** Estimativa contínua de potência inibitória expressa em $pIC_{50}$ em um receptor purinérgico acoplado à proteína G...`
  2. *Módulo 1 (Aprofundamento Técnico):* Atualizado para:  
     `- **Alvo Biológico 1 (Potência $A_{2A}$):** Receptor $A_{2A}$ humano (ChEMBL CHEMBL251). O modelo prediz a potência inibitória expressa em $pIC_{50} = 9{,}0 - \log_{10}(IC_{50\text{ [nM]}})$, inversamente proporcional à concentração inibitória semi-máxima ($IC_{50}$), sem inferir a seletividade frente aos subtipos $A_1$, $A_{2B}$ ou $A_3$.`
  3. *Módulo 2 (Controles de Referência):* Atualizado para:  
     `- *Taurina / L-Teanina:* Pequenas moléculas de baixa potência basal frente a $A_{2A}$ e baixo escore hERG.`
  4. *Módulo 5 (Visualização Multiobjetivo - Eixo X):* Atualizado para:  
     `- **Eixo X (Horizontal):** Potência predita frente ao receptor $A_{2A}$ ($pIC_{50}$). Valores mais à direita indicam maior potência inibitória frente ao receptor.`
- **Impacto:** Distinção rigorosa entre potência ($IC_{50} / pIC_{50}$) e afinidade de ligação, preservando integralmente o equacionamento logarítmico inverso.

---

### Ajuste 3 — Interpretação Metodológica de Smax (Aba Diretrizes e Boas Práticas)

- **Diagnóstico:** O card informativo da Aba 3 rotulava formalmente a faixa de similaridade $< 0{,}30$ como "regime de extrapolação matemática acentuada". Embora decorrente de observações exploratórias do projeto, a rotulação podia sugerir um ponto de corte algorítmico rígido inexistente no código do motor de AD.
- **Ajuste Realizado:** O texto e o cabeçalho do Card 2 foram reestruturados para refletir com exatidão a natureza contínua e a evidência exploratória:
  > **2. Interpretação Metodológica da Similaridade Máxima (S_max)**  
  > *O coeficiente `S_max` (calculado via similaridade de Tanimoto ECFP4 frente ao vizinho mais próximo da base de calibração) é uma métrica diagnóstica contínua entre 0 e 1, e o código de Domínio de Aplicabilidade não implementa um limiar rígido de 0,30. Valores de S_max inferiores a 0,30 foram observados em determinados compostos de baixa similaridade com as estruturas de referência, incluindo exemplos avaliados durante a validação do projeto. Essa observação exploratória sugere a necessidade de cautela adicional na interpretação, mas não constitui um limiar formal de validade ou rejeição da predição.*
- **Impacto:** Eliminação do rótulo formal de extrapolação acentuada; preservação da continuidade matemática de $S_{\max} \in [0, 1]$; transparência metodológica de que a observação exploratória orienta a cautela analítica sem atuar como critério de corte ou rejeição de software.

---

### Ajuste 4 — Padronização da Nomenclatura do Nível 4 do AD (Módulo 4)

- **Diagnóstico:** No Módulo 4, o Nível 4 vinha sendo referenciado com desdobramentos parentéticos adicionais ("não é erro de software ou falha de inferência"), os quais foram simplificados para harmonização estrita com a constante canônica do sistema (`PROVENANCE_LEVEL_4_NO_EXACT_MATCH`).
- **Ajuste Realizado:** A nomenclatura e o texto explicativo foram padronizados de forma límpida:
  > **Nível 4 — Sem correspondência exata conhecida:** Indica que não foi encontrada uma correspondência exata nos critérios de proveniência utilizados pelo sistema. Esse resultado deve ser considerado na interpretação da análise, sem ser confundido com erro de processamento.
- **Impacto:** A descrição registra com exatidão que nenhuma correspondência exata foi identificada pelos critérios de proveniência do sistema (sem sugerir ausência absoluta do composto em bases externas universais e sem atribuir conclusões sobre validade ou invalidade de predição).

---

## 3. ARQUIVOS MODIFICADOS

| Arquivo | Natureza da Modificação |
| :--- | :--- |
| [`src/ui/pages/p6_como_utilizar.py`](file:///c:/Users/Raphael/Documents/BioPredicta/src/ui/pages/p6_como_utilizar.py) | Aplicação dos quatro ajustes textuais de precisão científica (tabela de limiares, terminologia de potência, S_max metodológico e nomenclatura do Nível 4 do AD). |
| [`tests/test_ui_p6_como_utilizar.py`](file:///c:/Users/Raphael/Documents/BioPredicta/tests/test_ui_p6_como_utilizar.py) | Atualização das asserções de conformidade textual para validar exatamente as novas redações científicas (faixa micromolar, potência predita, ausência de 'afinidade inibitória', descrição canônica do Nível 4 e interpretação de S_max). |

---

## 4. RESULTADOS DA VALIDAÇÃO AUTOMATIZADA

### 4.1. Suíte Específica do Tutorial

A suíte [`tests/test_ui_p6_como_utilizar.py`](file:///c:/Users/Raphael/Documents/BioPredicta/tests/test_ui_p6_como_utilizar.py) foi executada para verificar a conformidade dos quatro ajustes:

```bash
.\.venv\Scripts\python.exe -m pytest tests/test_ui_p6_como_utilizar.py -v
```

**Resultado:**
```text
tests/test_ui_p6_como_utilizar.py::TestRenderComoUtilizarExecution::test_render_como_utilizar_runs_without_exceptions PASSED [  6%]
tests/test_ui_p6_como_utilizar.py::TestTutorialModulesContent::test_modulo_1_plataforma_e_ressalva_experimental PASSED [ 12%]
tests/test_ui_p6_como_utilizar.py::TestTutorialModulesContent::test_modulo_2_metodos_de_entrada_e_ketcher_apply PASSED [ 18%]
tests/test_ui_p6_como_utilizar.py::TestTutorialModulesContent::test_modulo_3_pipeline_computacional PASSED [ 25%]
tests/test_ui_p6_como_utilizar.py::TestTutorialModulesContent::test_modulo_4_limiares_canonicos_e_ad_niveis PASSED [ 31%]
tests/test_ui_p6_como_utilizar.py::TestTutorialModulesContent::test_modulo_5_visualizacao_multiobjetivo_e_quadrantes PASSED [ 37%]
tests/test_ui_p6_como_utilizar.py::TestTutorialModulesContent::test_modulo_6_shap_e_ressalva_nao_causalidade PASSED [ 43%]
tests/test_ui_p6_como_utilizar.py::TestTutorialModulesContent::test_modulo_7_historico_fifo_e_tanimoto_scaffold PASSED [ 50%]
tests/test_ui_p6_como_utilizar.py::TestAppNavigationIntegration::test_page_como_utilizar_connected_to_render_como_utilizar PASSED [ 56%]
tests/test_ui_p6_como_utilizar.py::TestAppNavigationIntegration::test_pages_navigation_preserves_como_utilizar_route PASSED [ 62%]
tests/test_ui_p6_como_utilizar.py::TestPasso11Corrections::test_correcao_1_abordagem_hibrida_em_todos_os_modulos PASSED [ 68%]
tests/test_ui_p6_como_utilizar.py::TestPasso11Corrections::test_correcao_2_tabela_unica_de_limiares PASSED [ 75%]
tests/test_ui_p6_como_utilizar.py::TestPasso11Corrections::test_correcao_3_descricao_canonica_ad_nivel_4 PASSED [ 81%]
tests/test_ui_p6_como_utilizar.py::TestPasso11Corrections::test_correcao_4_terminologia_cientifica_precisa PASSED [ 87%]
tests/test_ui_p6_como_utilizar.py::TestPasso11Corrections::test_correcao_5_smax_referencia_exploratoria PASSED [ 93%]
tests/test_ui_p6_como_utilizar.py::TestPasso11Corrections::test_correcao_6_exportacao_estrita_em_pages_init PASSED [100%]

============================= 16 passed in 12.80s =============================
```

### 4.2. Suíte Geral de Regressão

A suíte completa de testes unitários e de integração do MultiTarget BioPredict foi executada:

```bash
.\.venv\Scripts\python.exe -m pytest tests -m "not benchmark" -q
```

**Resultado:**
- **159 testes aprovados**, 1 desmarcado (benchmark) em 67,19 segundos.
- 0 falhas, 0 erros e 0 advertências de execução.

---

## 5. INTEGRIDADE CRIPTOGRÁFICA DOS ARTEFATOS PROTEGIDOS

A integridade dos 8 artefatos protegidos do projeto foi auditada via SHA-256 e comparada com os relatórios das etapas anteriores:

| Arquivo Protegido | Hash SHA-256 de Referência | Hash Obtido | Conformidade |
| :--- | :--- | :--- | :--- |
| `models/a2a_model.joblib` | `1907C7241EDF64FFAE3AA5C59A497037D427B1EAD6B83ABDF15ABBFE3B2A4787` | `1907C7241EDF64FFAE3AA5C59A497037D427B1EAD6B83ABDF15ABBFE3B2A4787` | ✅ ÍNTEGRO |
| `models/a2a_model_metadata.json` | `DA3E05936977254E08C34DAB6B21009C3DA2C632647E062522662BD47A1D5F8F` | `DA3E05936977254E08C34DAB6B21009C3DA2C632647E062522662BD47A1D5F8F` | ✅ ÍNTEGRO |
| `models/herg_model.joblib` | `FCEDBECB89843870FE1B7DA9C501EB62B8E9E6A0A254A9ED8139C17D6AD154E6` | `FCEDBECB89843870FE1B7DA9C501EB62B8E9E6A0A254A9ED8139C17D6AD154E6` | ✅ ÍNTEGRO |
| `models/herg_model_metadata.json` | `781F4B07B68EEEFF8F7FE245AA7513073811BB78819FE16996902C4D3146115F` | `781F4B07B68EEEFF8F7FE245AA7513073811BB78819FE16996902C4D3146115F` | ✅ ÍNTEGRO |
| `data/processed/a2a_curated.csv` | `88354901075BC9058B6AE6B5AD8920A4A710F10B2EFD32ABD8AB02541B977A3E` | `88354901075BC9058B6AE6B5AD8920A4A710F10B2EFD32ABD8AB02541B977A3E` | ✅ ÍNTEGRO |
| `data/processed/herg_curated.csv` | `DF7BE71479E64C05F735B17B35A6FD888CC6CA869615E9BDE50F489C715B3798` | `DF7BE71479E64C05F735B17B35A6FD888CC6CA869615E9BDE50F489C715B3798` | ✅ ÍNTEGRO |
| `data/processed/a2a_featurized.parquet` | `AC006ABB90B59AC74C6D3A14CAD25CEEE62C58CEE44FA263A8A69F90928B31B7` | `AC006ABB90B59AC74C6D3A14CAD25CEEE62C58CEE44FA263A8A69F90928B31B7` | ✅ ÍNTEGRO |
| `data/processed/herg_featurized.parquet` | `612E115EBA6E47788ADB92A2EDD067E89D5AF9F5095DBA0DFCAD6FC817D7EB92` | `612E115EBA6E47788ADB92A2EDD067E89D5AF9F5095DBA0DFCAD6FC817D7EB92` | ✅ ÍNTEGRO |

---

## 6. PRESERVAÇÃO DA ESTRUTURA DO TUTORIAL

A integridade estrutural da página [`src/ui/pages/p6_como_utilizar.py`](file:///c:/Users/Raphael/Documents/BioPredicta/src/ui/pages/p6_como_utilizar.py) foi rigorosamente mantida:
- **Três Abas:** `tab_rapido` (Início Rápido com 4 passos visuais), `tab_manual` (Manual Completo com os 7 módulos) e `tab_diretrizes` (Diretrizes e Boas Práticas);
- **Sete Módulos Sequenciais:** Todos presentes na ordem canônica (Módulo 1 ao 7), cada qual com sua `#### Apresentação Essencial` e respectivo `st.expander("🔬 Aprofundamento Técnico...", expanded=False)`;
- **Design System e Estilos:** Tokens CSS de superfície, bordas, cores semânticas e componentes de cabeçalho e rodapé mantidos sem alterações;
- **Rotas de Navegação:** Preservadas as integrações de `page_como_utilizar()` em `app.py`.

---

## 7. LIMITAÇÕES CONHECIDAS E CONDIÇÃO DE PARADA

- Conforme exigido nas diretrizes operacionais, a execução encerrou-se estritamente na finalização do **Passo 1.2 da Fase 5.1**.
- Nenhuma modificação foi realizada nas páginas de **Fundamentos Científicos** (`fundamentos`), **Governança e Validação** (`governanca`) ou **Sobre o Projeto** (`sobre`).
- Nenhum commit Git foi realizado.

O Tutorial de Utilização encontra-se em estado final de precisão científica, funcionalidade e elegância visual, aguardando avaliação e aprovação do usuário para as etapas posteriores.

