# RELATÓRIO DE IMPLEMENTAÇÃO — ETAPA 5.2
## Histórico Não-Destrutivo de Análises Moleculares — MultiTarget BioPredict

**Projeto:** MultiTarget BioPredict  
**Etapa:** Subetapa 5.2 — Gerenciamento do Histórico de Moléculas em Sessão  
**Data:** 31 de Outubro de 2026  
**Branch Git:** `feat/ajustes-funcionalidades`  
**Status de Integridade:** 8/8 Hashes SHA-256 Verificados e 100% Intactos  
**Status da Suíte de Testes:** 115/115 Testes Funcionais Aprovados (0 falhas, 0 erros)

---

## 1. OBJETIVO DA SUBETAPA 5.2

Implementar um mecanismo robusto, desacoplado e não-destrutivo para gerenciamento do histórico de análises moleculares durante a sessão ativa do usuário no Streamlit. 

Antes desta implementação, qualquer interação com a entrada (digitação de novo SMILES, desenho no canvas Ketcher, clique em "Limpar" ou erro de sintaxe química) destruía sumariamente os resultados pré-calculados em `st.session_state["current_analysis"]`. Com a Etapa 5.2, o usuário pode analisar múltiplas moléculas sequencialmente, mantendo todas as predições, métricas de Domínio de Aplicabilidade (AD) e tensores SHAP salvos na sessão, podendo alternar entre moléculas analisadas com um único clique.

---

## 2. ARQUIVOS CRIADOS E MODIFICADOS

| Arquivo | Ação | Descrição da Intervenção |
| :--- | :---: | :--- |
| `src/ui/history_manager.py` | **Criado** | Módulo especializado em gerenciamento do histórico em sessão (`st.session_state["analysis_history"]`), com controle FIFO (máx 10 itens), deduplicação por SMILES canônico, isolamento de memória (deep copies), supressão de imagens pesadas e reconstrução determinística de objetos RDKit `Mol` via `smiles_to_mol`. |
| `src/ui/pages/p2_nova_analise.py` | **Modificado** | 1. Registro automático da análise no histórico após a execução bem-sucedida de `execute_molecular_pipeline()`.<br>2. Eliminação de todas as purgas destrutivas que anulavam `current_analysis` em ações parciais de formulário.<br>3. Banner informativo de topo identificando a molécula ativa no painel.<br>4. Seção visual `render_session_history_section()` com lista ordenada reversa (mais recente no topo) e botões de reativação imediata. |
| `tests/test_ui_p5_history.py` | **Criado** | Suíte de testes automatizados com 13 testes unitários e de integração cobrindo os 12 requisitos mandatórios da especificação. |
| `models/*`, `data/processed/*` | **Preservados** | Nenhuma alteração; modelos, parâmetros, limiares canônicos e bases de dados 100% intactos. |

---

## 3. FUNCIONAMENTO DO HISTÓRICO E DA SELEÇÃO DE MOLÉCULAS

### 3.1. Arquitetura do Gerenciador de Histórico (`history_manager.py`)
O módulo disponibiliza as seguintes funções canônicas:

- `init_history(session_state=None) -> List[Dict[str, Any]]`:  
  Inicializa `session_state["analysis_history"]` como lista vazia se ausente.
- `add_to_history(analysis_dict, session_state=None, max_size=10) -> Optional[Dict[str, Any]]`:  
  Valida se a análise possui `canonical_smiles`. Cria uma cópia profunda sanitizada dos dados (tensores SHAP, predições, AD, metadados), omitindo objetos pesados em memória. Se a molécula já constar no histórico, atualiza seus dados e reposiciona-a como o registro mais recente (final da lista). Aplica política FIFO descartando o registro mais antigo quando a contagem ultrapassar `max_size=10`.
- `get_history(session_state=None) -> List[Dict[str, Any]]`:  
  Retorna os registros em ordem cronológica FIFO.
- `set_active_molecule(canonical_smi, session_state=None) -> Optional[Dict[str, Any]]`:  
  Localiza o registro pelo SMILES canônico, restaura o objeto `Chem.Mol` do RDKit deterministamente e atualiza atomicamente:
  - `session_state["current_analysis"] = active_dict`
  - `session_state["current_analysis_smiles"] = canonical_smi`
- `clear_history(session_state=None) -> None`:  
  Limpa os registros do histórico sem anular a molécula atualmente ativa no painel.
- `get_history_summary_list(session_state=None) -> List[Dict[str, Any]]`:  
  Formata e reverte a ordem da lista (mais recente no topo), indicando qual molécula é a ativa no momento via flag `is_active`.

### 3.2. Separação Epistemológica: "Molécula em Edição" vs. "Molécula Ativa no Painel"
A interface agora distingue categoricamente dois estados que antes colidiam:
1. **Molécula em Edição (`raw_input_structure` / `canonical_smi`):**  
   O rascunho sendo digitado no campo de texto ou desenhado no canvas do Ketcher. Alterar esse rascunho **não afeta nem apaga** a molécula ativa no painel. O preview 2D prévio reflete a estrutura em edição.
2. **Molécula Ativa no Painel (`st.session_state["current_analysis"]`):**  
   A última análise concluída ou a molécula recuperada do histórico. É ela que alimenta as páginas [p3_resultados.py](file:///c:/Users/Raphael/Documents/BioPredicta/src/ui/pages/p3_resultados.py) e [p4_interpretacao.py](file:///c:/Users/Raphael/Documents/BioPredicta/src/ui/pages/p4_interpretacao.py).

### 3.3. Experiência do Usuário (UX)
- No topo da página Nova Análise, um chip informativo discreto avisa:
  > 🔬 **Molécula Ativa no Painel de Resultados:** `[Nome/Título]` *(pIC₅₀: X.XX | Escore hERG: Y.YY)* `[✓ Carregada]`
- No rodapé da página (e visível mesmo se o canvas estiver limpo ou com erro de sintaxe), a seção:
  > **📚 Histórico de Análises na Sessão (N/10)**  
  Apresenta os cartões de cada composto avaliado com fórmula, massa, potências preditas, quadrante multiobjetivo e botão **"🔍 Ativar no Painel"**.
- Ao clicar em "Ativar no Painel", a molécula é restaurada em `current_analysis`, o campo de SMILES é sincronizado com a estrutura e um `st.rerun()` atualiza a aplicação instantaneamente.

---

## 4. ESTRATÉGIA DE MEMÓRIA E RECONSTRUÇÃO ESTRUTURAL

Para respeitar a restrição arquitetural de rodar localmente com limite de 8 GB RAM e evitar estouro de memória no Streamlit:

1. **Supressão de Imagens no Histórico:** Imagens base64 (como previews 2D gerados na renderização imediata) são omitidas do dicionário armazenado no histórico (`mol_b64`, `b64_subenv`, `b64_highlight`). Elas são geradas dinamicamente sob demanda sempre que a página as requisita.
2. **Eliminação de Referências C++ Mutáveis RDKit:** Objetos `Chem.Mol` instanciados em C++ podem causar vazamentos de memória ou falhas de serialização se duplicados repetidamente no heap do Python. No histórico, armazena-se `mol = None` e o `canonical_smiles`. Ao invocar `set_active_molecule()`, a função `smiles_to_mol(canonical_smiles)` reconstrói o objeto RDKit instantaneamente em microssegundos.
3. **Preservação Íntegra dos Tensores SHAP:** Os arrays NumPy de 1.030 floats gerados pelo `TreeExplainer` para $A_{2A}$ e hERG são duplicados via `np.copy()`. Isso garante que o cálculo computacionalmente pesado do SHAP não precise ser reexecutado ao alternar entre moléculas do histórico.
4. **Preservação dos DataFrames de Atributos:** O DataFrame `features` (1.030 colunas) é copiado via `.copy()`, assegurando compatibilidade com as rotinas de visualização de descritores de `p3_resultados` e `p4_interpretacao`.

---

## 5. RESULTADOS DA SUÍTE DE TESTES AUTOMATIZADOS

A suíte completa oficial foi executada via Pytest:
```powershell
.\.venv\Scripts\python.exe -m pytest tests -v -m "not benchmark"
```

### Sumário da Execução:
- **Testes Coletados:** 116
- **Testes Deselecionados (benchmark isolado):** 1
- **Testes Executados:** 115
- **Aprovados:** **115 (100% de sucesso)**
- **Falhas:** 0
- **Erros:** 0
- **Tempo de Execução:** 53.86 segundos

### Módulos Testados:
- `tests/test_applicability_domain.py`: 10 aprovados
- `tests/test_controls_and_interpretation.py`: 5 aprovados
- `tests/test_featurization_regression.py`: 7 aprovados
- `tests/test_multiobjective_plot.py`: 3 aprovados
- `tests/test_pubchem_identification.py`: 6 aprovados
- `tests/test_ui_p2_nova_analise.py`: 47 aprovados (todos os testes de validação, limites canônicos e Ketcher verdes)
- `tests/test_ui_p3_resultados.py`: 10 aprovados
- `tests/test_ui_p4_interpretacao.py`: 14 aprovados
- `tests/test_ui_p5_history.py` (**Novo**): 13 aprovados:
  1. `test_init_history`: Inicialização e idempotência.
  2. `test_add_valid_analysis`: Inserção com sucesso.
  3. `test_field_preservation`: Preservação de predições, AD, SHAP e PubChem.
  4. `test_deduplication_by_canonical_smiles`: Atualização sem duplicação de chave.
  5. `test_reanalyzed_molecule_repositioned_to_most_recent`: Reordenação temporal.
  6. `test_fifo_limit_enforcement`: Descarte estrito do 11º item em diante.
  7. `test_set_active_molecule`: Reativação atômica com reconstrução do objeto `mol`.
  8. `test_independent_copies_no_side_effects`: Ausência de mutação compartilhada.
  9. `test_history_preserved_during_form_editing`: Preservação do histórico durante digitação.
  10. `test_active_analysis_compatibility_with_pages`: Compatibilidade funcional com `p3_resultados` e `p4_interpretacao`.
  11. `test_rejection_of_invalid_or_incomplete_records`: Rejeição graciosa de entradas espúrias.
  12. `test_clear_history`: Limpeza controlada.
  13. `test_history_summary_list_presentation`: Formatação e marcação da molécula ativa.

---

## 6. VERIFICAÇÃO CRIPTOGRÁFICA DOS OITO HASHES PROTEGIDOS

| Arquivo Protegido | SHA-256 Esperado | SHA-256 Atual | Status |
| :--- | :--- | :--- | :---: |
| `models/a2a_model.joblib` | `1907c7241edf64ffae3aa5c59a497037d427b1ead6b83abdf15abbfe3b2a4787` | `1907c7241edf64ffae3aa5c59a497037d427b1ead6b83abdf15abbfe3b2a4787` | **CONFERE** |
| `models/a2a_model_metadata.json` | `da3e05936977254e08c34dab6b21009c3da2c632647e062522662bd47a1d5f8f` | `da3e05936977254e08c34dab6b21009c3da2c632647e062522662bd47a1d5f8f` | **CONFERE** |
| `models/herg_model.joblib` | `fcedbecb89843870fe1b7da9c501eb62b8e9e6a0a254a9ed8139c17d6ad154e6` | `fcedbecb89843870fe1b7da9c501eb62b8e9e6a0a254a9ed8139c17d6ad154e6` | **CONFERE** |
| `models/herg_model_metadata.json` | `781f4b07b68eeeff8f7fe245aa7513073811bb78819fe16996902c4d3146115f` | `781f4b07b68eeeff8f7fe245aa7513073811bb78819fe16996902c4d3146115f` | **CONFERE** |
| `data/processed/a2a_curated.csv` | `88354901075bc9058b6ae6b5ad8920a4a710f10b2efd32abd8ab02541b977a3e` | `88354901075bc9058b6ae6b5ad8920a4a710f10b2efd32abd8ab02541b977a3e` | **CONFERE** |
| `data/processed/herg_curated.csv` | `df7be71479e64c05f735b17b35a6fd888cc6ca869615e9bde50f489c715b3798` | `df7be71479e64c05f735b17b35a6fd888cc6ca869615e9bde50f489c715b3798` | **CONFERE** |
| `data/processed/a2a_featurized.parquet` | `ac006abb90b59ac74c6d3a14cad25ceee62c58cee44fa263a8a69f90928b31b7` | `ac006abb90b59ac74c6d3a14cad25ceee62c58cee44fa263a8a69f90928b31b7` | **CONFERE** |
| `data/processed/herg_featurized.parquet` | `612e115eba6e47788adb92a2edd067e89d5af9f5095dba0dfcad6fc817d7eb92` | `612e115eba6e47788adb92a2edd067e89d5af9f5095dba0dfcad6fc817d7eb92` | **CONFERE** |

---

## 7. CONCLUSÃO E CONFORMIDADE COM O CRITÉRIO DE ACEITAÇÃO

A Subetapa 5.2 foi implementada integralmente em conformidade com as especificações:
- O usuário pode avaliar uma molécula, digitar ou desenhar uma nova estrutura sem apagar a anterior, e recuperar qualquer uma das até 10 análises anteriores mantidas no histórico da sessão.
- A molécula reativada carrega todos os seus dados nas páginas `Resultados` e `Interpretação` sem recalcular modelos nem reprocessar SHAP.
- Nenhum modelo ou dataset foi modificado, e todos os 115 testes funcionais foram aprovados.

A execução encontra-se interrompida e aguarda sua validação formal antes de qualquer avanço para a Etapa 5.3.

