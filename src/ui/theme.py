"""
Módulo de Tema e Design System Centralizado — MultiTarget BioPredict.

Define os tokens semânticos de cores, superfícies hierárquicas (N0 a N3),
escalas tipográficas locais e regras responsivas.
Elimina a dependência externa de fontes remotas em favor da pilha de sistema local.
"""

from __future__ import annotations

import streamlit as st

THEME_CSS: str = """
<style>
/* -------------------------------------------------------------------------
 * Design System Tokens (Dark Mode) — MultiTarget BioPredict
 * ------------------------------------------------------------------------- */
:root {
  /* Superfícies Hierárquicas */
  --bg-app: #0E1117;          /* N0: Fundo geral da aplicação */
  --surface-n1: #1E222D;      /* N1: Blocos de seção e contêineres principais */
  --surface-n2: #252A37;      /* N2: Cartões de dados, KPIs e caixas modais */
  --surface-n3: #2E3445;      /* N3: Elementos interativos, alertas e hover */

  /* Bordas e Divisores */
  --border-subtle: #333B4D;   /* Divisores discretos e bordas neutras */
  --border-card: #3E4759;     /* Borda padrão de cartões N2 */

  /* Tipografia */
  --text-primary: #F8FAFC;    /* Texto principal (contraste 13.8:1 sobre N2) */
  --text-secondary: #CBD5E1;  /* Texto de apoio (contraste 10.2:1 sobre N2) */
  --text-muted: #94A3B8;      /* Rótulos e metadados (contraste 5.8:1 sobre N2) */

  /* Cores de Marca e Identidade */
  --brand-cyan: #00D2BE;      /* Primário / Ações principais / A2A Alto */
  --brand-cyan-glow: rgba(0, 210, 190, 0.15);

  /* Semáforos Semânticos (Classes Operacionais) */
  --status-good: #00D2BE;     /* Ciano/Verde: Alta Potência A2A / Baixo Escore hERG */
  --status-warn: #F59E0B;     /* Âmbar: Potência Moderada A2A / Escore Moderado hERG */
  --status-bad: #FF4B4B;      /* Vermelho: Baixa Potência A2A / Alto Escore hERG */
  --status-info: #38BDF8;     /* Azul Informativo: Metadados e Badges */

  /* Mapeamento de Compatibilidade para Estilos Legados */
  --bg-primary: var(--bg-app);
  --bg-card: var(--surface-n1);
  --bg-card-elevated: var(--surface-n2);
  --accent-teal: var(--brand-cyan);
  --accent-teal-glow: var(--brand-cyan-glow);
  --text-main: var(--text-primary);
  --danger-red: var(--status-bad);
  --warning-amber: var(--status-warn);

  /* Espaçamentos Padronizados (Grid 8px) */
  --space-xs: 4px;
  --space-sm: 8px;
  --space-md: 16px;
  --space-lg: 24px;
  --space-xl: 32px;
}

/* -------------------------------------------------------------------------
 * Configuração Global de Elementos e Fontes de Sistema
 * ------------------------------------------------------------------------- */
html, body, .stApp {
    background-color: var(--bg-app) !important;
    color: var(--text-primary) !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
}

code, pre, .mono-font {
    font-family: "JetBrains Mono", Consolas, "Liberation Mono", Menlo, Courier, monospace !important;
}

/* Barra Lateral (Sidebar) */
section[data-testid="stSidebar"] {
    background-color: var(--surface-n1) !important;
    border-right: 1px solid var(--border-subtle) !important;
}

/* Cabeçalhos */
h1, h2, h3, h4, h5, h6 {
    color: var(--text-primary) !important;
    letter-spacing: -0.02em;
    font-weight: 700;
}

/* -------------------------------------------------------------------------
 * Superfícies e Cartões Reutilizáveis
 * ------------------------------------------------------------------------- */
.surface-n1 {
    background-color: var(--surface-n1);
    border: 1px solid var(--border-subtle);
    border-radius: 12px;
    padding: var(--space-md);
}

.surface-n2 {
    background-color: var(--surface-n2);
    border: 1px solid var(--border-card);
    border-radius: 10px;
    padding: var(--space-md);
}

.surface-n3 {
    background-color: var(--surface-n3);
    border: 1px solid var(--border-subtle);
    border-radius: 8px;
    padding: var(--space-sm) var(--space-md);
}

/* Hero Banner */
.hero-banner {
    background: linear-gradient(135deg, rgba(30, 34, 45, 0.95) 0%, rgba(14, 17, 23, 0.98) 100%);
    border: 1px solid rgba(0, 210, 190, 0.28);
    border-left: 5px solid var(--brand-cyan);
    border-radius: 14px;
    padding: 1.4rem 1.75rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 8px 28px rgba(0, 0, 0, 0.45);
}

.main-title {
    font-size: 2.25rem;
    font-weight: 800;
    background: linear-gradient(90deg, #00D2BE 0%, #38BDF8 55%, #FFFFFF 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.3rem;
    line-height: 1.15;
}

.sub-title {
    font-size: 1.00rem;
    color: var(--text-secondary);
    margin-bottom: 0;
    font-weight: 400;
    line-height: 1.5;
}

/* Cartões Principais (metric-card) */
.metric-card {
    background-color: var(--surface-n1);
    border: 1px solid var(--border-subtle);
    border-radius: 12px;
    padding: 1.25rem 1.35rem;
    margin-bottom: 1rem;
    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.35);
    transition: border-color 0.2s ease, transform 0.2s ease;
}

.metric-card:hover {
    border-color: rgba(0, 210, 190, 0.40);
}

.card-header-title {
    font-size: 1.05rem;
    font-weight: 700;
    color: var(--text-primary);
    margin-bottom: 0.75rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    border-bottom: 1px solid var(--border-subtle);
    padding-bottom: 0.55rem;
}

/* Grades e Caixas de KPI */
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 0.75rem;
    margin: 0.85rem 0;
}

.kpi-box {
    background-color: #141820;
    border: 1px solid var(--border-subtle);
    border-radius: 8px;
    padding: 0.75rem 0.9rem;
}

.kpi-label {
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--text-muted);
    font-weight: 600;
    margin-bottom: 0.2rem;
}

.kpi-value {
    font-size: 1.55rem;
    font-weight: 700;
    color: var(--text-primary);
}

.kpi-value-accent {
    color: var(--brand-cyan);
}

/* Semáforos Semânticos */
.semaphore-green {
    background: rgba(0, 210, 190, 0.14);
    border: 1.5px solid var(--status-good);
    color: #00F5D4;
    padding: 0.45rem 0.95rem;
    border-radius: 9999px;
    font-weight: 700;
    display: inline-block;
    font-size: 0.92rem;
    letter-spacing: 0.01em;
}

.semaphore-yellow {
    background: rgba(245, 158, 11, 0.15);
    border: 1.5px solid var(--status-warn);
    color: #FBBF24;
    padding: 0.45rem 0.95rem;
    border-radius: 9999px;
    font-weight: 700;
    display: inline-block;
    font-size: 0.92rem;
    letter-spacing: 0.01em;
}

.semaphore-red {
    background: rgba(255, 75, 75, 0.15);
    border: 1.5px solid var(--status-bad);
    color: #FF7676;
    padding: 0.45rem 0.95rem;
    border-radius: 9999px;
    font-weight: 700;
    display: inline-block;
    font-size: 0.92rem;
    letter-spacing: 0.01em;
}

/* Badges e Identificadores */
.insample-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    background: rgba(245, 158, 11, 0.16);
    border: 1.5px solid var(--status-warn);
    color: #FBBF24;
    font-weight: 700;
    font-size: 0.80rem;
    padding: 0.24rem 0.68rem;
    border-radius: 9999px;
    letter-spacing: 0.01em;
}

.outsample-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    background: rgba(56, 189, 248, 0.14);
    border: 1.5px solid var(--status-info);
    color: #7DD3FC;
    font-weight: 700;
    font-size: 0.80rem;
    padding: 0.24rem 0.68rem;
    border-radius: 9999px;
    letter-spacing: 0.01em;
}

.provisional-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: rgba(56, 189, 248, 0.12);
    border: 1px solid var(--status-info);
    color: #38BDF8;
    font-size: 0.82rem;
    font-weight: 600;
    padding: 0.3rem 0.75rem;
    border-radius: 6px;
    margin-bottom: 0.8rem;
}

/* Contêiner de Imagem Molecular */
.mol-canvas-frame {
    background: rgba(14, 17, 23, 0.82);
    border: 1.5px solid rgba(0, 210, 190, 0.40);
    border-radius: 10px;
    padding: 0.5rem;
    text-align: center;
}

.smiles-box {
    font-family: "JetBrains Mono", Consolas, monospace;
    font-size: 0.80rem;
    background: #141820;
    color: var(--brand-cyan);
    border: 1px solid rgba(0, 210, 190, 0.25);
    border-radius: 6px;
    padding: 0.50rem 0.70rem;
    word-break: break-all;
    margin-top: 0.50rem;
}


/* -------------------------------------------------------------------------
 * Cartões de Farmacóforos e Explicabilidade SHAP
 * ------------------------------------------------------------------------- */
.xai-audit-banner {
    background: linear-gradient(90deg, rgba(0, 210, 190, 0.12) 0%, rgba(30, 34, 45, 0.95) 100%);
    border: 1px solid rgba(0, 210, 190, 0.45);
    border-left: 5px solid var(--brand-cyan);
    border-radius: 12px;
    padding: 1rem 1.35rem;
    margin: 0.75rem 0 1.25rem 0;
    color: var(--text-primary);
    font-size: 0.95rem;
    line-height: 1.55;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.25);
}

.pharmacophore-card {
    background-color: var(--surface-n1);
    border: 1px solid var(--border-subtle);
    border-radius: 12px;
    padding: 1.15rem 1.25rem;
    margin-bottom: 1rem;
    min-height: 220px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    box-shadow: 0 4px 18px rgba(0, 0, 0, 0.30);
}

.pharmacophore-card-pos {
    border-top: 3px solid var(--status-good);
}

.pharmacophore-card-neg {
    border-top: 3px solid var(--status-bad);
}

.pharm-card-header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    gap: 0.5rem;
    border-bottom: 1px solid var(--border-subtle);
    padding-bottom: 0.65rem;
    margin-bottom: 0.8rem;
}

.pharm-feature-id {
    font-family: "JetBrains Mono", Consolas, monospace;
    font-size: 0.84rem;
    font-weight: 600;
    color: var(--brand-cyan);
    background: rgba(0, 210, 190, 0.12);
    padding: 0.16rem 0.55rem;
    border-radius: 6px;
    border: 1px solid rgba(0, 210, 190, 0.30);
    display: inline-block;
    margin-bottom: 0.3rem;
}

.pharm-feature-title {
    font-size: 1.02rem;
    font-weight: 700;
    color: var(--text-primary);
    line-height: 1.25;
}

.shap-tag-pos {
    font-family: "JetBrains Mono", Consolas, monospace;
    background: rgba(0, 210, 190, 0.16);
    border: 1px solid var(--status-good);
    color: #00F5D4;
    padding: 0.28rem 0.65rem;
    border-radius: 8px;
    font-size: 0.82rem;
    font-weight: 700;
    white-space: nowrap;
}

.shap-tag-neg {
    font-family: "JetBrains Mono", Consolas, monospace;
    background: rgba(255, 75, 75, 0.16);
    border: 1px solid var(--status-bad);
    color: #FF8080;
    padding: 0.28rem 0.65rem;
    border-radius: 8px;
    font-size: 0.82rem;
    font-weight: 700;
    white-space: nowrap;
}

.pharm-body-present {
    display: flex;
    gap: 0.95rem;
    align-items: center;
}

.bit-img-wrapper {
    background: rgba(14, 17, 23, 0.78);
    border: 1.5px solid rgba(0, 210, 190, 0.50);
    border-radius: 10px;
    padding: 6px;
    width: 140px;
    min-width: 140px;
    max-width: 140px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    box-shadow: inset 0 0 12px rgba(0, 0, 0, 0.45);
}

.bit-img-wrapper img {
    width: 126px;
    height: 92px;
    object-fit: contain;
    border-radius: 6px;
    display: block;
}

.bit-smarts-caption {
    font-family: "JetBrains Mono", Consolas, monospace;
    font-size: 0.68rem;
    color: var(--brand-cyan);
    background: rgba(30, 34, 45, 0.95);
    border: 1px solid var(--border-subtle);
    width: 100%;
    text-align: center;
    border-radius: 4px;
    padding: 2px 4px;
    margin-top: 4px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}

.absent-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    background: rgba(255, 75, 75, 0.14);
    border: 1px solid rgba(255, 75, 75, 0.45);
    color: #FF9494;
    font-weight: 700;
    font-size: 0.84rem;
    padding: 0.32rem 0.75rem;
    border-radius: 8px;
    margin-bottom: 0.55rem;
}

.present-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    background: rgba(0, 210, 190, 0.14);
    border: 1px solid rgba(0, 210, 190, 0.45);
    color: #00F5D4;
    font-weight: 700;
    font-size: 0.80rem;
    padding: 0.22rem 0.6rem;
    border-radius: 6px;
    margin-bottom: 0.45rem;
}

.descriptor-val-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.45rem;
    background: #141820;
    border: 1px solid rgba(0, 210, 190, 0.40);
    color: var(--text-primary);
    font-family: "JetBrains Mono", Consolas, monospace;
    font-weight: 700;
    font-size: 0.95rem;
    padding: 0.4rem 0.8rem;
    border-radius: 8px;
    margin-bottom: 0.6rem;
}

.pharm-desc-text {
    color: var(--text-secondary);
    font-size: 0.88rem;
    line-height: 1.5;
    margin: 0;
}

/* -------------------------------------------------------------------------
 * Regras Responsivas (Tablet e Mobile)
 * ------------------------------------------------------------------------- */
@media (max-width: 1024px) {
    .kpi-grid {
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    }
    .hero-banner {
        padding: 1.1rem 1.3rem;
    }
    .main-title {
        font-size: 1.85rem;
    }
}

@media (max-width: 640px) {
    .kpi-grid {
        grid-template-columns: 1fr;
    }
    .hero-banner {
        padding: 0.9rem 1.0rem;
    }
    .main-title {
        font-size: 1.50rem;
    }
    .metric-card {
        padding: 1rem;
    }
}
</style>
"""


def get_theme_css() -> str:
    """Retorna o bloco completo de CSS do Design System dark."""
    return THEME_CSS


def apply_theme() -> None:
    """Injeta a folha de estilos centralizada do Design System no Streamlit."""
    st.markdown(THEME_CSS, unsafe_allow_html=True)
