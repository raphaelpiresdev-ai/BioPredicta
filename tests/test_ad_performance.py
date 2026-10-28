"""
Suíte de testes de desempenho (benchmark) do Domínio de Aplicabilidade (AD) — Fase 3D-3.

Mede:
1. Tempo de carregamento inicial (cold load) das referências A2A e hERG.
2. Latência de consultas com cache aquecido (warm cache) para 100 repetições.
3. Mediana, percentil 95 (P95), mínimo e máximo das latências.
4. Consumo de memória pelo cache de referência via tracemalloc.

Isolada da suíte funcional: não falha por oscilações espúrias de hardware.
"""

from __future__ import annotations

import platform
import time
import tracemalloc
from pathlib import Path
from typing import List

import numpy as np
import pytest

from src.ad_utils import (
    compute_ad_diagnostics,
    get_ad_reference,
    load_ad_reference,
)

BENCHMARK_REPETITIONS = 100
TEST_SMILES = "CC(C)(C)c1ccc(cc1)C(O)CCCN2CCC(CC2)C(O)(c3ccccc3)c4ccccc4"  # Terfenadina


@pytest.mark.benchmark
def test_ad_performance_benchmark():
    """Executa a medição controlada de latência e memória para o diagnóstico AD."""
    print("\n--- Benchmark de Desempenho do Domínio de Aplicabilidade (AD) ---")
    print(f"Ambiente: Python {platform.python_version()} on {platform.system()} ({platform.machine()})")

    # 1. Carregamento inicial (Cold Load)
    t0 = time.perf_counter()
    ref_a2a = load_ad_reference("A2A")
    t1 = time.perf_counter()
    a2a_load_time_ms = (t1 - t0) * 1000

    t2 = time.perf_counter()
    ref_herg = load_ad_reference("hERG")
    t3 = time.perf_counter()
    herg_load_time_ms = (t3 - t2) * 1000

    print(f"Tempo de carga inicial A2A (1.364 amostras): {a2a_load_time_ms:.2f} ms")
    print(f"Tempo de carga inicial hERG (13.136 amostras): {herg_load_time_ms:.2f} ms")

    # 2. Aquecimento de cache (Warm-up)
    _ = compute_ad_diagnostics(TEST_SMILES, "A2A", reference=ref_a2a)
    _ = compute_ad_diagnostics(TEST_SMILES, "hERG", reference=ref_herg)

    # 3. Medição de latência com cache aquecido (100 repetições)
    latencies_a2a: List[float] = []
    latencies_herg: List[float] = []

    for _ in range(BENCHMARK_REPETITIONS):
        t_start = time.perf_counter()
        _ = compute_ad_diagnostics(TEST_SMILES, "A2A", reference=ref_a2a)
        latencies_a2a.append((time.perf_counter() - t_start) * 1000)

        t_start = time.perf_counter()
        _ = compute_ad_diagnostics(TEST_SMILES, "hERG", reference=ref_herg)
        latencies_herg.append((time.perf_counter() - t_start) * 1000)

    # Estatísticas
    a2a_med = float(np.median(latencies_a2a))
    a2a_p95 = float(np.percentile(latencies_a2a, 95))
    a2a_min = float(np.min(latencies_a2a))
    a2a_max = float(np.max(latencies_a2a))

    herg_med = float(np.median(latencies_herg))
    herg_p95 = float(np.percentile(latencies_herg, 95))
    herg_min = float(np.min(latencies_herg))
    herg_max = float(np.max(latencies_herg))

    print(f"\nLatência de consulta A2A ({BENCHMARK_REPETITIONS} reps):")
    print(f"  Mín: {a2a_min:.2f} ms | Mediana: {a2a_med:.2f} ms | P95: {a2a_p95:.2f} ms | Máx: {a2a_max:.2f} ms")

    print(f"\nLatência de consulta hERG ({BENCHMARK_REPETITIONS} reps):")
    print(f"  Mín: {herg_min:.2f} ms | Mediana: {herg_med:.2f} ms | P95: {herg_p95:.2f} ms | Máx: {herg_max:.2f} ms")

    # 4. Medição de memória consumida
    tracemalloc.start()
    snap_before = tracemalloc.take_snapshot()

    # Criação de referência isolada para quantificar alocação de heap
    ref_herg_mem = load_ad_reference("hERG")
    snap_after = tracemalloc.take_snapshot()
    tracemalloc.stop()

    stats = snap_after.compare_to(snap_before, "lineno")
    total_bytes = sum(s.size for s in stats if s.size > 0)
    mem_mb = total_bytes / (1024 * 1024)
    print(f"\nMemória alocada pelo cache hERG (tracemalloc): ~{mem_mb:.2f} MB")

    # Asserções não-frágeis de integridade (não dependem de relógio do hardware)
    assert len(latencies_a2a) == BENCHMARK_REPETITIONS
    assert len(latencies_herg) == BENCHMARK_REPETITIONS
    assert a2a_med > 0.0
    assert herg_med > 0.0
    assert mem_mb > 0.0

