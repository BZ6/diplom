#!/usr/bin/env python3
from __future__ import annotations

import argparse
import heapq
import os
import random
import sys
import time
import statistics
from dataclasses import dataclass, field

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

_MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")
sys.path.insert(0, _MODEL_DIR)

import torch
from model import RoutingModel, NeuralDijkstraModel, build_graph_tensors

_SAVED = os.path.join(_MODEL_DIR, "saved_model")

MODEL_PATHS = {
    "GNN+DQN":        os.path.join(_SAVED, "routing_model_gnn.pt"),
    "ST-GNN+DQN":     os.path.join(_SAVED, "routing_model.pt"),
    "SAGE+REINFORCE": os.path.join(_SAVED, "routing_model_sage_reinforce.pt"),
    "SAGE+PPO":       os.path.join(_SAVED, "routing_model_sage_ppo.pt"),
}
ND_PATH = os.path.join(_SAVED, "routing_model_nd.pt")


def _w(snr):
    return max(0.1, 10.0 - float(snr))


def _make_graph(rng, nodes, extra_edges=0):
    seen  = set()
    edges = []

    def _add(a, b):
        key = tuple(sorted([a, b]))
        if key in seen:
            return
        seen.add(key)
        snr = rng.uniform(8.0, 15.0) if "SERVER" in (a, b) else rng.uniform(0.5, 12.0)
        edges.append({"from": a, "to": b, "snr": snr})

    shuffled = nodes[:]
    rng.shuffle(shuffled)
    for i in range(1, len(shuffled)):
        _add(shuffled[i - 1], shuffled[i])
    if len(nodes) >= 2:
        for _ in range(extra_edges):
            _add(*rng.sample(nodes, 2))
    return nodes, edges


def gen_small_sparse(rng):
    n = rng.randint(6, 8)
    return _make_graph(rng, [f"n{i}" for i in range(n)], extra_edges=0)

def gen_small_dense(rng):
    n = rng.randint(6, 8)
    return _make_graph(rng, [f"n{i}" for i in range(n)], extra_edges=n * 2)

def gen_medium(rng):
    n = rng.randint(12, 16)
    return _make_graph(rng, [f"n{i}" for i in range(n)], extra_edges=n // 2)

def gen_large(rng):
    n = rng.randint(22, 30)
    return _make_graph(rng, [f"n{i}" for i in range(n)], extra_edges=n // 2)

def gen_xlarge(rng):
    n = rng.randint(40, 60)
    return _make_graph(rng, [f"n{i}" for i in range(n)], extra_edges=n // 2)

def gen_xxlarge(rng):
    n = rng.randint(70, 100)
    return _make_graph(rng, [f"n{i}" for i in range(n)], extra_edges=n // 2)

def gen_server_relay(rng):
    n = rng.randint(8, 12)
    return _make_graph(rng, [f"n{i}" for i in range(n)] + ["SERVER"], extra_edges=n // 3)


SCENARIOS = [
    ("Малая разреженная  (N=6-8,   редкие рёбра)",    gen_small_sparse),
    ("Малая плотная      (N=6-8,   плотные рёбра)",   gen_small_dense),
    ("Средняя            (N=12-16)",                   gen_medium),
    ("Большая            (N=22-30)",                   gen_large),
    ("Очень большая      (N=40-60)",                   gen_xlarge),
    ("Сверхбольшая       (N=70-100)",                  gen_xxlarge),
    ("С SERVER-ретранслятором (N=8-12 + SERVER)",      gen_server_relay),
]


def dijkstra_path(nodes, edges, src, dst):
    n2i = {n: i for i, n in enumerate(nodes)}
    if src not in n2i or dst not in n2i:
        return None, None
    N   = len(nodes)
    adj = {i: [] for i in range(N)}
    for e in edges:
        fi = n2i.get(e.get("from", ""))
        ti = n2i.get(e.get("to", ""))
        if fi is None or ti is None:
            continue
        ww = _w(e.get("snr", 5.0))
        adj[fi].append((ww, ti))
        adj[ti].append((ww, fi))
    si, di = n2i[src], n2i[dst]
    dist = [float("inf")] * N
    prev = [-1] * N
    dist[si] = 0.0
    heap = [(0.0, si)]
    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u]:
            continue
        for ww, v in adj[u]:
            nd = d + ww
            if nd < dist[v]:
                dist[v] = nd
                prev[v] = u
                heapq.heappush(heap, (nd, v))
    if dist[di] == float("inf"):
        return None, None
    path = []
    cur = di
    while cur != -1:
        path.append(cur)
        cur = prev[cur]
    path.reverse()
    return [nodes[i] for i in path], dist[di]


def greedy_route(adj_w, src, dst, max_hops=30):
    visited = {src}
    path    = [src]
    current = src
    for _ in range(max_hops):
        if current == dst:
            return path
        nbrs = [(ww, v) for ww, v in adj_w[current] if v not in visited]
        if not nbrs:
            return None
        _, nxt = min(nbrs)
        path.append(nxt)
        visited.add(nxt)
        current = nxt
    return None


def _path_cost(edges, path_ids):
    w_map = {}
    for e in edges:
        fn, tn = e.get("from", ""), e.get("to", "")
        ww = _w(e.get("snr", 5.0))
        w_map[(fn, tn)] = ww
        w_map[(tn, fn)] = ww
    total = 0.0
    for i in range(len(path_ids) - 1):
        ww = w_map.get((path_ids[i], path_ids[i + 1]))
        if ww is None:
            return float("inf")
        total += ww
    return total


def _path_cost_idx(adj_w, path):
    w_flat = {(u, v): ww for u, nbrs in adj_w.items() for ww, v in nbrs}
    total  = 0.0
    for i in range(len(path) - 1):
        ww = w_flat.get((path[i], path[i + 1]))
        if ww is None:
            return float("inf")
        total += ww
    return total


@dataclass
class MethodStats:
    success:     int  = 0
    costs:       list = field(default_factory=list)
    hops:        list = field(default_factory=list)
    times_us:    list = field(default_factory=list)
    opt_ratios:  list = field(default_factory=list)
    uses_server: int  = 0


@dataclass
class ScenarioResult:
    name:  str
    total: int  = 0
    stats: dict = field(default_factory=dict)


QUERIES_PER_GRAPH = 10


def run_scenario(name, gen_fn, rl_models, nd_model, n_queries, rng):
    method_names = ["Дейкстра", "Жадный"] + list(rl_models.keys()) + (
        ["GNN+Dijkstra"] if nd_model is not None else []
    )

    result = ScenarioResult(name=name, stats={m: MethodStats() for m in method_names})
    collected      = 0
    graph_attempts = 0

    while collected < n_queries and graph_attempts < n_queries * 20:
        graph_attempts += 1
        nodes, edges = gen_fn(rng)
        non_srv = [n for n in nodes if n != "SERVER"]
        if len(non_srv) < 2:
            continue

        nf, ei, ew, n2i = build_graph_tensors(nodes, edges)

        N     = len(nodes)
        adj_w = {i: [] for i in range(N)}
        for e in edges:
            fi = n2i.get(e.get("from", ""))
            ti = n2i.get(e.get("to", ""))
            if fi is None or ti is None:
                continue
            ww = _w(e.get("snr", 5.0))
            adj_w[fi].append((ww, ti))
            adj_w[ti].append((ww, fi))

        want = min(QUERIES_PER_GRAPH, n_queries - collected)

        t0 = time.perf_counter()
        rl_caches = {}
        for mname, mmodel in rl_models.items():
            emb, adj = mmodel.encode_graph(nf, ei, ew)
            rl_caches[mname] = (emb, adj)
        rl_prep_us  = (time.perf_counter() - t0) * 1_000_000
        per_rl_prep = rl_prep_us / max(len(rl_models), 1) / want

        nd_emb = nd_ei = nd_ew = None
        nd_prep_us = 0.0
        if nd_model is not None:
            t0 = time.perf_counter()
            nd_emb, nd_ei, nd_ew = nd_model.encode_graph(nf, ei, ew)
            nd_prep_us = (time.perf_counter() - t0) * 1_000_000 / want

        batch_collected = 0
        pair_attempts   = 0

        while batch_collected < want and pair_attempts < want * 10:
            pair_attempts += 1
            src, dst = rng.sample(non_srv, 2)

            t0 = time.perf_counter()
            d_path, d_cost = dijkstra_path(nodes, edges, src, dst)
            dijk_us = (time.perf_counter() - t0) * 1_000_000

            if d_path is None:
                continue

            result.total    += 1
            collected       += 1
            batch_collected += 1

            ms_d = result.stats["Дейкстра"]
            ms_d.success += 1
            ms_d.costs.append(d_cost)
            ms_d.hops.append(len(d_path) - 1)
            ms_d.times_us.append(dijk_us)
            ms_d.opt_ratios.append(1.0)
            if "SERVER" in d_path:
                ms_d.uses_server += 1

            si = n2i.get(src, -1)
            di = n2i.get(dst, -1)

            ms_g = result.stats["Жадный"]
            if si >= 0 and di >= 0:
                t0 = time.perf_counter()
                g_path_idx = greedy_route(adj_w, si, di)
                greedy_us  = (time.perf_counter() - t0) * 1_000_000
                ms_g.times_us.append(greedy_us)
                if g_path_idx is not None:
                    g_cost = _path_cost_idx(adj_w, g_path_idx)
                    if g_cost < float("inf"):
                        ms_g.success += 1
                        ms_g.costs.append(g_cost)
                        ms_g.hops.append(len(g_path_idx) - 1)
                        ms_g.opt_ratios.append(g_cost / d_cost)
                        g_path_ids = [nodes[i] for i in g_path_idx]
                        if "SERVER" in g_path_ids:
                            ms_g.uses_server += 1
            else:
                ms_g.times_us.append(0.0)

            for mname, mmodel in rl_models.items():
                ms = result.stats[mname]
                if si < 0 or di < 0:
                    ms.times_us.append(per_rl_prep)
                    continue

                emb, adj = rl_caches[mname]
                t0 = time.perf_counter()
                idx_path = mmodel.route_cached(emb, adj, si, di)
                route_us = (time.perf_counter() - t0) * 1_000_000

                ms.times_us.append(per_rl_prep + route_us)

                if idx_path is not None:
                    path_ids = [nodes[i] for i in idx_path]
                    cost = _path_cost(edges, path_ids)
                    if cost < float("inf"):
                        ms.success += 1
                        ms.costs.append(cost)
                        ms.hops.append(len(path_ids) - 1)
                        ms.opt_ratios.append(cost / d_cost)
                        if "SERVER" in path_ids:
                            ms.uses_server += 1

            if nd_model is not None:
                ms = result.stats["GNN+Dijkstra"]
                if si < 0 or di < 0 or nd_emb is None:
                    ms.times_us.append(nd_prep_us)
                else:
                    t0 = time.perf_counter()
                    idx_path = nd_model.route_cached(nd_emb, nd_ei, nd_ew, si, di)
                    nd_route_us = (time.perf_counter() - t0) * 1_000_000

                    ms.times_us.append(nd_prep_us + nd_route_us)

                    if idx_path is not None:
                        path_ids = [nodes[i] for i in idx_path]
                        cost = _path_cost(edges, path_ids)
                        if cost < float("inf"):
                            ms.success += 1
                            ms.costs.append(cost)
                            ms.hops.append(len(path_ids) - 1)
                            ms.opt_ratios.append(cost / d_cost)
                            if "SERVER" in path_ids:
                                ms.uses_server += 1

    return result


W = 88

def _hr(ch="─"):  return ch * W
def _center(t):   return t.center(W)

def _stats(arr):
    if not arr:
        return 0.0, 0.0, 0.0, 0.0
    m = statistics.mean(arr)
    s = statistics.stdev(arr) if len(arr) > 1 else 0.0
    return m, s, min(arr), max(arr)

def _pct(arr, thr):
    if not arr:
        return 0.0
    return sum(1 for x in arr if x <= thr) / len(arr) * 100


def format_results(results, method_order):
    lines = []
    add = lines.append

    add(_hr("═"))
    add(_center("СРАВНИТЕЛЬНЫЙ АНАЛИЗ: 7 МЕТОДОВ МАРШРУТИЗАЦИИ"))
    add(_center("Экспериментальное исследование маршрутизации в mesh-сети"))
    add(_hr("═"))

    for r in results:
        add("")
        add(_hr())
        add(f"  Сценарий: {r.name}")
        add(f"  Запросов: {r.total}")
        add(_hr())
        add(
            f"  {'Метод':<20}"
            f"  {'Успех':>8}"
            f"  {'Стоимость':>12}"
            f"  {'Ratio':>7}"
            f"  {'Хопы':>6}"
            f"  {'Время(мкс)':>12}"
        )
        add(f"  {'─'*20}  {'─'*8}  {'─'*12}  {'─'*7}  {'─'*6}  {'─'*12}")

        for mname in method_order:
            if mname not in r.stats:
                continue
            ms    = r.stats[mname]
            total = r.total
            sr    = ms.success / total * 100 if total else 0.0
            cm, cs, _, _ = _stats(ms.costs)
            om, os_, _, _ = _stats(ms.opt_ratios)
            hm, hs, _, _ = _stats([float(h) for h in ms.hops])
            tm, ts, _, _ = _stats(ms.times_us)
            add(
                f"  {mname:<20}"
                f"  {sr:>7.1f}%"
                f"  {cm:>8.2f}±{cs:<4.2f}"
                f"  {om:>7.3f}"
                f"  {hm:>5.1f}"
                f"  {tm:>8.1f}±{ts:<4.1f}"
            )

        add("")
        add("  Оптимальность (ratio = ML_стоимость / Дейкстра):")
        for mname in method_order:
            if mname == "Дейкстра" or mname not in r.stats:
                continue
            ms = r.stats[mname]
            if not ms.opt_ratios:
                continue
            p100 = _pct(ms.opt_ratios, 1.001)
            p110 = _pct(ms.opt_ratios, 1.10)
            p150 = _pct(ms.opt_ratios, 1.50)
            add(f"    {mname:<20}  ≤1.00:{p100:5.1f}%  ≤1.10:{p110:5.1f}%  ≤1.50:{p150:5.1f}%")

    add("")
    add(_hr("═"))
    add(_center("ИТОГОВАЯ СВОДКА"))
    add(_hr("═"))

    total_q = sum(r.total for r in results)
    add(f"  Всего запросов: {total_q}")
    add("")
    add(
        f"  {'Метод':<20}  {'Успех':>8}  {'Avg ratio':>10}  "
        f"{'≤1.10':>7}  {'≤1.50':>7}  {'Avg мкс':>10}"
    )
    add(f"  {'─'*20}  {'─'*8}  {'─'*10}  {'─'*7}  {'─'*7}  {'─'*10}")

    for mname in method_order:
        all_success = sum(r.stats[mname].success for r in results if mname in r.stats)
        all_ratios  = [x for r in results if mname in r.stats for x in r.stats[mname].opt_ratios]
        all_times   = [t for r in results if mname in r.stats for t in r.stats[mname].times_us]
        sr    = all_success / total_q * 100 if total_q else 0.0
        om, _, _, _ = _stats(all_ratios)
        tm, _, _, _ = _stats(all_times)
        p110  = _pct(all_ratios, 1.10)
        p150  = _pct(all_ratios, 1.50)
        add(
            f"  {mname:<20}  {sr:>7.1f}%  {om:>10.4f}  "
            f"{p110:>6.1f}%  {p150:>6.1f}%  {tm:>10.1f}"
        )

    add("")
    add("  Примечания:")
    add("    ratio = 1.000 → метод нашёл тот же оптимальный маршрут, что Дейкстра")
    add("    ratio > 1.000 → субоптимальный маршрут")
    add("    Время ML = амортизированное: (build+encode)/10 + маршрутизация")
    add(f"    (граф кодируется 1 раз на каждые {QUERIES_PER_GRAPH} запросов)")
    add(_hr("═"))

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", type=int, default=300)
    parser.add_argument("--seed",    type=int, default=42)
    parser.add_argument("--output",  type=str, default="")
    args = parser.parse_args()

    encoder_cfg = {
        "GNN+DQN":        dict(encoder_type='stgnn', use_global_attn=False),
        "ST-GNN+DQN":     dict(encoder_type='stgnn', use_global_attn=True),
        "SAGE+REINFORCE": dict(encoder_type='sage'),
        "SAGE+PPO":       dict(encoder_type='sage'),
    }

    rl_models = {}
    missing   = []

    for mname, path in MODEL_PATHS.items():
        if not os.path.exists(path):
            missing.append(f"  {mname:<20} → {path}")
            continue
        cfg   = encoder_cfg[mname]
        model = RoutingModel(**cfg)
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        model.eval()
        rl_models[mname] = model
        print(f"  Загружена модель: {mname:<20} ← {os.path.basename(path)}")

    nd_model = None
    if os.path.exists(ND_PATH):
        nd_model = NeuralDijkstraModel()
        nd_model.load_state_dict(torch.load(ND_PATH, map_location="cpu", weights_only=True))
        nd_model.eval()
        print(f"  Загружена модель: {'GNN+Dijkstra':<20} ← {os.path.basename(ND_PATH)}")
    else:
        missing.append(f"  {'GNN+Dijkstra':<20} → {ND_PATH}")

    if missing:
        print("\n  Не найдены модели (будут пропущены):")
        for m in missing:
            print(m)
        if not rl_models and nd_model is None:
            print("\nОшибка: ни одна модель не загружена. Обучите модели сначала.")
            sys.exit(1)

    method_order = ["Дейкстра", "Жадный"]
    for mname in ["GNN+DQN", "ST-GNN+DQN", "SAGE+REINFORCE", "SAGE+PPO"]:
        if mname in rl_models:
            method_order.append(mname)
    if nd_model is not None:
        method_order.append("GNN+Dijkstra")

    print(f"\nМетодов: {len(method_order)}   Запросов/сценарий: {args.queries}   seed={args.seed}\n")

    rng     = random.Random(args.seed)
    results = []

    for name, gen_fn in SCENARIOS:
        label = name[:55].ljust(55)
        print(f"  {label} … ", end="", flush=True)
        r = run_scenario(name, gen_fn, rl_models, nd_model, args.queries, rng)
        parts = []
        for mname in method_order:
            if mname in r.stats:
                ms = r.stats[mname]
                sr = ms.success / r.total * 100 if r.total else 0.0
                parts.append(f"{mname}:{sr:.0f}%")
        print(f"{r.total}q  " + "  ".join(parts))
        results.append(r)

    report = format_results(results, method_order)
    print()
    print(report)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(report + "\n")
        print(f"\nРезультаты записаны: {args.output}")


if __name__ == "__main__":
    main()
