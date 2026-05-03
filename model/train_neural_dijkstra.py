from __future__ import annotations

import argparse
import heapq
import os
import random

import torch
import torch.nn.functional as F

from model import NeuralDijkstraModel, build_graph_tensors

SAVE_DIR        = os.path.join(os.path.dirname(__file__), "saved_model")
SAVE_PATH       = os.path.join(SAVE_DIR, "routing_model_nd.pt")

CURRICULUM = [
    (4,   12,  0.20),
    (8,   20,  0.20),
    (15,  40,  0.25),
    (30,  70,  0.20),
    (50, 100,  0.15),
]

LR              = 3e-4
PAIRS_PER_GRAPH = 8

_W = lambda snr: max(0.1, 10.0 - float(snr))


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


def gen_graph(rng, n_min, n_max):
    n = rng.randint(n_min, n_max)
    nodes = [f"n{i}" for i in range(n)]
    if rng.random() < 0.3:
        nodes.append("SERVER")
    extra = rng.randint(n // 4, n // 2)
    return _make_graph(rng, nodes, extra_edges=extra)


def _dijkstra_path_edges(adj_w, src, dst, N):
    dist = [float("inf")] * N
    prev = [-1] * N
    dist[src] = 0.0
    heap = [(0.0, src)]

    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u]:
            continue
        for w, v in adj_w[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                prev[v] = u
                heapq.heappush(heap, (nd, v))

    if dist[dst] == float("inf"):
        return None

    path_edges = set()
    cur = dst
    while prev[cur] != -1:
        path_edges.add((prev[cur], cur))
        path_edges.add((cur, prev[cur]))
        cur = prev[cur]
    return path_edges


def train(total_episodes=8_000, save_path=SAVE_PATH):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    rng       = random.Random(42)
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model     = NeuralDijkstraModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    print(f"=== GNN+Dijkstra  device={device}  graphs={total_episodes} ===")

    stage_eps = [(int(total_episodes * f), lo, hi) for lo, hi, f in CURRICULUM]
    ep_total  = 0

    for stage_idx, (n_ep, n_min, n_max) in enumerate(stage_eps):
        total_loss = 0.0
        trained    = 0

        for ep in range(n_ep):
            nodes, edges = gen_graph(rng, n_min, n_max)
            non_srv = [n for n in nodes if n != "SERVER"]
            if len(non_srv) < 2:
                continue

            n2i = {n: i for i, n in enumerate(nodes)}
            N   = len(nodes)

            adj_w = {i: [] for i in range(N)}
            for e in edges:
                fi = n2i.get(e.get("from", ""))
                ti = n2i.get(e.get("to", ""))
                if fi is None or ti is None:
                    continue
                w = _W(e.get("snr", 5.0))
                adj_w[fi].append((w, ti))
                adj_w[ti].append((w, fi))

            path_edge_set = set()
            pairs_found   = 0
            attempts      = 0

            while pairs_found < PAIRS_PER_GRAPH and attempts < PAIRS_PER_GRAPH * 3:
                attempts += 1
                if len(non_srv) < 2:
                    break
                src_id, dst_id = rng.sample(non_srv, 2)
                si, di = n2i[src_id], n2i[dst_id]
                pes = _dijkstra_path_edges(adj_w, si, di, N)
                if pes is not None:
                    path_edge_set |= pes
                    pairs_found += 1

            if pairs_found == 0:
                continue

            nf, ei, ew, _ = build_graph_tensors(nodes, edges)
            model.train()
            emb  = model.encoder(nf.to(device), ei.to(device), ew.to(device))
            ei_d = ei.to(device)

            if ei_d.shape[1] == 0:
                continue

            src_l = ei_d[0].tolist()
            dst_l = ei_d[1].tolist()
            labels = torch.tensor(
                [1.0 if (s, d) in path_edge_set else 0.0 for s, d in zip(src_l, dst_l)],
                dtype=torch.float32,
                device=device,
            )

            h_src = emb[ei_d[0]]
            h_dst = emb[ei_d[1]]

            if path_edge_set:
                dst_nodes   = list({d for _, d in path_edge_set} | {s for s, _ in path_edge_set})
                h_dest_mean = emb[dst_nodes].mean(0, keepdim=True).expand(ei_d.shape[1], -1)
            else:
                h_dest_mean = emb.mean(0, keepdim=True).expand(ei_d.shape[1], -1)

            feats  = torch.cat([h_src, h_dst, h_dest_mean], dim=1)
            scores = model.edge_scorer(feats).squeeze(-1)

            loss = F.binary_cross_entropy(scores, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            trained    += 1

        ep_total += n_ep
        avg_loss  = total_loss / max(trained, 1)
        print(f"  [ND stage {stage_idx+1}/5  N={n_min}-{n_max}]  "
              f"graphs={n_ep}  trained={trained}  avg_loss={avg_loss:.4f}  total={ep_total}")

    torch.save(model.state_dict(), save_path)
    print(f"\nSaved: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=8_000)
    parser.add_argument("--save",     type=str, default=SAVE_PATH)
    args = parser.parse_args()
    train(args.episodes, args.save)
