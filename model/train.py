"""
Training script for ST-GNN + DQN mesh routing model.

Generates synthetic random mesh graphs and trains a DQN agent to find
optimal (minimum-weight) paths.  Saves weights to saved_model/routing_model.pt.

Usage:
    python train.py [--episodes N] [--save PATH]
"""
from __future__ import annotations

import argparse
import collections
import os
import random
import heapq
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from model import RoutingModel, build_graph_tensors

SAVE_DIR = os.path.join(os.path.dirname(__file__), "saved_model")
SAVE_PATH = os.path.join(SAVE_DIR, "routing_model.pt")

# RL hyper-parameters
GAMMA           = 0.95
LR              = 1e-3
BATCH_SIZE      = 64
REPLAY_CAPACITY = 10_000
MIN_REPLAY      = 500
EPS_START       = 1.0
EPS_END         = 0.05
EPS_DECAY       = 0.9995
TARGET_UPDATE   = 200
MAX_HOPS        = 15


Experience = collections.namedtuple(
    "Experience",
    ["state", "action_emb", "reward", "next_state", "next_cands", "done"],
)


# ---------------------------------------------------------------------------
# Synthetic graph generator
# ---------------------------------------------------------------------------

def generate_graph(
    n_nodes: Optional[int] = None,
    rng: random.Random = random,
) -> Tuple[List[str], List[dict]]:
    """Random connected graph; ~50 % chance of a SERVER node."""
    n = n_nodes or rng.randint(4, 12)
    nodes = [f"n{i}" for i in range(n)]
    if rng.random() > 0.5:
        nodes.append("SERVER")

    seen: set = set()
    edges: List[dict] = []

    def _add(a, b):
        key = tuple(sorted([a, b]))
        if key in seen:
            return
        seen.add(key)
        snr = rng.uniform(8.0, 15.0) if ("SERVER" in (a, b)) else rng.uniform(0.5, 12.0)
        edges.append({"from": a, "to": b, "snr": snr})

    # spanning tree for guaranteed connectivity
    shuffled = nodes[:]
    rng.shuffle(shuffled)
    for i in range(1, len(shuffled)):
        _add(shuffled[i - 1], shuffled[i])

    # extra random edges
    for _ in range(rng.randint(0, n)):
        _add(*rng.sample(nodes, 2))

    return nodes, edges


# ---------------------------------------------------------------------------
# Dijkstra reference (for checking reachability)
# ---------------------------------------------------------------------------

def dijkstra(nodes, edges, src, dst) -> Optional[float]:
    """Returns shortest-path cost or None if unreachable."""
    n2i = {n: i for i, n in enumerate(nodes)}
    if src not in n2i or dst not in n2i:
        return None
    adj: Dict[int, List[Tuple[float, int]]] = {i: [] for i in range(len(nodes))}
    for e in edges:
        fi, ti = n2i.get(e["from"]), n2i.get(e["to"])
        if fi is None or ti is None:
            continue
        w = max(0.1, 10.0 - float(e["snr"]))
        adj[fi].append((w, ti))
        adj[ti].append((w, fi))
    dist = [float("inf")] * len(nodes)
    dist[n2i[src]] = 0.0
    heap = [(0.0, n2i[src])]
    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u]:
            continue
        for w, v in adj[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(heap, (nd, v))
    cost = dist[n2i[dst]]
    return None if cost == float("inf") else cost


# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buf: collections.deque = collections.deque(maxlen=capacity)

    def push(self, *args):
        self.buf.append(Experience(*args))

    def sample(self, n: int) -> List[Experience]:
        return random.sample(self.buf, n)

    def __len__(self):
        return len(self.buf)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def _make_state(emb, curr, dest, progress):
    prog = torch.tensor([progress], dtype=torch.float32)
    return torch.cat([emb[curr], emb[dest], prog])


def train(num_episodes: int = 5_000, save_path: str = SAVE_PATH):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    rng = random.Random(42)

    policy = RoutingModel()
    target = RoutingModel()
    target.load_state_dict(policy.state_dict())
    target.eval()

    optimizer = torch.optim.Adam(policy.parameters(), lr=LR)
    replay    = ReplayBuffer(REPLAY_CAPACITY)
    epsilon   = EPS_START
    step_n    = wins = losses = 0

    for ep in range(1, num_episodes + 1):
        nodes, edges = generate_graph(rng=rng)
        nf, ei, ew, n2i = build_graph_tensors(nodes, edges)

        non_srv = [n for n in nodes if n != "SERVER"]
        if len(non_srv) < 2:
            continue
        src, dst = rng.sample(non_srv, 2)
        if dijkstra(nodes, edges, src, dst) is None:
            continue

        si, di = n2i[src], n2i[dst]

        # Build adjacency list
        N   = len(nodes)
        adj: Dict[int, List[int]] = {i: [] for i in range(N)}
        for e in edges:
            fi, ti = n2i.get(e["from"]), n2i.get(e["to"])
            if fi is not None and ti is not None:
                adj[fi].append(ti)
                adj[ti].append(fi)

        with torch.no_grad():
            emb = policy.encode(nf, ei, ew)

        current  = si
        visited  = {si}
        buf_ep: List[tuple] = []
        done     = False

        for step in range(MAX_HOPS):
            progress  = step / MAX_HOPS
            state     = _make_state(emb, current, di, progress)
            neighbors = [n for n in adj[current] if n not in visited]

            if not neighbors:
                for exp in buf_ep:
                    replay.push(*exp)
                break

            # ε-greedy
            if rng.random() < epsilon:
                nxt = rng.choice(neighbors)
            else:
                with torch.no_grad():
                    q = policy.q_values(emb, current, di, neighbors, progress)
                nxt = neighbors[q.argmax().item()]

            reward = 10.0 if nxt == di else -0.1
            done   = nxt == di

            np1    = (step + 1) / MAX_HOPS
            ns     = _make_state(emb, nxt, di, np1)
            nxt_nb = [n for n in adj[nxt] if n not in visited | {nxt}]
            nc     = [emb[n].detach() for n in nxt_nb]

            buf_ep.append((
                state.detach(),
                emb[nxt].detach(),
                torch.tensor([reward], dtype=torch.float32),
                ns.detach(),
                nc,
                torch.tensor([float(done)], dtype=torch.float32),
            ))

            visited.add(nxt)
            current = nxt
            if done:
                wins += 1
                for exp in buf_ep:
                    replay.push(*exp)
                break
        else:
            losses += 1
            for exp in buf_ep:
                replay.push(*exp)

        step_n  += 1
        epsilon  = max(EPS_END, epsilon * EPS_DECAY)

        # --- training step ---
        if len(replay) >= MIN_REPLAY:
            batch = replay.sample(BATCH_SIZE)

            with torch.no_grad():
                targets = []
                for ex in batch:
                    if ex.done.item() or not ex.next_cands:
                        targets.append(ex.reward)
                    else:
                        nc_t = torch.stack(ex.next_cands)
                        nq   = target.dqn(ex.next_state, nc_t)
                        targets.append(ex.reward + GAMMA * nq.max())

            target_t = torch.stack(targets).squeeze()

            preds = torch.stack([
                policy.dqn(ex.state, ex.action_emb.unsqueeze(0)).squeeze()
                for ex in batch
            ])

            loss = F.smooth_l1_loss(preds, target_t.detach())
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()

            if step_n % TARGET_UPDATE == 0:
                target.load_state_dict(policy.state_dict())

        if ep % 500 == 0:
            total = wins + losses
            rate  = wins / total * 100 if total else 0.0
            print(f"ep {ep:5d}  eps={epsilon:.3f}  win={rate:.1f}%  replay={len(replay)}")
            wins = losses = 0

    torch.save(policy.state_dict(), save_path)
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=5_000)
    parser.add_argument("--save",     type=str, default=SAVE_PATH)
    args = parser.parse_args()
    train(args.episodes, args.save)
