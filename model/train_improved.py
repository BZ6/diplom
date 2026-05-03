from __future__ import annotations

import argparse
import collections
import copy
import heapq
import os
import random

import torch
import torch.nn.functional as F

from model import RoutingModel, build_graph_tensors

SAVE_DIR      = os.path.join(os.path.dirname(__file__), "saved_model")
SAVE_PATH     = os.path.join(SAVE_DIR, "routing_model.pt")
SAVE_PATH_GNN = os.path.join(SAVE_DIR, "routing_model_gnn.pt")

_W = lambda snr: max(0.1, 10.0 - float(snr))

CURRICULUM = [
    (4,   12,  0.20),
    (8,   20,  0.20),
    (15,  40,  0.25),
    (30,  70,  0.20),
    (50, 100,  0.15),
]

GAMMA         = 0.95
LR            = 5e-4
BATCH_SIZE    = 64
REPLAY_CAP    = 15_000
MIN_REPLAY    = 500
EPS_START_RL  = 0.15
EPS_END       = 0.02
EPS_DECAY     = 0.9997
TARGET_UPDATE = 300
MAX_HOPS      = 20

Experience = collections.namedtuple(
    "Experience",
    ["state", "action_emb", "reward", "next_state", "next_cands", "done"],
)


def _make_graph(rng, nodes, extra_edges=0):
    seen = set()
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


def _build_adj_weighted(nodes, edges):
    n2i = {n: i for i, n in enumerate(nodes)}
    N   = len(nodes)
    adj = {i: [] for i in range(N)}
    for e in edges:
        fi = n2i.get(e.get("from", ""))
        ti = n2i.get(e.get("to", ""))
        if fi is None or ti is None:
            continue
        w = _W(e.get("snr", 5.0))
        adj[fi].append((w, ti))
        adj[ti].append((w, fi))
    return adj, n2i


def _build_adj_unweighted(nodes, edges, n2i):
    N   = len(nodes)
    adj = {i: [] for i in range(N)}
    for e in edges:
        fi = n2i.get(e.get("from", ""))
        ti = n2i.get(e.get("to", ""))
        if fi is None or ti is None:
            continue
        adj[fi].append(ti)
        adj[ti].append(fi)
    return adj


def dijkstra_from(adj_w, start, N):
    dist = [float("inf")] * N
    dist[start] = 0.0
    heap = [(0.0, start)]
    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u]:
            continue
        for w, v in adj_w[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(heap, (nd, v))
    return dist


def dijkstra_path_indices(adj_w, n2i, src_id, dst_id):
    if src_id not in n2i or dst_id not in n2i:
        return None, None
    N  = len(n2i)
    si = n2i[src_id]
    di = n2i[dst_id]
    dist = [float("inf")] * N
    prev = [-1] * N
    dist[si] = 0.0
    heap = [(0.0, si)]
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
    if dist[di] == float("inf"):
        return None, None
    path, cur = [], di
    while cur != -1:
        path.append(cur)
        cur = prev[cur]
    path.reverse()
    return path, dist[di]


def soft_labels_from_dist(dist_to_dest, neighbor_idxs, temperature=2.0):
    costs = torch.tensor([dist_to_dest[n] for n in neighbor_idxs], dtype=torch.float32)
    finite = ~torch.isinf(costs)
    if finite.any():
        costs[~finite] = costs[finite].max() * 3.0
    else:
        return torch.ones(len(neighbor_idxs)) / len(neighbor_idxs)
    return torch.softmax(-costs / temperature, dim=0)


def _make_state(emb, curr, dest, progress):
    prog = torch.tensor([progress], dtype=torch.float32, device=emb.device)
    return torch.cat([emb[curr], emb[dest], prog])


def train_phase1_bc(model, total_episodes, optimizer, rng,
                    device=torch.device("cpu")):
    model.train()
    stage_eps = [(int(total_episodes * f), lo, hi) for lo, hi, f in CURRICULUM]
    ep_total  = 0

    for stage_idx, (n_ep, n_min, n_max) in enumerate(stage_eps):
        wins = losses = 0
        for ep in range(n_ep):
            nodes, edges = gen_graph(rng, n_min, n_max)
            non_srv = [n for n in nodes if n != "SERVER"]
            if len(non_srv) < 2:
                continue

            adj_w, n2i = _build_adj_weighted(nodes, edges)
            adj_u      = _build_adj_unweighted(nodes, edges, n2i)

            src, dst = rng.sample(non_srv, 2)
            path_idx, cost = dijkstra_path_indices(adj_w, n2i, src, dst)
            if path_idx is None or len(path_idx) < 2:
                continue

            di           = n2i[dst]
            dist_to_dest = dijkstra_from(adj_w, di, len(nodes))

            nf, ei, ew, _ = build_graph_tensors(nodes, edges)
            emb = model.encode(nf.to(device), ei.to(device), ew.to(device))

            ep_loss = torch.tensor(0.0)
            n_steps = 0

            for step in range(len(path_idx) - 1):
                curr_idx = path_idx[step]
                next_idx = path_idx[step + 1]
                progress = step / max(len(path_idx) - 1, 1)

                neighbors = adj_u[curr_idx]
                if not neighbors or next_idx not in neighbors:
                    continue

                q        = model.q_values(emb, curr_idx, di, neighbors, progress)
                hard_tgt = torch.tensor([neighbors.index(next_idx)], device=q.device)
                ce_loss  = F.cross_entropy(q.unsqueeze(0), hard_tgt)

                soft    = soft_labels_from_dist(dist_to_dest, neighbors).to(q.device)
                kl_loss = F.kl_div(F.log_softmax(q, dim=0), soft, reduction="sum")

                ep_loss  = ep_loss + 0.6 * ce_loss + 0.4 * kl_loss
                n_steps += 1

            if n_steps > 0 and ep_loss.requires_grad:
                optimizer.zero_grad()
                (ep_loss / n_steps).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                wins += 1
            else:
                losses += 1

        ep_total += n_ep
        rate = wins / max(wins + losses, 1) * 100
        print(f"  [BC stage {stage_idx+1}/5  N={n_min}-{n_max}]  "
              f"ep={n_ep}  trained={rate:.0f}%  total={ep_total}")


class ReplayBuffer:
    def __init__(self, capacity):
        self.buf = collections.deque(maxlen=capacity)

    def push(self, *args):
        self.buf.append(Experience(*args))

    def sample(self, n):
        return random.sample(self.buf, n)

    def __len__(self):
        return len(self.buf)


def train_phase2_rl(model, total_episodes, optimizer, rng,
                    device=torch.device("cpu")):
    target  = copy.deepcopy(model).to(device)
    target.eval()

    replay  = ReplayBuffer(REPLAY_CAP)
    epsilon = EPS_START_RL
    step_n  = 0

    stage_eps = [(int(total_episodes * f), lo, hi) for lo, hi, f in CURRICULUM]
    ep_total  = 0

    for stage_idx, (n_ep, n_min, n_max) in enumerate(stage_eps):
        stage_wins = stage_losses = 0

        for ep in range(n_ep):
            nodes, edges = gen_graph(rng, n_min, n_max)
            non_srv = [n for n in nodes if n != "SERVER"]
            if len(non_srv) < 2:
                continue

            adj_w, n2i = _build_adj_weighted(nodes, edges)
            adj_u      = _build_adj_unweighted(nodes, edges, n2i)

            src, dst = rng.sample(non_srv, 2)
            _, cost  = dijkstra_path_indices(adj_w, n2i, src, dst)
            if cost is None:
                continue

            si, di = n2i[src], n2i[dst]
            nf, ei, ew, _ = build_graph_tensors(nodes, edges)

            with torch.no_grad():
                emb = model.encode(nf.to(device), ei.to(device), ew.to(device))

            w_map = {}
            for e in edges:
                fi = n2i.get(e.get("from", ""))
                ti = n2i.get(e.get("to", ""))
                if fi is None or ti is None:
                    continue
                w = _W(e.get("snr", 5.0))
                w_map[(fi, ti)] = w
                w_map[(ti, fi)] = w

            current = si
            visited = {si}
            buf_ep  = []

            for step in range(MAX_HOPS):
                progress  = step / MAX_HOPS
                state     = _make_state(emb, current, di, progress)
                neighbors = [n for n in adj_u[current] if n not in visited]

                if not neighbors:
                    for exp in buf_ep:
                        replay.push(*exp)
                    break

                if rng.random() < epsilon:
                    nxt = rng.choice(neighbors)
                else:
                    with torch.no_grad():
                        q = model.q_values(emb, current, di, neighbors, progress)
                    nxt = neighbors[q.argmax().item()]

                edge_w = w_map.get((current, nxt), 1.0)
                reward = -edge_w / max(cost, 0.1)
                done   = nxt == di
                if done:
                    reward += 2.0

                ns     = _make_state(emb, nxt, di, (step + 1) / MAX_HOPS)
                nxt_nb = [n for n in adj_u[nxt] if n not in visited | {nxt}]
                nc     = [emb[n].detach() for n in nxt_nb]

                buf_ep.append((
                    state.detach(),
                    emb[nxt].detach(),
                    torch.tensor([reward], dtype=torch.float32, device=device),
                    ns.detach(),
                    nc,
                    torch.tensor([float(done)], dtype=torch.float32, device=device),
                ))

                visited.add(nxt)
                current = nxt
                if done:
                    stage_wins += 1
                    for exp in buf_ep:
                        replay.push(*exp)
                    break
            else:
                stage_losses += 1
                for exp in buf_ep:
                    replay.push(*exp)

            step_n  += 1
            epsilon  = max(EPS_END, epsilon * EPS_DECAY)

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
                    model.dqn(ex.state, ex.action_emb.unsqueeze(0)).squeeze()
                    for ex in batch
                ])
                loss = F.smooth_l1_loss(preds, target_t.detach())
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                if step_n % TARGET_UPDATE == 0:
                    target.load_state_dict(model.state_dict())

        ep_total += n_ep
        rate = stage_wins / max(stage_wins + stage_losses, 1) * 100
        print(f"  [RL stage {stage_idx+1}/5  N={n_min}-{n_max}]  "
              f"ep={n_ep}  win={rate:.0f}%  eps={epsilon:.3f}  total={ep_total}")


def train(bc_episodes=8_000, rl_episodes=2_000,
          save_path=SAVE_PATH, use_global_attn=True):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    rng    = random.Random(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = RoutingModel(use_global_attn=use_global_attn).to(device)
    opt    = torch.optim.Adam(model.parameters(), lr=LR)

    label = "ST-GNN+DQN" if use_global_attn else "GNN+DQN"
    print(f"=== {label}  device={device} ===")
    print(f"Phase 1: Behavioral Cloning ({bc_episodes} ep)")
    train_phase1_bc(model, bc_episodes, opt, rng, device)
    print(f"\nPhase 2: RL fine-tuning ({rl_episodes} ep)")
    train_phase2_rl(model, rl_episodes, opt, rng, device)

    torch.save(model.state_dict(), save_path)
    print(f"\nSaved: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bc",           type=int, default=8_000)
    parser.add_argument("--rl",           type=int, default=2_000)
    parser.add_argument("--save",         type=str, default=SAVE_PATH)
    parser.add_argument("--no-attention", action="store_true")
    args = parser.parse_args()

    if args.no_attention and args.save == SAVE_PATH:
        args.save = SAVE_PATH_GNN

    train(args.bc, args.rl, args.save, use_global_attn=not args.no_attention)
