from __future__ import annotations

import argparse
import collections
import heapq
import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import RoutingModel, build_graph_tensors

SAVE_DIR = os.path.join(os.path.dirname(__file__), "saved_model")

CURRICULUM = [
    (4,   12,  0.20),
    (8,   20,  0.20),
    (15,  40,  0.25),
    (30,  70,  0.20),
    (50, 100,  0.15),
]

GAMMA        = 0.95
LR           = 3e-4
MAX_HOPS     = 20
ROLLOUT_SIZE = 512
PPO_EPOCHS   = 4
PPO_CLIP     = 0.2
LAMBDA       = 0.95
MINIBATCH    = 64

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
    n     = rng.randint(n_min, n_max)
    nodes = [f"n{i}" for i in range(n)]
    if rng.random() < 0.3:
        nodes.append("SERVER")
    extra = rng.randint(n // 4, n // 2)
    return _make_graph(rng, nodes, extra_edges=extra)


def _build_adj(nodes, edges, n2i):
    N   = len(nodes)
    adj = {i: [] for i in range(N)}
    w_map = {}
    for e in edges:
        fi = n2i.get(e.get("from", ""))
        ti = n2i.get(e.get("to", ""))
        if fi is None or ti is None:
            continue
        w = _W(e.get("snr", 5.0))
        adj[fi].append(ti)
        adj[ti].append(fi)
        w_map[(fi, ti)] = w
        w_map[(ti, fi)] = w
    return adj, w_map


def _dijkstra_cost(w_map, n2i, src, dst, N):
    si, di = n2i.get(src), n2i.get(dst)
    if si is None or di is None:
        return None
    wadj = {i: [] for i in range(N)}
    for (s, d), w in w_map.items():
        wadj[s].append((w, d))
    dist = [float("inf")] * N
    dist[si] = 0.0
    heap = [(0.0, si)]
    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u]:
            continue
        for w, v in wadj[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(heap, (nd, v))
    return dist[di] if dist[di] < float("inf") else None


def _make_state(emb, curr, dest, progress):
    prog = torch.tensor([progress], dtype=torch.float32, device=emb.device)
    return torch.cat([emb[curr], emb[dest], prog])


class ValueNet(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, state):
        return self.net(state).squeeze(-1)


def train_reinforce(model, total_episodes, optimizer, rng,
                    device=torch.device("cpu")):
    model.train()
    stage_eps = [(int(total_episodes * f), lo, hi) for lo, hi, f in CURRICULUM]
    ep_total  = 0

    for stage_idx, (n_ep, n_min, n_max) in enumerate(stage_eps):
        wins = losses = skipped = 0

        for ep in range(n_ep):
            nodes, edges = gen_graph(rng, n_min, n_max)
            non_srv = [n for n in nodes if n != "SERVER"]
            if len(non_srv) < 2:
                skipped += 1
                continue

            n2i = {n: i for i, n in enumerate(nodes)}
            N   = len(nodes)
            adj, w_map = _build_adj(nodes, edges, n2i)

            src, dst = rng.sample(non_srv, 2)
            opt_cost = _dijkstra_cost(w_map, n2i, src, dst, N)
            if opt_cost is None:
                skipped += 1
                continue

            si, di = n2i[src], n2i[dst]
            nf, ei, ew, _ = build_graph_tensors(nodes, edges)
            emb = model.encode(nf.to(device), ei.to(device), ew.to(device))

            visited   = {si}
            current   = si
            log_probs = []
            rewards   = []

            for step in range(MAX_HOPS):
                neighbors = [n for n in adj[current] if n not in visited]
                if not neighbors:
                    break

                q          = model.q_values(emb, current, di, neighbors, step / MAX_HOPS)
                probs      = F.softmax(q, dim=0)
                action_idx = torch.multinomial(probs, 1).item()
                log_prob   = torch.log(probs[action_idx] + 1e-9)

                nxt    = neighbors[action_idx]
                done   = nxt == di
                reward = -w_map.get((current, nxt), 1.0) / max(opt_cost, 0.1)
                if done:
                    reward += 2.0

                log_probs.append(log_prob)
                rewards.append(reward)
                visited.add(nxt)
                current = nxt
                if done:
                    wins += 1
                    break
            else:
                losses += 1

            if not log_probs:
                skipped += 1
                continue

            G, returns = 0.0, []
            for r in reversed(rewards):
                G = r + GAMMA * G
                returns.insert(0, G)
            returns_t = torch.tensor(returns, dtype=torch.float32, device=device)
            if returns_t.std() > 1e-6:
                returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)

            loss = -torch.stack(log_probs).dot(returns_t)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        ep_total += n_ep
        rate = wins / max(wins + losses, 1) * 100
        print(f"  [REINFORCE stage {stage_idx+1}/5  N={n_min}-{n_max}]  "
              f"ep={n_ep}  win={rate:.0f}%  total={ep_total}")


PPOExp = collections.namedtuple(
    "PPOExp",
    ["state", "cands", "action_idx", "log_prob_old", "value_old", "reward", "done"],
)


def _collect_rollout(model, value_net, rng, n_min, n_max, rollout_sz,
                     device=torch.device("cpu")):
    model.eval()
    value_net.eval()
    exps = []

    while len(exps) < rollout_sz:
        nodes, edges = gen_graph(rng, n_min, n_max)
        non_srv = [n for n in nodes if n != "SERVER"]
        if len(non_srv) < 2:
            continue

        n2i = {n: i for i, n in enumerate(nodes)}
        N   = len(nodes)
        adj, w_map = _build_adj(nodes, edges, n2i)

        src, dst = rng.sample(non_srv, 2)
        opt_cost = _dijkstra_cost(w_map, n2i, src, dst, N)
        if opt_cost is None:
            continue

        si, di = n2i[src], n2i[dst]
        nf, ei, ew, _ = build_graph_tensors(nodes, edges)

        with torch.no_grad():
            emb = model.encode(nf.to(device), ei.to(device), ew.to(device))

        visited = {si}
        current = si

        for step in range(MAX_HOPS):
            neighbors = [n for n in adj[current] if n not in visited]
            if not neighbors:
                break

            state = _make_state(emb, current, di, step / MAX_HOPS)

            with torch.no_grad():
                q        = model.q_values(emb, current, di, neighbors, step / MAX_HOPS)
                probs    = F.softmax(q, dim=0)
                action_i = torch.multinomial(probs, 1).item()
                log_prob = torch.log(probs[action_i] + 1e-9)
                value    = value_net(state)

            nxt    = neighbors[action_i]
            done   = nxt == di
            reward = -w_map.get((current, nxt), 1.0) / max(opt_cost, 0.1) + (2.0 if done else 0.0)

            exps.append(PPOExp(
                state.detach(), emb[neighbors].detach(), action_i,
                log_prob.detach(), value.detach(),
                torch.tensor(reward, dtype=torch.float32),
                done,
            ))

            visited.add(nxt)
            current = nxt
            if done or len(exps) >= rollout_sz:
                break

    return exps[:rollout_sz]


def _compute_gae(exps):
    T          = len(exps)
    advantages = torch.zeros(T)
    returns    = torch.zeros(T)
    gae        = 0.0

    for t in reversed(range(T)):
        r      = exps[t].reward.item()
        v      = exps[t].value_old.item()
        done   = exps[t].done
        v_next = 0.0 if (done or t == T - 1) else exps[t + 1].value_old.item()
        delta  = r + GAMMA * v_next - v
        gae    = delta + GAMMA * LAMBDA * (0.0 if done else gae)
        advantages[t] = gae
        returns[t]    = gae + v

    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    return advantages, returns


def train_ppo(model, value_net, total_steps, policy_opt, value_opt, rng,
              device=torch.device("cpu")):
    stage_steps = [(int(total_steps * f), lo, hi) for lo, hi, f in CURRICULUM]
    step_total  = 0

    for stage_idx, (n_steps, n_min, n_max) in enumerate(stage_steps):
        collected    = 0
        update_count = 0

        while collected < n_steps:
            exps = _collect_rollout(model, value_net, rng, n_min, n_max, ROLLOUT_SIZE, device)
            collected += len(exps)

            advantages, returns = _compute_gae(exps)
            advantages    = advantages.to(device)
            returns       = returns.to(device)
            log_probs_old = torch.stack([e.log_prob_old for e in exps])

            model.train()
            value_net.train()

            for _ in range(PPO_EPOCHS):
                idxs = torch.randperm(len(exps))
                for start in range(0, len(exps), MINIBATCH):
                    mb = idxs[start:start + MINIBATCH]
                    if len(mb) < 2:
                        continue

                    p_losses, v_losses = [], []
                    for i in mb.tolist():
                        e         = exps[i]
                        q         = model.dqn(e.state, e.cands)
                        probs_new = F.softmax(q, dim=0)
                        log_p_new = torch.log(probs_new[e.action_idx] + 1e-9)

                        ratio = torch.exp(log_p_new - log_probs_old[i])
                        adv   = advantages[i]
                        surr  = torch.min(ratio * adv,
                                          torch.clamp(ratio, 1 - PPO_CLIP, 1 + PPO_CLIP) * adv)
                        p_losses.append(-surr)
                        v_losses.append(F.mse_loss(value_net(e.state), returns[i]))

                    loss = torch.stack(p_losses).mean() + 0.5 * torch.stack(v_losses).mean()
                    policy_opt.zero_grad()
                    value_opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                    torch.nn.utils.clip_grad_norm_(value_net.parameters(), 0.5)
                    policy_opt.step()
                    value_opt.step()

            update_count += 1

        step_total += n_steps
        print(f"  [PPO stage {stage_idx+1}/5  N={n_min}-{n_max}]  "
              f"steps={n_steps}  updates={update_count}  total={step_total}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ppo",      action="store_true")
    parser.add_argument("--episodes", type=int, default=8_000)
    args = parser.parse_args()

    os.makedirs(SAVE_DIR, exist_ok=True)
    rng    = random.Random(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = RoutingModel(encoder_type='sage').to(device)

    if args.ppo:
        save_path  = os.path.join(SAVE_DIR, "routing_model_sage_ppo.pt")
        value_net  = ValueNet(64 * 2 + 1).to(device)
        policy_opt = torch.optim.Adam(model.parameters(),     lr=LR)
        value_opt  = torch.optim.Adam(value_net.parameters(), lr=LR)
        print(f"=== GraphSAGE+PPO  {args.episodes} steps  device={device} ===")
        train_ppo(model, value_net, args.episodes, policy_opt, value_opt, rng, device)
    else:
        save_path = os.path.join(SAVE_DIR, "routing_model_sage_reinforce.pt")
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        print(f"=== GraphSAGE+REINFORCE  {args.episodes} ep  device={device} ===")
        train_reinforce(model, args.episodes, optimizer, rng, device)

    torch.save(model.state_dict(), save_path)
    print(f"\nSaved: {save_path}")


if __name__ == "__main__":
    main()
