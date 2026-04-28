"""
ST-GNN + DQN routing model.

STGNNEncoder:   message-passing GNN that encodes graph topology → node embeddings.
DQNHead:        maps (state, candidate_embedding) → Q-value.
RoutingModel:   combines both; exposes .route() for greedy inference.

Node features (3 dims): [degree_norm, avg_snr_norm, is_server]
Edge features:          SNR values, normalised to [0, 1] by dividing by 15.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


HIDDEN_DIM    = 64
NODE_FEAT_DIM = 3   # degree_norm, avg_snr_norm, is_server
MAX_SNR       = 15.0


class STGNNEncoder(nn.Module):
    """2-layer spatial GNN using SNR-weighted message passing."""

    def __init__(self, node_feat_dim: int = NODE_FEAT_DIM, hidden_dim: int = HIDDEN_DIM,
                 num_layers: int = 2):
        super().__init__()
        self.num_layers  = num_layers
        self.input_proj  = nn.Linear(node_feat_dim, hidden_dim)
        # each layer: concat(src, dst, edge_snr) → message
        self.msg_layers  = nn.ModuleList([
            nn.Linear(hidden_dim * 2 + 1, hidden_dim) for _ in range(num_layers)
        ])
        self.norms       = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])

    def forward(
        self,
        node_feats:   torch.Tensor,  # [N, F]
        edge_index:   torch.Tensor,  # [2, E]  long
        edge_weights: torch.Tensor,  # [E]     float, normalised SNR
    ) -> torch.Tensor:               # [N, H]
        N = node_feats.shape[0]
        x = F.relu(self.input_proj(node_feats))           # [N, H]

        for layer, norm in zip(self.msg_layers, self.norms):
            if edge_index.shape[1] == 0:                  # graph with no edges
                break
            src, dst = edge_index[0], edge_index[1]
            w   = edge_weights.unsqueeze(1)               # [E, 1]
            msg = F.relu(layer(torch.cat([x[src], x[dst], w], dim=1)))  # [E, H]

            agg = torch.zeros(N, msg.shape[1], device=x.device)
            agg.index_add_(0, dst, msg)

            x = norm(x + agg)                             # residual

        return x                                          # [N, H]


class DQNHead(nn.Module):
    """Q-value head: (state ∥ candidate_emb) → scalar."""

    def __init__(self, embedding_dim: int = HIDDEN_DIM, hidden_dim: int = 128):
        super().__init__()
        state_dim = embedding_dim * 2 + 1               # curr + dest + progress
        self.net  = nn.Sequential(
            nn.Linear(state_dim + embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        """
        state:      [state_dim]
        candidates: [K, embedding_dim]
        returns:    [K]  Q-values
        """
        K  = candidates.shape[0]
        s  = state.unsqueeze(0).expand(K, -1)           # [K, state_dim]
        return self.net(torch.cat([s, candidates], dim=1)).squeeze(-1)


class RoutingModel(nn.Module):
    """Full ST-GNN + DQN routing model."""

    def __init__(self, hidden_dim: int = HIDDEN_DIM):
        super().__init__()
        self.encoder = STGNNEncoder(NODE_FEAT_DIM, hidden_dim)
        self.dqn     = DQNHead(hidden_dim)

    def encode(self, node_feats, edge_index, edge_weights) -> torch.Tensor:
        return self.encoder(node_feats, edge_index, edge_weights)

    def q_values(
        self,
        embeddings:    torch.Tensor,
        current_idx:   int,
        dest_idx:      int,
        neighbor_idxs: List[int],
        progress:      float = 0.0,
    ) -> torch.Tensor:
        curr = embeddings[current_idx]
        dest = embeddings[dest_idx]
        prog = torch.tensor([progress], dtype=torch.float32, device=embeddings.device)
        state = torch.cat([curr, dest, prog])             # [state_dim]
        cands = embeddings[neighbor_idxs]                 # [K, H]
        return self.dqn(state, cands)

    @torch.no_grad()
    def route(
        self,
        node_feats:   torch.Tensor,
        edge_index:   torch.Tensor,
        edge_weights: torch.Tensor,
        source_idx:   int,
        dest_idx:     int,
        max_hops:     int = 20,
    ) -> Optional[List[int]]:
        """Greedy routing. Returns list of node indices or None if no path."""
        self.eval()
        emb = self.encode(node_feats, edge_index, edge_weights)

        # Build undirected adjacency list from (already undirected) edge_index
        N   = node_feats.shape[0]
        adj: Dict[int, List[int]] = {i: [] for i in range(N)}
        if edge_index.shape[1] > 0:
            src_np = edge_index[0].tolist()
            dst_np = edge_index[1].tolist()
            for s, d in zip(src_np, dst_np):
                adj[s].append(d)

        path    = [source_idx]
        visited = {source_idx}
        current = source_idx

        for step in range(max_hops):
            if current == dest_idx:
                return path
            neighbors = [n for n in adj[current] if n not in visited]
            if not neighbors:
                return None                               # dead end

            q = self.q_values(emb, current, dest_idx, neighbors, step / max_hops)
            best = neighbors[q.argmax().item()]
            path.append(best)
            visited.add(best)
            current = best

        return None                                       # exceeded max_hops


# ---------------------------------------------------------------------------
# Graph tensor helpers (shared by train.py and serve.py)
# ---------------------------------------------------------------------------

def build_graph_tensors(
    nodes: List[str],
    edges: List[dict],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, int]]:
    """
    Convert API graph dicts to PyTorch tensors.

    edges elements: {"from": str, "to": str, "snr": float}

    Returns (node_feats [N,3], edge_index [2,E], edge_weights [E], node_to_idx).
    Edges are made undirected (each pair duplicated).
    """
    node_to_idx = {n: i for i, n in enumerate(nodes)}
    N           = len(nodes)
    degree      = [0]   * N
    snr_sum     = [0.0] * N
    snr_cnt     = [0]   * N

    valid: List[Tuple[int, int, float]] = []
    for e in edges:
        fn, tn = e.get("from", ""), e.get("to", "")
        if fn not in node_to_idx or tn not in node_to_idx:
            continue
        fi, ti = node_to_idx[fn], node_to_idx[tn]
        snr = float(e.get("snr", 5.0))
        valid.append((fi, ti, snr))
        for idx in (fi, ti):
            degree[idx]  += 1
            snr_sum[idx] += snr
            snr_cnt[idx] += 1

    max_deg = max(degree) if any(degree) else 1
    feats   = []
    for i, n in enumerate(nodes):
        deg  = degree[i] / max_deg
        asnr = (snr_sum[i] / snr_cnt[i] / MAX_SNR) if snr_cnt[i] > 0 else 0.0
        srv  = 1.0 if n == "SERVER" else 0.0
        feats.append([deg, asnr, srv])

    node_feats_t = torch.tensor(feats, dtype=torch.float32)

    if not valid:
        return node_feats_t, torch.zeros((2, 0), dtype=torch.long), torch.zeros(0), node_to_idx

    srcs, dsts, snrs = zip(*valid)
    # make undirected
    all_src = list(srcs) + list(dsts)
    all_dst = list(dsts) + list(srcs)
    all_snr = list(snrs) + list(snrs)

    edge_index   = torch.tensor([all_src, all_dst], dtype=torch.long)
    edge_weights = torch.tensor(all_snr, dtype=torch.float32) / MAX_SNR

    return node_feats_t, edge_index, edge_weights, node_to_idx
