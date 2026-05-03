from __future__ import annotations

import heapq
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


HIDDEN_DIM    = 64
NODE_FEAT_DIM = 3
MAX_SNR       = 15.0


class STGNNEncoder(nn.Module):
    def __init__(self, node_feat_dim=NODE_FEAT_DIM, hidden_dim=HIDDEN_DIM,
                 num_layers=2, num_attn_heads=4, use_global_attn=True):
        super().__init__()
        self.num_layers      = num_layers
        self.use_global_attn = use_global_attn
        self.input_proj      = nn.Linear(node_feat_dim, hidden_dim)
        self.msg_layers      = nn.ModuleList([
            nn.Linear(hidden_dim * 2 + 1, hidden_dim) for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        if use_global_attn:
            self.global_attn = nn.MultiheadAttention(
                embed_dim=hidden_dim, num_heads=num_attn_heads,
                batch_first=True, dropout=0.0,
            )
            self.attn_norm = nn.LayerNorm(hidden_dim)

    def forward(self, node_feats, edge_index, edge_weights):
        N = node_feats.shape[0]
        x = F.relu(self.input_proj(node_feats))

        for layer, norm in zip(self.msg_layers, self.norms):
            if edge_index.shape[1] == 0:
                break
            src, dst = edge_index[0], edge_index[1]
            w   = edge_weights.unsqueeze(1)
            msg = F.relu(layer(torch.cat([x[src], x[dst], w], dim=1)))
            agg = torch.zeros(N, msg.shape[1], device=x.device)
            agg.index_add_(0, dst, msg)
            x = norm(x + agg)

        if self.use_global_attn:
            x_seq = x.unsqueeze(0)
            attn_out, _ = self.global_attn(x_seq, x_seq, x_seq)
            x = self.attn_norm(x + attn_out.squeeze(0))

        return x


class GraphSAGEEncoder(nn.Module):
    def __init__(self, node_feat_dim=NODE_FEAT_DIM, hidden_dim=HIDDEN_DIM, num_layers=2):
        super().__init__()
        self.num_layers  = num_layers
        self.input_proj  = nn.Linear(node_feat_dim, hidden_dim)
        self.sage_layers = nn.ModuleList([
            nn.Linear(hidden_dim * 2, hidden_dim) for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])

    def forward(self, node_feats, edge_index, edge_weights):
        N = node_feats.shape[0]
        x = F.relu(self.input_proj(node_feats))

        has_edges = edge_index.shape[1] > 0
        src = edge_index[0] if has_edges else None
        dst = edge_index[1] if has_edges else None

        for layer, norm in zip(self.sage_layers, self.norms):
            if has_edges:
                w        = edge_weights.unsqueeze(1)
                weighted = x[src] * w
                sum_w    = torch.zeros(N, 1, device=x.device)
                sum_feat = torch.zeros_like(x)
                sum_w.index_add_(0, dst, w)
                sum_feat.index_add_(0, dst, weighted)
                mean_agg = sum_feat / sum_w.clamp(min=1e-8)
            else:
                mean_agg = torch.zeros_like(x)
            x = norm(F.relu(layer(torch.cat([x, mean_agg], dim=1))))

        return x


class DQNHead(nn.Module):
    def __init__(self, embedding_dim=HIDDEN_DIM, hidden_dim=128):
        super().__init__()
        state_dim = embedding_dim * 2 + 1
        self.net  = nn.Sequential(
            nn.Linear(state_dim + embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state, candidates):
        K = candidates.shape[0]
        s = state.unsqueeze(0).expand(K, -1)
        return self.net(torch.cat([s, candidates], dim=1)).squeeze(-1)


class RoutingModel(nn.Module):
    def __init__(self, hidden_dim=HIDDEN_DIM, encoder_type='stgnn', use_global_attn=True):
        super().__init__()
        if encoder_type == 'sage':
            self.encoder = GraphSAGEEncoder(NODE_FEAT_DIM, hidden_dim)
        else:
            self.encoder = STGNNEncoder(NODE_FEAT_DIM, hidden_dim,
                                        use_global_attn=use_global_attn)
        self.dqn = DQNHead(hidden_dim)

    def encode(self, node_feats, edge_index, edge_weights):
        return self.encoder(node_feats, edge_index, edge_weights)

    def q_values(self, embeddings, current_idx, dest_idx, neighbor_idxs, progress=0.0):
        prog  = torch.tensor([progress], dtype=torch.float32, device=embeddings.device)
        state = torch.cat([embeddings[current_idx], embeddings[dest_idx], prog])
        return self.dqn(state, embeddings[neighbor_idxs])

    @staticmethod
    def _build_adj(edge_index, N):
        adj = [[] for _ in range(N)]
        if edge_index.shape[1] > 0:
            for s, d in zip(edge_index[0].tolist(), edge_index[1].tolist()):
                adj[s].append(d)
        return adj

    @property
    def _enc_device(self):
        return next(self.encoder.parameters()).device

    @torch.inference_mode()
    def route(self, node_feats, edge_index, edge_weights, source_idx, dest_idx, max_hops=20):
        self.eval()
        dev = self._enc_device
        emb = self.encoder(node_feats.to(dev), edge_index.to(dev), edge_weights.to(dev))
        if dev.type != "cpu":
            emb = emb.cpu()
        adj = self._build_adj(edge_index, node_feats.shape[0])
        return self._route_from_emb(emb, adj, source_idx, dest_idx, max_hops)

    @torch.inference_mode()
    def encode_graph(self, node_feats, edge_index, edge_weights):
        self.eval()
        dev = self._enc_device
        emb = self.encoder(node_feats.to(dev), edge_index.to(dev), edge_weights.to(dev))
        if dev.type != "cpu":
            emb = emb.cpu()
        adj = self._build_adj(edge_index, node_feats.shape[0])
        return emb, adj

    @torch.inference_mode()
    def route_cached(self, emb, adj, source_idx, dest_idx, max_hops=20):
        return self._route_from_emb(emb, adj, source_idx, dest_idx, max_hops)

    def _route_from_emb(self, emb, adj, source_idx, dest_idx, max_hops=20):
        N, H = emb.shape
        visited = [False] * N
        visited[source_idx] = True

        state = torch.empty(H * 2 + 1, dtype=torch.float32)
        state[H: 2 * H].copy_(emb[dest_idx])
        inv_max = 1.0 / max_hops

        path    = [source_idx]
        current = source_idx

        for step in range(max_hops):
            if current == dest_idx:
                return path
            neighbors = [n for n in adj[current] if not visited[n]]
            if not neighbors:
                return None
            state[:H].copy_(emb[current])
            state[-1] = step * inv_max
            q    = self.dqn(state, emb[neighbors])
            best = neighbors[q.argmax().item()]
            path.append(best)
            visited[best] = True
            current = best

        return None


class NeuralDijkstraModel(nn.Module):
    def __init__(self, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.encoder = GraphSAGEEncoder(NODE_FEAT_DIM, hidden_dim)
        self.edge_scorer = nn.Sequential(
            nn.Linear(hidden_dim * 3, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    @property
    def _enc_device(self):
        return next(self.encoder.parameters()).device

    def predict_scores(self, emb, edge_index, dest_idx):
        if edge_index.shape[1] == 0:
            return torch.zeros(0)
        src, dst = edge_index[0], edge_index[1]
        h_dest = emb[dest_idx].unsqueeze(0).expand(edge_index.shape[1], -1)
        feats  = torch.cat([emb[src], emb[dst], h_dest], dim=1)
        return self.edge_scorer(feats).squeeze(-1)

    @torch.inference_mode()
    def encode_graph(self, node_feats, edge_index, edge_weights):
        self.eval()
        dev = self._enc_device
        emb = self.encoder(node_feats.to(dev), edge_index.to(dev), edge_weights.to(dev))
        if dev.type != "cpu":
            emb = emb.cpu()
        return emb, edge_index.cpu(), edge_weights.cpu()

    @torch.inference_mode()
    def route_cached(self, emb, edge_index, edge_weights, src_idx, dst_idx):
        N = emb.shape[0]
        if edge_index.shape[1] == 0:
            return None if src_idx != dst_idx else [src_idx]

        scores = self.predict_scores(emb, edge_index, dst_idx)
        mod_w  = (edge_weights * (2.0 - scores)).tolist()
        src_l  = edge_index[0].tolist()
        dst_l  = edge_index[1].tolist()

        adj = [[] for _ in range(N)]
        for i, (s, d) in enumerate(zip(src_l, dst_l)):
            adj[s].append((mod_w[i], d))

        dist = [float("inf")] * N
        prev = [-1] * N
        dist[src_idx] = 0.0
        heap = [(0.0, src_idx)]

        while heap:
            d, u = heapq.heappop(heap)
            if d > dist[u]:
                continue
            for w, v in adj[u]:
                nd = d + w
                if nd < dist[v]:
                    dist[v] = nd
                    prev[v] = u
                    heapq.heappush(heap, (nd, v))

        if dist[dst_idx] == float("inf"):
            return None

        path, cur = [], dst_idx
        while cur != -1:
            path.append(cur)
            cur = prev[cur]
        path.reverse()
        return path

    @torch.inference_mode()
    def route(self, node_feats, edge_index, edge_weights, src_idx, dst_idx):
        emb, ei, ew = self.encode_graph(node_feats, edge_index, edge_weights)
        return self.route_cached(emb, ei, ew, src_idx, dst_idx)


def build_graph_tensors(nodes, edges):
    n2i = {n: i for i, n in enumerate(nodes)}
    N   = len(nodes)

    srcs, dsts, snrs = [], [], []
    for e in edges:
        fi = n2i.get(e.get("from", ""), -1)
        ti = n2i.get(e.get("to",   ""), -1)
        if fi < 0 or ti < 0:
            continue
        srcs.append(fi)
        dsts.append(ti)
        snrs.append(float(e.get("snr", 5.0)))

    if not srcs:
        feats = np.zeros((N, NODE_FEAT_DIM), dtype=np.float32)
        feats[:, 2] = [1.0 if nd == "SERVER" else 0.0 for nd in nodes]
        return (torch.from_numpy(feats),
                torch.zeros((2, 0), dtype=torch.long),
                torch.zeros(0, dtype=torch.float32),
                n2i)

    sa = np.array(srcs, dtype=np.int64)
    da = np.array(dsts, dtype=np.int64)
    wa = np.array(snrs, dtype=np.float32)

    degree  = np.zeros(N, dtype=np.float32)
    snr_sum = np.zeros(N, dtype=np.float32)
    snr_cnt = np.zeros(N, dtype=np.float32)
    np.add.at(degree,  sa, 1); np.add.at(degree,  da, 1)
    np.add.at(snr_sum, sa, wa); np.add.at(snr_sum, da, wa)
    np.add.at(snr_cnt, sa, 1); np.add.at(snr_cnt, da, 1)

    max_deg  = degree.max() if degree.any() else 1.0
    srv_flag = np.array([1.0 if nd == "SERVER" else 0.0 for nd in nodes], dtype=np.float32)
    feats = np.column_stack([
        degree / max_deg,
        np.where(snr_cnt > 0, snr_sum / snr_cnt / MAX_SNR, 0.0),
        srv_flag,
    ])

    all_src = np.concatenate([sa, da])
    all_dst = np.concatenate([da, sa])
    all_snr = np.concatenate([wa, wa])

    return (
        torch.from_numpy(feats),
        torch.from_numpy(np.stack([all_src, all_dst])),
        torch.from_numpy(all_snr / MAX_SNR),
        n2i,
    )
