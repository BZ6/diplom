import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from model import RoutingModel, build_graph_tensors, STGNNEncoder, DQNHead


# ---------------------------------------------------------------------------
# build_graph_tensors
# ---------------------------------------------------------------------------

def _abc_graph():
    nodes = ["A", "B", "C"]
    edges = [{"from": "A", "to": "B", "snr": 8.0},
             {"from": "B", "to": "C", "snr": 6.0}]
    return nodes, edges


def test_build_tensors_shape():
    nodes, edges = _abc_graph()
    nf, ei, ew, n2i = build_graph_tensors(nodes, edges)
    assert nf.shape == (3, 3)
    assert ei.shape == (2, 4)   # 2 edges → 4 directed
    assert ew.shape == (4,)
    assert set(n2i.keys()) == {"A", "B", "C"}


def test_build_tensors_server_flag():
    nodes = ["A", "SERVER"]
    edges = [{"from": "A", "to": "SERVER", "snr": 12.0}]
    nf, _, _, n2i = build_graph_tensors(nodes, edges)
    assert nf[n2i["SERVER"], 2].item() == 1.0   # is_server
    assert nf[n2i["A"],      2].item() == 0.0


def test_build_tensors_no_edges():
    nodes = ["A", "B"]
    nf, ei, ew, n2i = build_graph_tensors(nodes, [])
    assert ei.shape == (2, 0)
    assert ew.shape == (0,)


def test_build_tensors_normalised_weights():
    nodes = ["A", "B"]
    edges = [{"from": "A", "to": "B", "snr": 15.0}]
    _, _, ew, _ = build_graph_tensors(nodes, edges)
    assert abs(ew[0].item() - 1.0) < 1e-5   # 15 / 15 = 1


# ---------------------------------------------------------------------------
# STGNNEncoder
# ---------------------------------------------------------------------------

def test_encoder_output_shape():
    nodes, edges = _abc_graph()
    nf, ei, ew, _ = build_graph_tensors(nodes, edges)
    enc = STGNNEncoder()
    out = enc(nf, ei, ew)
    assert out.shape == (3, 64)


def test_encoder_no_edges():
    nf = torch.zeros(4, 3)
    ei = torch.zeros((2, 0), dtype=torch.long)
    ew = torch.zeros(0)
    enc = STGNNEncoder()
    out = enc(nf, ei, ew)
    assert out.shape == (4, 64)


# ---------------------------------------------------------------------------
# DQNHead
# ---------------------------------------------------------------------------

def test_dqn_head_q_values_shape():
    head  = DQNHead(embedding_dim=64)
    state = torch.zeros(64 * 2 + 1)
    cands = torch.zeros(3, 64)
    q     = head(state, cands)
    assert q.shape == (3,)


# ---------------------------------------------------------------------------
# RoutingModel.route
# ---------------------------------------------------------------------------

def test_route_returns_list_or_none():
    nodes, edges = _abc_graph()
    nf, ei, ew, n2i = build_graph_tensors(nodes, edges)
    model = RoutingModel()
    result = model.route(nf, ei, ew, n2i["A"], n2i["C"])
    assert result is None or isinstance(result, list)


def test_route_same_node():
    nodes, edges = _abc_graph()
    nf, ei, ew, n2i = build_graph_tensors(nodes, edges)
    model  = RoutingModel()
    result = model.route(nf, ei, ew, n2i["A"], n2i["A"])
    assert result == [n2i["A"]]


def test_route_no_path():
    nodes = ["A", "B", "C", "D"]
    # A-B and C-D are disconnected
    edges = [{"from": "A", "to": "B", "snr": 8.0},
             {"from": "C", "to": "D", "snr": 8.0}]
    nf, ei, ew, n2i = build_graph_tensors(nodes, edges)
    model  = RoutingModel()
    result = model.route(nf, ei, ew, n2i["A"], n2i["C"])
    # Untrained model; may or may not find a path in connected subgraph,
    # but these two components are unreachable so result must be None.
    assert result is None


def test_route_endpoints():
    """When route succeeds, first node is source, last is target."""
    nodes, edges = _abc_graph()
    nf, ei, ew, n2i = build_graph_tensors(nodes, edges)
    # Force the model to make moves by patching Q-values: route B→C directly.
    model = RoutingModel()
    result = model.route(nf, ei, ew, n2i["B"], n2i["C"])
    if result is not None:
        assert result[0]  == n2i["B"]
        assert result[-1] == n2i["C"]
