"""
FastAPI server for the ST-GNN + DQN routing model.

Endpoints:
  POST /route   — compute a route for the given graph + source/target
  GET  /health  — liveness probe; reports whether the model is loaded

If the model weights file is missing the server still starts, but /route
returns HTTP 503. recommendation_engine treats 503 as a signal to fall
back to Dijkstra.
"""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import List, Optional

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from model import RoutingModel, build_graph_tensors

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("model_server")

MODEL_PATH = os.getenv(
    "MODEL_PATH",
    os.path.join(os.path.dirname(__file__), "saved_model", "routing_model.pt"),
)

_model: Optional[RoutingModel] = None


def _load_model() -> None:
    global _model
    if not os.path.exists(MODEL_PATH):
        logger.warning("Model weights not found at %s — /route will return 503", MODEL_PATH)
        return
    m = RoutingModel()
    m.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    m.eval()
    _model = m
    logger.info("Model loaded from %s", MODEL_PATH)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    _load_model()
    yield


app = FastAPI(title="Mesh Routing Model Server", version="0.1.0", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------

class EdgeIn(BaseModel):
    from_node: str = ""
    to_node:   str = ""
    snr:       float = 5.0


class RouteRequest(BaseModel):
    nodes:  List[str]
    edges:  List[dict]   # raw dicts — key "from" is a Python keyword, easier as dict
    source: str
    target: str


class RouteResponse(BaseModel):
    path:   Optional[List[str]]
    score:  float
    method: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/route", response_model=RouteResponse)
def route(req: RouteRequest) -> RouteResponse:
    if req.source not in req.nodes:
        raise HTTPException(400, f"Source '{req.source}' not in nodes")
    if req.target not in req.nodes:
        raise HTTPException(400, f"Target '{req.target}' not in nodes")

    if req.source == req.target:
        return RouteResponse(path=[req.source], score=1.0, method="ml")

    if _model is None:
        raise HTTPException(503, "Model not loaded — run train.py first")

    nf, ei, ew, n2i = build_graph_tensors(req.nodes, req.edges)
    si, di          = n2i[req.source], n2i[req.target]

    with torch.no_grad():
        idx_path = _model.route(nf, ei, ew, si, di)

    if idx_path is None:
        raise HTTPException(404, "No path found by model")

    i2n  = {v: k for k, v in n2i.items()}
    path = [i2n[i] for i in idx_path]

    # validate path endpoints (safety check)
    if path[0] != req.source or path[-1] != req.target:
        raise HTTPException(500, "Model returned malformed path")

    score = round(1.0 / (1.0 + len(path)), 3)
    return RouteResponse(path=path, score=score, method="ml")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": _model is not None}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
