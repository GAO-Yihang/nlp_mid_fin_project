import torch
from pathlib import Path
from typing import Optional, Dict, Any

def save_checkpoint(path: str, model, optimizer=None, scheduler=None, meta: Optional[Dict[str, Any]] = None):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    state = {
        "model": model.state_dict(),
        "meta": meta or {}
    }
    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        state["scheduler"] = scheduler.state_dict()
    torch.save(state, path)

def load_checkpoint(path: str, model, optimizer=None, scheduler=None, map_location="cpu"):
    state = torch.load(path, map_location=map_location)
    model.load_state_dict(state["model"])
    if optimizer is not None and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None and "scheduler" in state:
        scheduler.load_state_dict(state["scheduler"])
    return state.get("meta", {})
