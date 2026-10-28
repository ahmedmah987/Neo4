"""Load a qwen4_exp checkpoint without its ~95 GiB n-gram table in memory.

The table (layer 1's PLE n-gram embedding, saved as `ngram_embedding.shard_k.weight` pieces in files of
their own) is left out of the load and read row by row from memory-mapped files instead; everything
else loads where the device map says.
"""

import json
import os

import numpy as np
import torch
from torch import nn

SHARD = "ngram_embedding.shard_"
DTYPES = {"BF16": (np.uint16, torch.bfloat16), "F16": (np.float16, torch.float16), "F32": (np.float32, torch.float32)}


def lazy_dir(src: str, dst: str) -> dict:
    """A copy of the model directory (symlinks) whose index leaves out the files holding only table pieces.
    Returns {module prefix: [(shard number, file), ...]} for the pieces left out."""
    os.makedirs(dst, exist_ok=True)
    index = json.load(open(os.path.join(src, "model.safetensors.index.json")))
    weight_map = index["weight_map"]
    by_file: dict = {}
    for key, f in weight_map.items():
        by_file.setdefault(f, []).append(key)
    table_files = {f for f, keys in by_file.items() if all(SHARD in k for k in keys)}
    mixed = [f for f, keys in by_file.items() if f not in table_files and any(SHARD in k for k in keys)]
    if mixed:
        raise SystemExit(f"table pieces share files with other weights: {mixed[:3]}")
    pieces: dict = {}
    for key, f in weight_map.items():
        if SHARD in key:
            prefix, rest = key.split(SHARD)
            pieces.setdefault(prefix, []).append((int(rest.split(".")[0]), os.path.join(src, f)))
    index["weight_map"] = {k: f for k, f in weight_map.items() if f not in table_files}
    for name in os.listdir(src):
        target = os.path.join(dst, name)
        if name == "model.safetensors.index.json" or name in table_files or os.path.lexists(target):
            continue
        os.symlink(os.path.join(src, name), target)
    json.dump(index, open(os.path.join(dst, "model.safetensors.index.json"), "w"))
    return {p: sorted(v) for p, v in pieces.items()}


def patch_class(cls) -> None:
    """Build the n-gram embedding as a one-row placeholder (the real table is attached after loading)."""
    if getattr(cls, "_lazy_patched", False):
        return
    original = cls.__init__

    def init(self, *args, **kwargs):
        original(self, *args, **kwargs)
        rows, dim = self.ngram_embedding.num_embeddings, self.ngram_embedding.embedding_dim
        self.ngram_embedding = nn.Embedding(1, dim)
        self.table_rows = rows

    cls.__init__ = init
    cls._lazy_patched = True


def _open(path: str, key: str):
    with open(path, "rb") as fh:
        n = int.from_bytes(fh.read(8), "little")
        header = json.loads(fh.read(n))
    meta = header[key]
    np_dtype, torch_dtype = DTYPES[meta["dtype"]]
    start = 8 + n + meta["data_offsets"][0]
    return np.memmap(path, dtype=np_dtype, mode="r", offset=start, shape=tuple(meta["shape"])), torch_dtype


class MappedRows(nn.Module):
    """Embedding rows read from memory-mapped pieces of one table (concatenated along rows, in order)."""

    def __init__(self, pieces: list, key_prefix: str):
        super().__init__()
        self.maps, dtype = [], None
        for number, path in pieces:
            arr, dtype = _open(path, f"{key_prefix}{SHARD}{number}.weight")
            self.maps.append(arr)
        sizes = np.array([m.shape[0] for m in self.maps], dtype=np.int64)
        self.starts = np.concatenate([[0], np.cumsum(sizes)[:-1]])
        self.rows = int(sizes.sum())
        self.dim = self.maps[0].shape[1]
        self.dtype = dtype
        self.weight = nn.Parameter(torch.zeros(1, self.dim, dtype=dtype), requires_grad=False)  # where ids go

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        flat = ids.reshape(-1).cpu().numpy()
        which = np.searchsorted(self.starts, flat, side="right") - 1
        out = np.empty((flat.size, self.dim), dtype=self.maps[0].dtype)
        for k in np.unique(which):
            sel = which == k
            out[sel] = self.maps[k][flat[sel] - self.starts[k]]
        t = torch.from_numpy(out)
        if self.dtype == torch.bfloat16:
            t = t.view(torch.bfloat16)
        return t.reshape(*ids.shape, self.dim)


def attach(model: nn.Module, pieces: dict) -> int:
    """Put the memory-mapped tables in place of the placeholders; returns the number of rows attached."""
    total = 0
    for name, module in model.named_modules():
        if hasattr(module, "table_rows"):
            prefix = name + "."
            if prefix not in pieces:  # module names may differ from checkpoint names at the front
                matches = [p for p in pieces if p.endswith("layers." + name.rsplit("layers.", 1)[-1] + ".")]
                if len(matches) != 1:
                    raise SystemExit(f"no table pieces for {name}: {list(pieces)[:3]}")
                prefix = matches[0]
            rows = MappedRows(pieces[prefix], prefix)
            if rows.rows < module.table_rows:
                raise SystemExit(f"{name}: {rows.rows} rows mapped, {module.table_rows} expected")
            module.ngram_embedding = rows
            total += rows.rows
    return total
