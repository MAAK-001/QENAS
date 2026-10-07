"""DARTS block (fifth block type) — NAS-Unet down/up cells built from a DARTS genotype.

In Mixed-GGNAS the DARTS block is a cell whose internal operations were found by
the DARTS (NAS-Unet) gradient-based cell search; the genetic algorithm treats
"DARTS" as one block type and instantiates the *down* cell at encoder positions
and the *up* cell at decoder positions. The official repository ships the derived
genotypes; QENAS uses them as fixed cell structures (DARTS-derived block), so the
QENAS search itself remains training-free.

Cell semantics (NAS-Unet): two inputs ``s0`` (higher resolution / horizontal) and
``s1`` (previous cell), 1x1 preprocessing to ``c`` channels, four intermediate
nodes each summing two operations, output = concat of the four nodes (4c channels).

QENAS adaptations (documented in DESIGN_DECISIONS.md):
* Wiring follows NAS-Unet exactly: down cell ``s0`` = feature two levels back
  (2x resolution, preprocessed with stride 2), up cell ``s0`` = encoder skip at the
  output resolution. At the first encoder position both inputs are the stem output
  (same resolution), so ``preprocess0`` uses stride 1 there instead of relying on the
  official code's nearest-neighbour resize of mismatched tensors.
* Genotypes are validated: input edges of down cells must be down-sampling ops,
  the ``s1`` edges of up cells must be up-sampling ops, everything else must be a
  resolution-preserving op. A violation raises an error instead of being silently
  resized.
"""

from __future__ import annotations

from collections import namedtuple
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .darts_ops import DOWN_OPS, NORMAL_OPS, OPS, UP_OPS, ConvOps

Genotype = namedtuple("Genotype", "down down_concat up up_concat")

# Official Mixed-GGNAS genotypes (Mixed_GGNAS/cell_module/cells/darts_genotypes.py), verbatim.
GENOTYPES: Dict[str, Genotype] = {
    "darts_cell_busi": Genotype(
        down=[("down_cweight", 0), ("down_cweight", 1), ("down_dep_conv", 0), ("dil_conv", 2), ("down_cweight", 1),
              ("dil_conv", 2), ("dil_conv", 3), ("dil_conv", 4)], down_concat=range(2, 6),
        up=[("dil_conv", 0), ("up_cweight", 1), ("dil_conv", 0), ("dil_conv", 2), ("dil_conv", 3), ("dil_conv", 2),
            ("dil_conv", 3), ("dil_conv", 4)], up_concat=range(2, 6)),
    "darts_cell_cvc": Genotype(
        down=[("avg_pool", 1), ("avg_pool", 0), ("avg_pool", 1), ("avg_pool", 0), ("down_conv", 0), ("avg_pool", 1),
              ("down_dep_conv", 1), ("avg_pool", 0)], down_concat=range(2, 6),
        up=[("dep_conv", 0), ("up_dil_conv", 1), ("dil_conv", 2), ("up_dep_conv", 1), ("dep_conv", 3),
            ("up_dep_conv", 1), ("up_dep_conv", 1), ("dil_conv", 2)], up_concat=range(2, 6)),
    "darts_cell_idrid": Genotype(
        down=[("down_dil_conv", 0), ("down_dil_conv", 1), ("down_dil_conv", 0), ("down_dil_conv", 1), ("cweight", 2),
              ("down_dil_conv", 1), ("down_dep_conv", 1), ("dil_conv", 4)], down_concat=range(2, 6),
        up=[("cweight", 0), ("up_cweight", 1), ("up_cweight", 1), ("identity", 0), ("cweight", 2), ("identity", 0),
            ("dil_conv", 0), ("dil_conv", 4)], up_concat=range(2, 6)),
    "darts_cell_polyp": Genotype(
        down=[("down_cweight", 1), ("avg_pool", 0), ("down_cweight", 0), ("dil_conv", 2), ("down_dil_conv", 1),
              ("conv", 3), ("down_conv", 1), ("dep_conv", 2)], down_concat=range(2, 6),
        up=[("up_dil_conv", 1), ("shuffle_conv", 0), ("dil_conv", 2), ("dil_conv", 0), ("conv", 3), ("up_cweight", 1),
            ("dep_conv", 2), ("dil_conv", 0)], up_concat=range(2, 6)),
    "darts_unet": Genotype(
        down=[("max_pool", 1), ("down_conv", 0), ("max_pool", 1), ("down_conv", 0), ("dep_conv", 3),
              ("down_dep_conv", 0), ("down_dil_conv", 0), ("dep_conv", 2)], down_concat=range(2, 6),
        up=[("shuffle_conv", 0), ("up_dil_conv", 1), ("conv", 0), ("up_dil_conv", 1), ("up_dil_conv", 1),
            ("dep_conv", 3), ("dep_conv", 2), ("shuffle_conv", 0)], up_concat=range(2, 6)),
    # NAS-Unet genotype (searched on PROMISE12 by the NAS-Unet authors): dataset-independent option.
    "nasunet": Genotype(
        down=[("max_pool", 1), ("down_dil_conv", 0), ("cweight", 2), ("down_cweight", 0), ("down_dil_conv", 1),
              ("dil_conv", 3), ("dil_conv", 4), ("avg_pool", 0)], down_concat=range(2, 6),
        up=[("up_dil_conv", 1), ("dep_conv", 0), ("up_cweight", 1), ("shuffle_conv", 0), ("up_dil_conv", 1),
            ("dil_conv", 0), ("up_dil_conv", 1), ("shuffle_conv", 0)], up_concat=range(2, 6)),
}

DATASET_GENOTYPE = {"BUSI": "darts_cell_busi", "CVC": "darts_cell_cvc", "IDRID": "darts_cell_idrid"}


def resolve_genotype_name(name: str, dataset: str) -> str:
    if name == "auto":
        return DATASET_GENOTYPE[dataset.upper()]
    if name not in GENOTYPES:
        raise ValueError(f"Unknown DARTS genotype '{name}'. Available: {sorted(GENOTYPES)}")
    return name


def validate_genotype(g: Genotype) -> None:
    for kind in ("down", "up"):
        edges = list(getattr(g, kind))
        concat = list(getattr(g, f"{kind}_concat"))
        if len(edges) % 2 or len(edges) // 2 < 1:
            raise ValueError(f"genotype.{kind} must contain pairs of edges")
        n_nodes = len(edges) // 2
        for e, (op, idx) in enumerate(edges):
            node = 2 + e // 2
            if op not in OPS:
                raise ValueError(f"unknown op '{op}' in genotype.{kind}")
            if not 0 <= idx < node:
                raise ValueError(f"edge {e} of genotype.{kind} references future node {idx}")
            if kind == "down":
                allowed = DOWN_OPS if idx in (0, 1) else NORMAL_OPS
            else:
                allowed = UP_OPS if idx == 1 else NORMAL_OPS
            if op not in allowed:
                raise ValueError(f"genotype.{kind}: op '{op}' on input {idx} would break spatial consistency")
        if not all(2 <= c < 2 + n_nodes for c in concat):
            raise ValueError(f"genotype.{kind}_concat out of range")


class DartsCell(nn.Module):
    """NAS-Unet cell. ``cell_type='down'`` halves, ``'up'`` doubles the spatial resolution of ``s1``."""

    kind = "DARTS"

    def __init__(self, genotype: Genotype, c_prev_prev: int, c_prev: int, c: int, cell_type: str,
                 s0_stride: int = 1, genotype_name: str = ""):
        super().__init__()
        if cell_type not in ("down", "up"):
            raise ValueError(cell_type)
        validate_genotype(genotype)
        self.cell_type = cell_type
        self.genotype_name = genotype_name
        self.preprocess0 = ConvOps(c_prev_prev, c, kernel_size=1, stride=s0_stride, ops_order="act_weight_norm")
        self.preprocess1 = ConvOps(c_prev, c, kernel_size=1, ops_order="act_weight_norm")
        edges = genotype.down if cell_type == "down" else genotype.up
        concat = genotype.down_concat if cell_type == "down" else genotype.up_concat
        self.op_names: Tuple[str, ...] = tuple(op for op, _ in edges)
        self.indices: Tuple[int, ...] = tuple(i for _, i in edges)
        self.concat: Tuple[int, ...] = tuple(concat)
        self.n_nodes = len(edges) // 2
        self.ops = nn.ModuleList([OPS[name](c) for name in self.op_names])
        self.out_channels = c * len(self.concat)

    def forward(self, s0: torch.Tensor, s1: torch.Tensor) -> torch.Tensor:
        states: List[torch.Tensor] = [self.preprocess0(s0), self.preprocess1(s1)]
        for i in range(self.n_nodes):
            h1 = self.ops[2 * i](states[self.indices[2 * i]])
            h2 = self.ops[2 * i + 1](states[self.indices[2 * i + 1]])
            if h1.shape != h2.shape:
                raise RuntimeError(f"DARTS {self.cell_type} cell: incompatible node shapes {tuple(h1.shape)} "
                                   f"vs {tuple(h2.shape)} (check input resolutions)")
            states.append(h1 + h2)
        return torch.cat([states[i] for i in self.concat], dim=1)

    def describe(self) -> str:
        pairs = [f"n{2 + i // 2}<-{op}(s{idx})" if idx < 2 else f"n{2 + i // 2}<-{op}(n{idx})"
                 for i, (op, idx) in enumerate(zip(self.op_names, self.indices))]
        return f"{self.genotype_name} {self.cell_type}: " + ", ".join(pairs)
