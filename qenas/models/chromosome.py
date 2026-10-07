"""Chromosome encoding of the QENAS search space.

A chromosome is ``[g1, ..., g8]`` with ``gi in {0, 1, 2, 3, 4}``:
genes 1-4 choose the block type of the four encoder positions (each halves the
resolution), genes 5-8 the four decoder positions (each doubles it).

    0 Residual | 1 Dense | 2 Inception | 3 ConvNeXt | 4 DARTS

(The numbering is the official Mixed-GGNAS numbering.) Search-space size: 5^8 = 390,625.
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np

CHROMOSOME_LENGTH = 8
N_BLOCK_TYPES = 5
BLOCK_NAMES = {0: "Residual", 1: "Dense", 2: "Inception", 3: "ConvNeXt", 4: "DARTS"}
BLOCK_SHORT = {0: "RES", 1: "DEN", 2: "INC", 3: "CNX", 4: "DRT"}
POSITION_NAMES = ["enc1", "enc2", "enc3", "enc4", "dec1", "dec2", "dec3", "dec4"]
SEARCH_SPACE_SIZE = N_BLOCK_TYPES ** CHROMOSOME_LENGTH


class InvalidChromosomeError(ValueError):
    pass


def validate_chromosome(ch: Sequence) -> List[int]:
    arr = np.asarray(ch)
    if arr.ndim != 1 or arr.shape[0] != CHROMOSOME_LENGTH:
        raise InvalidChromosomeError(f"chromosome must have length {CHROMOSOME_LENGTH}, got shape {arr.shape}")
    out = []
    for g in arr.tolist():
        if isinstance(g, float):
            if not float(g).is_integer():
                raise InvalidChromosomeError(f"non-integer gene {g} in {list(ch)}")
            g = int(g)
        if not isinstance(g, (int, np.integer)) or not 0 <= int(g) < N_BLOCK_TYPES:
            raise InvalidChromosomeError(f"gene {g} outside {{0..{N_BLOCK_TYPES - 1}}} in {list(ch)}")
        out.append(int(g))
    return out


def chromosome_to_str(ch: Sequence[int]) -> str:
    return "[" + ", ".join(str(int(g)) for g in ch) + "]"


def chromosome_key(ch: Sequence[int]) -> str:
    return "".join(str(int(g)) for g in ch)


def decode(ch: Sequence[int]) -> List[str]:
    return [BLOCK_NAMES[g] for g in validate_chromosome(ch)]


def decode_short(ch: Sequence[int]) -> str:
    gs = validate_chromosome(ch)
    return "-".join(BLOCK_SHORT[g] for g in gs[:4]) + " | " + "-".join(BLOCK_SHORT[g] for g in gs[4:])
