"""QENAS — Quick Evolutionary Neural Architecture Search.

Training-free NSGA-II search (objectives: parameter count, ZiCo) over a mixed search space of four manually
designed blocks and one DARTS block, followed by successive-halving supervised evaluation of 16 candidates,
learning-capability based selection, and ordinary supervised training of the selected architecture.
"""

__version__ = "3.0.0"
