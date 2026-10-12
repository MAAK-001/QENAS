"""QENAS — Quick Evolutionary Neural Architecture Search.

Training-free NSGA-II search (objectives: parameter count, SynFlow, ZiCo) over a
mixed search space of four manually designed blocks and one DARTS block,
followed by short supervised evaluation of Pareto candidates, learning-capability
based selection, and ordinary supervised training of the selected architecture.
"""

__version__ = "2.1.0"
