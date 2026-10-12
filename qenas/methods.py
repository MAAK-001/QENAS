"""Methods of the QENAS evaluation study.

``qenas`` is the QENAS method (the only method ``main.py`` runs). Every other entry is a *comparison condition*
of the paper study (``study.py``), defined here and nowhere else, so that each one differs from QENAS in exactly
the documented way and shares all other code with it.

kind
    ``qenas``            the full pipeline (search -> Pareto subset -> candidates -> selection -> loss weights ->
                         final training -> test), with the listed search objectives and selection rule;
    ``qenas_selection``  re-uses the search, Pareto subset and 20-epoch candidate runs of the ``qenas`` experiment
                         with the same dataset and seed, and only changes how the final architecture is chosen;
    ``baseline``         a fixed hand-designed network trained with exactly the protocol QENAS uses for its
                         selected architecture (loss-weight selection -> final training -> TTA decision -> test);
    ``search_only``      a training-free search without any training (search-efficiency comparison).
"""

from __future__ import annotations

from typing import Any, Dict

ALL_OBJECTIVES = ("f1_params", "f2_synflow", "f3_zico")

METHODS: Dict[str, Dict[str, Any]] = {
    "qenas": {
        "kind": "qenas", "objectives": ALL_OBJECTIVES, "selection": "learning_capability",
        "label": "QENAS",
        "doc": "QENAS: NSGA-II on (params, SynFlow, ZiCo) + learning-capability selection",
    },
    "qenas_no_zico": {
        "kind": "qenas", "objectives": ("f1_params", "f2_synflow"), "selection": "learning_capability",
        "label": "QENAS w/o ZiCo",
        "doc": "ablation: two objectives (params, SynFlow), i.e. without the data-aware third objective",
    },
    "qenas_no_synflow": {
        "kind": "qenas", "objectives": ("f1_params", "f3_zico"), "selection": "learning_capability",
        "label": "QENAS w/o SynFlow",
        "doc": "ablation: two objectives (params, ZiCo), i.e. without SynFlow",
    },
    "qenas_random_pick": {
        "kind": "qenas_selection", "parent": "qenas", "selection": "random",
        "label": "QENAS, random Pareto candidate",
        "doc": "ablation: a uniformly random one of QENAS' Pareto candidates instead of learning-capability selection",
    },
    "qenas_zico_pick": {
        "kind": "qenas_selection", "parent": "qenas", "selection": "max_zico",
        "label": "QENAS, zero-shot pick (max ZiCo)",
        "doc": "ablation / zero-shot NAS rule: the Pareto candidate with the highest ZiCo, chosen without training",
    },
    "unet": {
        "kind": "baseline", "network": "unet",
        "label": "U-Net",
        "doc": "baseline: U-Net (Ronneberger et al. 2015), 31.04 M parameters as in the Mixed-GGNAS tables",
    },
    "attention_unet": {
        "kind": "baseline", "network": "attention_unet",
        "label": "Attention U-Net",
        "doc": "baseline: Attention U-Net (Oktay et al. 2018), 8.73 M parameters as in the Mixed-GGNAS tables",
    },
    "random_search": {
        "kind": "search_only", "objectives": ALL_OBJECTIVES, "sampler": "random",
        "label": "Random search",
        "doc": "search baseline: uniform random sampling with the evaluation budget of NSGA-II (training-free)",
    },
}

STAGES_BY_KIND = {
    "qenas": ["split", "search", "pareto", "candidates", "selection", "loss_weights", "final", "test", "report"],
    "qenas_selection": ["split", "inherit", "selection", "loss_weights", "final", "test", "report"],
    "baseline": ["split", "loss_weights", "final", "test", "report"],
    "search_only": ["split", "search"],
}


def method_spec(name: str) -> Dict[str, Any]:
    if name not in METHODS:
        raise ValueError(f"Unknown method '{name}'. Available: {sorted(METHODS)}")
    return METHODS[name]


def stages_for(name: str):
    return STAGES_BY_KIND[method_spec(name)["kind"]]
