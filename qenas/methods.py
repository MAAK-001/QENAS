"""Methods of the QENAS evaluation study.

``qenas`` is the QENAS method (the only method ``main.py`` runs). Every other entry is a *comparison condition*
of the paper study (``study.py``), defined here and nowhere else, so that each one differs from QENAS in exactly
the documented way and shares all other code with it.

kind
    ``qenas``          the full pipeline (search -> candidates -> successive halving -> selection -> loss weights
                       -> final training -> test);
    ``qenas_inherit``  re-uses the search, candidates and successive-halving runs of the ``qenas`` experiment with
                       the same dataset and seed, and changes only the selection rule (``selection``) or the
                       loss-weight rule (``loss_weights``);
    ``baseline``       a fixed hand-designed network trained with exactly the protocol QENAS uses for its selected
                       architecture (loss-weight selection -> final training -> test);
    ``nnunet``         nnU-Net (Isensee et al., Nature Methods 2021) with its own self-configuring pipeline on the
                       same split and the same images (qenas/study/nnunet.py);
    ``search_only``    a training-free search without any training (search-efficiency comparison);
    ``random_arch``    one architecture drawn uniformly from the search space (no search, no ZiCo), trained with the
                       QENAS protocol (loss-weight selection -> final training -> test).

sampler (kind ``qenas`` / ``search_only``)
    ``nsga2`` (default) the evolutionary search; ``random`` uniform random sampling with NSGA-II's evaluation budget.

loss_weights
    ``grid``   alpha chosen on validation from ``loss_weights.alphas`` (the QENAS protocol);
    ``fixed``  alpha = ``candidate_training.loss_alpha`` (0.5) without any search (loss-coefficient ablation).
"""

from __future__ import annotations

from typing import Any, Dict, List

ALL_OBJECTIVES = ("f1_params", "f2_zico")

METHODS: Dict[str, Dict[str, Any]] = {
    "qenas": {
        "kind": "qenas", "objectives": ALL_OBJECTIVES, "selection": "learning_capability", "loss_weights": "grid",
        "label": "QENAS",
        "doc": "QENAS: NSGA-II on (params, ZiCo) + successive-halving learning-capability selection",
    },
    "qenas_fixed_alpha": {
        "kind": "qenas_inherit", "parent": "qenas", "selection": "learning_capability", "loss_weights": "fixed",
        "label": "QENAS, fixed α = 0.5",
        "doc": "ablation: the QENAS architecture trained with the neutral loss weights alpha = beta = 0.5, without "
               "the loss-coefficient search",
    },
    "qenas_random_pick": {
        "kind": "qenas_inherit", "parent": "qenas", "selection": "random", "loss_weights": "grid",
        "label": "QENAS, random candidate",
        "doc": "ablation: a uniformly random one of QENAS' 16 candidates instead of learning-capability selection",
    },
    "qenas_zico_pick": {
        "kind": "qenas_inherit", "parent": "qenas", "selection": "max_zico", "loss_weights": "grid",
        "label": "QENAS, zero-shot pick (max ZiCo)",
        "doc": "ablation / zero-shot NAS rule: the candidate with the highest ZiCo, chosen without training",
    },
    "unet": {
        "kind": "baseline", "network": "unet", "loss_weights": "grid",
        "label": "U-Net",
        "doc": "baseline: U-Net (Ronneberger et al. 2015), 31.04 M parameters as in the Mixed-GGNAS tables",
    },
    "attention_unet": {
        "kind": "baseline", "network": "attention_unet", "loss_weights": "grid",
        "label": "Attention U-Net",
        "doc": "baseline: Attention U-Net (Oktay et al. 2018), 8.73 M parameters as in the Mixed-GGNAS tables",
    },
    "nnunet": {
        "kind": "nnunet",
        "label": "nnU-Net",
        "doc": "baseline: nnU-Net v2 (2d configuration, default plans, nnUNetTrainer_100epochs), fold 0 = the "
               "QENAS train/validation split",
    },
    "random_search": {
        "kind": "search_only", "objectives": ALL_OBJECTIVES, "sampler": "random",
        "label": "Random search",
        "doc": "search baseline: uniform random sampling with the evaluation budget of NSGA-II (training-free)",
    },
    # ---- component ablation study (IDRiD, Kvasir-SEG; plan.ablation_plan)
    "abl_random_population": {
        "kind": "qenas", "objectives": ALL_OBJECTIVES, "sampler": "random", "selection": "learning_capability",
        "loss_weights": "grid",
        "label": "QENAS, random-search population (no NSGA-II)",
        "doc": "ablation: the 220 architectures are sampled uniformly (NSGA-II's budget) instead of evolved; the 16 "
               "candidates are their non-dominated front in (params, ZiCo), filled up by rank and crowding; the rest "
               "of QENAS is unchanged",
    },
    "abl_params_only": {
        "kind": "qenas", "objectives": ("f1_params",), "selection": "learning_capability", "loss_weights": "grid",
        "label": "QENAS, parameters only (no ZiCo)",
        "doc": "ablation: NSGA-II minimises the number of parameters only (ZiCo is recorded but not optimised); the "
               "rest of QENAS is unchanged",
    },
    "abl_random_arch": {
        "kind": "random_arch", "loss_weights": "grid",
        "label": "Random architecture (no search, no ZiCo)",
        "doc": "ablation: one architecture drawn uniformly from the 5^8 search space, then the QENAS loss-weight "
               "selection and final training",
    },
}


def method_spec(name: str) -> Dict[str, Any]:
    if name not in METHODS:
        raise ValueError(f"Unknown method '{name}'. Available: {sorted(METHODS)}")
    return METHODS[name]


def stages_for(name: str) -> List[str]:
    m = method_spec(name)
    grid = ["loss_weights"] if m.get("loss_weights") == "grid" else []
    return {
        "qenas": ["split", "search", "pareto", "candidates", "selection"] + grid + ["final", "test", "report"],
        "qenas_inherit": ["split", "inherit", "selection"] + grid + ["final", "test", "report"],
        "baseline": ["split"] + grid + ["final", "test", "report"],
        "random_arch": ["split"] + grid + ["final", "test", "report"],
        "nnunet": ["split", "nnunet_data", "nnunet_train", "nnunet_test", "report"],
        "search_only": ["split", "search"],
    }[m["kind"]]
