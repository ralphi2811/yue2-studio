"""Le catalogue de paramètres est la source de vérité du formulaire : il doit rester cohérent avec le runtime."""
from yue2.protocol import GenerationConfig, Sampling

from studio import params as P


def test_keys_are_unique_and_namespaced():
    keys = [p.key for p in (*P.REQUEST, *P.ABC_SAMPLING, *P.SEMANTIC_SAMPLING, *P.GENERATION, *P.ENGINE)]
    assert len(keys) == len(set(keys))
    assert all(k.startswith("abc.") for k in (p.key for p in P.ABC_SAMPLING))
    assert all(k.startswith("semantic.") for k in (p.key for p in P.SEMANTIC_SAMPLING))
    assert set(P.ALL_PARAMS) == set(keys)


def test_sampling_defaults_match_runtime():
    cfg = GenerationConfig()
    for catalog, defaults in ((P.ABC_SAMPLING, cfg.abc), (P.SEMANTIC_SAMPLING, cfg.semantic)):
        for p in catalog:
            field = p.key.split(".", 1)[1]
            assert p.default == getattr(defaults, field), p.key
    assert P.ALL_PARAMS["ode_steps"].default == cfg.ode_steps


def test_numeric_defaults_are_within_bounds():
    for p in P.ALL_PARAMS.values():
        if p.kind in ("int", "float") and p.default is not None:
            assert p.min is None or p.default >= p.min, p.key
            assert p.max is None or p.default <= p.max, p.key


def test_select_defaults_are_valid_choices():
    for p in P.ALL_PARAMS.values():
        if p.kind == "select":
            assert p.default in {c[0] for c in p.choices}, p.key


def test_every_param_has_help_text():
    for p in P.ALL_PARAMS.values():
        assert len(p.help) > 40, f"{p.key} manque d'explication"


def test_sampling_bounds_accepted_by_runtime():
    """Les bornes affichées doivent être acceptées par la dataclass Sampling (sinon l'UI promet l'impossible)."""
    for catalog in (P.ABC_SAMPLING, P.SEMANTIC_SAMPLING):
        lo = {p.key.split(".", 1)[1]: p.min for p in catalog}
        hi = {p.key.split(".", 1)[1]: p.max for p in catalog}
        Sampling(temperature=lo["temperature"], top_p=lo["top_p"], top_k=int(lo["top_k"]),
                 repetition_penalty=lo["repetition_penalty"], penalty_window=int(lo["penalty_window"]),
                 min_tokens=int(lo["min_tokens"]), max_tokens=int(hi["max_tokens"]))
        Sampling(temperature=hi["temperature"], top_p=hi["top_p"], top_k=int(hi["top_k"]),
                 repetition_penalty=hi["repetition_penalty"], penalty_window=int(hi["penalty_window"]),
                 min_tokens=int(lo["min_tokens"]), max_tokens=int(hi["max_tokens"]))
