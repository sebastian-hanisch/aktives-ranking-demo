"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen; Adressparameter, Szenario und PDF."""

import akt_constants as C
import akt_evaluation as ev
import akt_pdf_export as X
import akt_presets as P
from akt_scenario import Settings

FIELDS = {"n": "n", "delta": "delta", "budget": "budget", "strategy": "strategy", "lam": "lam", "round": "round_size", "seed": "seed"}


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_inside_the_slider_bounds():
    for name, p in C.PRESETS.items():
        for key, state_key in P.PRESET_KEYS.items():
            spec = P.SETTING_SPECS[state_key]
            if spec.lo is not None:
                assert spec.lo <= p[key] <= spec.hi, (name, key)
            if spec.choices is not None:
                assert p[key] in spec.choices, (name, key)


def test_default_preset_equals_the_default_settings():
    p = C.PRESETS["Rundenturnier gegen aktiv"]
    assert Settings(**{field: p[key] for key, field in FIELDS.items()}) == Settings()


def test_preset_keys_are_exactly_the_settings_fields():
    assert set(FIELDS.values()) == set(Settings.__dataclass_fields__)


def test_setting_specs_cover_every_preset_key_and_have_unique_parameters():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)
    params = [spec.url_param for spec in P.SETTING_SPECS.values()]
    assert len(params) == len(set(params))


def test_presets_show_their_mechanism():
    assert C.PRESETS["Enges Feld"]["delta"] == C.DELTA_NARROW and C.PRESETS["Die naive Regel"]["strategy"] == "Aktiv: Fehlordnung"
    assert C.PRESETS["Wenige Partien"]["budget"] < C.DEFAULT_BUDGET


def test_sweep_instances_are_fixed_and_distinct_and_the_default_strategy_is_active():
    assert len(C.SWEEP_SEEDS) == len(set(C.SWEEP_SEEDS)) == 60
    assert C.DEFAULT_STRATEGY in C.ACTIVE and set(C.ACTIVE) <= set(C.STRATEGIES) and set(C.COLORS) == set(C.STRATEGIES)


def test_pdf_is_built_for_every_preset():
    for name, p in C.PRESETS.items():
        s = Settings(n=8, delta=p["delta"], budget=40, strategy=p["strategy"], lam=p["lam"], round_size=p["round"], seed=p["seed"])
        data = X.build_pdf(ev.analyse(s))
        assert data[:5] == b"%PDF-" and len(data) > 1500


def test_latin_filter_replaces_greek_and_math_symbols():
    out = X._latin("λ ≈ 0,5; σ → τ µ θ − **fett** „Zitat“")
    assert out == 'lambda ca. 0,5; sigma -> Tau mu theta - fett "Zitat"'
    assert out.encode("latin-1")
