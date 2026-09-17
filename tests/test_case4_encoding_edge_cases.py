import math

from app.cases.case4_accelerator.encoding import record_to_startup


def test_record_to_startup_nan_scores_fall_back_to_defaults():
    record = {
        "startup_id": "nan-scores",
        "traction_score": float("nan"),
        "team_score": float("nan"),
        "market_score": float("nan"),
        "runway_months": float("nan"),
        "past_response_score": float("nan"),
    }

    startup = record_to_startup(record)

    assert startup.traction_score == 0.5
    assert startup.team_score == 0.5
    assert startup.market_score == 0.5
    assert startup.runway_months == 6.0
    assert startup.past_response_score == 0.5


def test_record_to_startup_none_falls_back_to_defaults():
    record = {
        "startup_id": "none-scores",
        "traction_score": None,
        "team_score": None,
        "market_score": None,
        "runway_months": None,
        "past_response_score": None,
    }

    startup = record_to_startup(record)

    assert startup.traction_score == 0.5
    assert startup.team_score == 0.5
    assert startup.market_score == 0.5
    assert startup.runway_months == 6.0
    assert startup.past_response_score == 0.5


def test_record_to_startup_zero_runway_months_is_preserved():
    """Regression test: runway_months=0 (la startup literalmente se quedó sin
    runway, la señal más importante del caso) NO debe ser reemplazado por el
    default de 6.0."""
    record = {
        "startup_id": "out-of-runway",
        "traction_score": 0.7,
        "team_score": 0.7,
        "market_score": 0.7,
        "runway_months": 0.0,
        "past_response_score": 0.5,
    }

    startup = record_to_startup(record)

    assert startup.runway_months == 0.0


def test_record_to_startup_zero_score_is_preserved():
    """Regression test: un score real en 0.0 (ej. traction_score=0) no debe ser
    reemplazado en silencio por el default de 0.5."""
    record = {
        "startup_id": "zero-traction",
        "traction_score": 0.0,
        "team_score": 0.6,
        "market_score": 0.6,
        "runway_months": 12.0,
        "past_response_score": 0.0,
    }

    startup = record_to_startup(record)

    assert startup.traction_score == 0.0
    assert startup.past_response_score == 0.0


def test_record_to_startup_does_not_crash_on_mixed_nan_and_real_values():
    record = {
        "startup_id": "mixed",
        "traction_score": float("nan"),
        "team_score": 0.0,
        "market_score": math.nan,
        "runway_months": 0.0,
        "past_response_score": float("nan"),
    }

    startup = record_to_startup(record)

    assert startup.traction_score == 0.5  # NaN -> default
    assert startup.team_score == 0.0  # real zero preserved
    assert startup.market_score == 0.5  # NaN -> default
    assert startup.runway_months == 0.0  # real zero preserved
    assert startup.past_response_score == 0.5  # NaN -> default
