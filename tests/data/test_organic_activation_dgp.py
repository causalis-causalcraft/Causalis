"""Structural checks for the organic activation panel and its oracle targets."""

from copy import deepcopy
from dataclasses import FrozenInstanceError

import numpy as np
import pandas as pd
import pytest

from causalis.dgp import generate_organic_activation_data as generate_top_level
from causalis.dgp.organic_activation import (
    OrganicActivationConfig,
    OrganicActivationGenerator,
    generate_data,
    generate_organic_activation_data,
    summarize_data,
    validate_data,
)
from causalis.dgp.organic_activation.base import replay_activation

SCENARIOS = (
    "baseline",
    "latent_level_confounding",
    "observed_timevarying_confounding",
    "hidden_timevarying_confounding",
    "null_effect",
    "weak_overlap",
)
EFFECT_COLUMNS = (
    "att_no_activation",
    "att_defer_natural",
    "cumulative_no_activation",
    "cumulative_defer_natural",
)
STATIC_COLUMNS = (
    "x_income",
    "x_salary",
    "x_segment",
    "x_digital",
    "x_noise_1",
    "x_noise_2",
)
PANEL_COLUMNS = {
    "client_id",
    "period",
    "date",
    "entry_period",
    "tenure_months",
    "n_pre_periods_available",
    "first_activation_period",
    "activation_now",
    "treated",
    "eligible_to_activate",
    "event_time",
    "l_activity_pre",
    "ltv_monthly",
    "ltv_cumulative",
    *STATIC_COLUMNS,
}


@pytest.fixture(scope="module", params=SCENARIOS)
def generated(request):
    config = OrganicActivationConfig(n_clients=1600, scenario=request.param, seed=163)
    df, oracle, metadata = OrganicActivationGenerator(config).generate()
    return config, df, oracle, metadata


def test_all_scenarios_have_the_observed_panel_contract_and_invariants(generated):
    config, df, oracle, metadata = generated
    assert isinstance(df, pd.DataFrame)
    assert isinstance(oracle, dict)
    assert isinstance(metadata, dict)
    assert PANEL_COLUMNS.issubset(df.columns)
    assert {"U", "V", "hazard", "y_no_activation", "effect_factual"}.isdisjoint(
        df.columns
    )
    assert not df.duplicated(["client_id", "period"]).any()
    assert df["client_id"].nunique() == config.n_clients

    ordered = df.sort_values(["client_id", "period"])
    groups = ordered.groupby("client_id", sort=False)
    assert groups["activation_now"].sum().le(1).all()
    np.testing.assert_array_equal(
        ordered["treated"], groups["activation_now"].cumsum().astype(bool)
    )
    assert (
        not ordered.loc[ordered["activation_now"].astype(bool), "eligible_to_activate"]
        .eq(False)
        .any()
    )
    assert (
        ordered.loc[ordered["activation_now"].astype(bool), "tenure_months"].ge(1).all()
    )
    np.testing.assert_array_equal(
        ordered["tenure_months"], ordered["period"] - ordered["entry_period"]
    )
    np.testing.assert_array_equal(
        ordered["n_pre_periods_available"], ordered["tenure_months"]
    )
    np.testing.assert_array_equal(
        groups["period"].min(), groups["entry_period"].first()
    )
    assert groups["period"].max().eq(config.n_periods - 1).all()
    np.testing.assert_array_equal(
        groups.size(), config.n_periods - groups["entry_period"].first()
    )
    np.testing.assert_allclose(
        ordered["ltv_cumulative"], groups["ltv_monthly"].cumsum()
    )
    assert groups[list(STATIC_COLUMNS)].nunique(dropna=False).le(1).all().all()
    calendar = pd.period_range(
        config.calendar_start, periods=config.n_periods, freq="M"
    ).to_timestamp()
    np.testing.assert_array_equal(ordered["date"], calendar.take(ordered["period"]))

    activated = ordered["first_activation_period"].notna()
    np.testing.assert_array_equal(
        ordered.loc[activated, "event_time"].to_numpy(dtype=int),
        (
            ordered.loc[activated, "period"]
            - ordered.loc[activated, "first_activation_period"]
        ).to_numpy(dtype=int),
    )
    assert ordered.loc[~activated, "event_time"].isna().all()
    activation_rows = ordered[ordered["activation_now"].astype(bool)]
    np.testing.assert_array_equal(
        activation_rows["period"].to_numpy(dtype=int),
        activation_rows["first_activation_period"].to_numpy(dtype=int),
    )
    main = activation_rows[activation_rows["period"].isin(config.main_cohorts)]
    assert main["n_pre_periods_available"].between(1, 10).all()


def test_potential_outcomes_are_separate_consistent_and_have_no_anticipation(generated):
    _, df, oracle, _ = generated
    potential = oracle["potential_outcomes"]
    assert len(potential) == len(df)
    assert potential.index.equals(df.index)
    assert {
        "y_no_activation",
        "y_factual",
        "effect_factual",
        "hazard",
        "U",
        "V",
    }.issubset(potential)
    np.testing.assert_array_equal(potential["y_factual"], df["ltv_monthly"])
    np.testing.assert_allclose(
        potential["y_no_activation"] + potential["effect_factual"], df["ltv_monthly"]
    )
    untreated = ~df["treated"].astype(bool)
    assert potential.loc[untreated, "effect_factual"].eq(0.0).all()
    np.testing.assert_array_equal(
        potential.loc[untreated, "y_no_activation"], df.loc[untreated, "ltv_monthly"]
    )
    risk_hazard = potential.loc[df["eligible_to_activate"].astype(bool), "hazard"]
    assert risk_hazard.between(0, 1, inclusive="neither").all()
    assert (
        np.isfinite(df[["ltv_monthly", "ltv_cumulative", "l_activity_pre"]]).all().all()
    )


def test_factual_replay_and_skip_recompute_the_first_future_activation(generated):
    config, df, oracle, _ = generated
    exogenous = oracle["exogenous"]
    client_rows = df.drop_duplicates("client_id").set_index("client_id").sort_index()
    actual_g = client_rows["first_activation_period"].fillna(-1).to_numpy(dtype=int)
    np.testing.assert_array_equal(replay_activation(exogenous, config), actual_g)
    np.testing.assert_array_equal(
        replay_activation(exogenous, config, never_activate=True),
        np.full(config.n_clients, -1),
    )

    hazard = np.full((config.n_clients, config.n_periods), np.nan)
    client_ids = df["client_id"].to_numpy(dtype=int)
    periods = df["period"].to_numpy(dtype=int)
    hazard[client_ids, periods] = oracle["potential_outcomes"]["hazard"].to_numpy()
    structural_eligible = (
        np.arange(config.n_periods)[None, :]
        > client_rows["entry_period"].to_numpy()[:, None]
    )
    successes = (exogenous["activation_uniform"] < hazard) & structural_eligible
    any_success = successes.any(axis=1)
    expected_factual = np.where(any_success, successes.argmax(axis=1), -1)
    np.testing.assert_array_equal(actual_g, expected_factual)

    skip = oracle["skip_activation_period"]
    main_ids = np.flatnonzero(np.isin(actual_g, config.main_cohorts))
    assert skip.index.is_unique
    np.testing.assert_array_equal(np.sort(skip.index.to_numpy(dtype=int)), main_ids)
    future_successes = successes[main_ids] & (
        np.arange(config.n_periods)[None, :] > actual_g[main_ids, None]
    )
    expected_skip = np.where(
        future_successes.any(axis=1), future_successes.argmax(axis=1), -1
    )
    actual_skip = skip.reindex(main_ids).fillna(-1).to_numpy(dtype=int)
    np.testing.assert_array_equal(actual_skip, expected_skip)
    assert ((actual_skip == -1) | (actual_skip > actual_g[main_ids])).all()
    # These seeds have later successes: forbidding the factual month must not
    # silently turn every treated client's alternative into "never activate".
    assert (actual_skip >= 0).any()


def test_individual_oracles_use_the_replayed_timing_and_same_innovations(generated):
    config, df, oracle, _ = generated
    individual = oracle["individual_effects"]
    assert {
        "client_id",
        "g",
        "event_time",
        "y_start",
        "y_no_activation",
        "y_defer_natural",
        *EFFECT_COLUMNS,
    }.issubset(individual)
    assert not individual.duplicated(["client_id", "event_time"]).any()
    assert individual.groupby("client_id").size().eq(config.horizon).all()
    assert individual["event_time"].between(0, config.horizon - 1).all()

    potential = oracle["potential_outcomes"]
    lookup = pd.Series(
        potential["effect_factual"].to_numpy(),
        index=pd.MultiIndex.from_arrays([df["client_id"], df["period"]]),
    )
    rows = individual.sort_values(["client_id", "event_time"])
    factual_index = pd.MultiIndex.from_arrays(
        [rows["client_id"], rows["g"] + rows["event_time"]]
    )
    factual_effect = lookup.reindex(factual_index).to_numpy()
    np.testing.assert_allclose(rows["att_no_activation"], factual_effect)

    skipped_g = (
        oracle["skip_activation_period"]
        .reindex(rows["client_id"])
        .fillna(-1)
        .to_numpy(dtype=int)
    )
    original_g = rows["g"].to_numpy(dtype=int)
    h = rows["event_time"].to_numpy(dtype=int)
    delay = skipped_g - original_g
    active_in_alternative = (skipped_g >= 0) & (h >= delay)
    alternative_effect = np.zeros(len(rows))
    # With outcome-only treatment, a later start uses the same client's
    # event-time response at h - delay, with all untreated innovations fixed.
    shifted_index = pd.MultiIndex.from_arrays(
        [
            rows.loc[active_in_alternative, "client_id"],
            original_g[active_in_alternative]
            + h[active_in_alternative]
            - delay[active_in_alternative],
        ]
    )
    alternative_effect[active_in_alternative] = lookup.reindex(shifted_index).to_numpy()
    np.testing.assert_allclose(
        rows["att_defer_natural"], factual_effect - alternative_effect
    )
    untreated_lookup = pd.Series(
        potential["y_no_activation"].to_numpy(), index=lookup.index
    )
    factual_lookup = pd.Series(df["ltv_monthly"].to_numpy(), index=lookup.index)
    untreated_outcomes = untreated_lookup.reindex(factual_index).to_numpy()
    np.testing.assert_array_equal(
        rows["y_start"], factual_lookup.reindex(factual_index).to_numpy()
    )
    np.testing.assert_array_equal(rows["y_no_activation"], untreated_outcomes)
    np.testing.assert_allclose(
        rows["y_defer_natural"], untreated_outcomes + alternative_effect
    )
    np.testing.assert_allclose(
        rows["y_start"] - rows["y_no_activation"], rows["att_no_activation"]
    )
    np.testing.assert_allclose(
        rows["y_start"] - rows["y_defer_natural"], rows["att_defer_natural"]
    )
    for monthly, cumulative in (
        ("att_no_activation", "cumulative_no_activation"),
        ("att_defer_natural", "cumulative_defer_natural"),
    ):
        np.testing.assert_allclose(
            rows[cumulative], rows.groupby("client_id", sort=False)[monthly].cumsum()
        )


def test_cohort_and_aggregate_oracles_keep_fixed_treated_weights(generated):
    config, _, oracle, _ = generated
    individual = oracle["individual_effects"]
    by_cohort = oracle["att_by_cohort"].sort_values(["g", "event_time"])
    aggregate = oracle["att_aggregate"].sort_values("event_time")
    counts = individual.groupby("g")["client_id"].nunique()
    expected_counts = by_cohort["g"].map(counts).fillna(0).to_numpy()
    np.testing.assert_array_equal(by_cohort["n_treated"], expected_counts)
    np.testing.assert_allclose(by_cohort["weight"], expected_counts / counts.sum())
    np.testing.assert_allclose(
        by_cohort.groupby("event_time")["weight"].sum(), np.ones(config.horizon)
    )
    np.testing.assert_array_equal(aggregate["event_time"], np.arange(config.horizon))

    expected = individual.groupby(["g", "event_time"])[list(EFFECT_COLUMNS)].mean()
    nonempty = by_cohort[by_cohort["n_treated"] > 0].set_index(["g", "event_time"])
    for column in EFFECT_COLUMNS:
        np.testing.assert_allclose(
            nonempty[column], expected.reindex(nonempty.index)[column]
        )
        weighted = by_cohort[column].fillna(0) * by_cohort["weight"]
        np.testing.assert_allclose(
            aggregate[column],
            weighted.groupby(by_cohort["event_time"]).sum().to_numpy(),
        )
    np.testing.assert_allclose(
        aggregate["cumulative_no_activation"], aggregate["att_no_activation"].cumsum()
    )
    np.testing.assert_allclose(
        aggregate["cumulative_defer_natural"], aggregate["att_defer_natural"].cumsum()
    )


def test_metadata_separates_temporal_feature_safety_from_identification(generated):
    config, _, _, metadata = generated
    assert metadata["scenario"] == config.scenario
    manifests = metadata["model_features"]
    assert set(manifests) == set(SCENARIOS)
    allowed = {
        *STATIC_COLUMNS,
        "entry_period",
        "period",
        "tenure_months",
        "l_activity_pre",
    }
    for scenario_features in manifests.values():
        assert set(scenario_features["did"]) == {*STATIC_COLUMNS, "entry_period"}
        assert "l_activity_pre" in scenario_features["dml"]
        for names in scenario_features.values():
            unprefixed = {name.removeprefix("b_") for name in names}
            assert unprefixed.issubset(allowed)
    table = metadata["scenario_table"]
    assert not table.duplicated(["scenario", "method"]).any()
    assert len(table) == len(SCENARIOS) * 3
    for row in table.itertuples(index=False):
        expected_identification = (
            row.scenario in {"baseline", "null_effect", "weak_overlap"}
            or (row.scenario == "latent_level_confounding" and row.method == "did")
            or (
                row.scenario == "observed_timevarying_confounding"
                and row.method in {"dml", "ipcw"}
            )
        )
        assert bool(row.identified) is expected_identification
        assert row.estimand == (
            "ATT_defer_natural" if row.method == "dml" else "ATT_no_activation"
        )
        assert row.features and row.assumptions


def test_validation_and_diagnostics_are_nonmutating_and_use_the_factual_risk_set(
    generated,
):
    config, df, oracle, metadata = generated
    original_df = df.copy(deep=True)
    original_frames = {
        name: oracle[name].copy(deep=True)
        for name in (
            "potential_outcomes",
            "individual_effects",
            "att_by_cohort",
            "att_aggregate",
        )
    }
    original_skip = oracle["skip_activation_period"].copy(deep=True)
    original_exogenous = {
        name: values.copy() for name, values in oracle["exogenous"].items()
    }
    original_metadata = deepcopy(metadata)

    checks = validate_data(df, oracle, metadata)
    assert checks and all(isinstance(value, bool) for value in checks.values())
    assert all(
        value for name, value in checks.items() if not name.startswith("support_")
    )
    assert {
        "factual_replay",
        "deferred_regime_replay",
        "feature_isolation",
        "artificial_censoring",
    }.issubset(checks)
    diagnostics = summarize_data(df, oracle, metadata)
    overview = diagnostics["overview"]
    assert overview["n_clients"] == config.n_clients
    assert overview["n_rows"] == len(df)
    clients = df.drop_duplicates("client_id")
    first = clients["first_activation_period"]
    assert overview["never_activated_fraction"] == first.isna().mean()
    expected_quantiles = (
        oracle["potential_outcomes"]
        .loc[df["eligible_to_activate"].astype(bool), "hazard"]
        .quantile(diagnostics["hazard_quantiles"].index)
    )
    pd.testing.assert_series_equal(diagnostics["hazard_quantiles"], expected_quantiles)
    for row in diagnostics["control_support"].itertuples(index=False):
        eligible = (clients["entry_period"] < row.g) & (first.isna() | first.ge(row.g))
        expected = eligible & (first.isna() | first.gt(row.t))
        assert row.n_controls == expected.sum()
    for row in diagnostics["dml_later_activation"].itertuples(index=False):
        controls = (clients["entry_period"] < row.g) & (first.isna() | first.gt(row.g))
        later = controls & first.gt(row.g)
        inside = later & first.lt(row.g + config.horizon)
        assert row.n_controls == controls.sum()
        assert row.n_later_activators == later.sum()
        assert row.n_activations_within_horizon == inside.sum()
        if row.n_controls:
            assert row.fraction_later_activation == later.sum() / row.n_controls
            assert (
                row.fraction_activation_within_horizon == inside.sum() / row.n_controls
            )
    aggregate = diagnostics["att_aggregate"]
    np.testing.assert_allclose(
        aggregate["monthly_estimand_gap"],
        aggregate["att_no_activation"] - aggregate["att_defer_natural"],
    )
    np.testing.assert_allclose(
        aggregate["cumulative_estimand_gap"],
        aggregate["cumulative_no_activation"] - aggregate["cumulative_defer_natural"],
    )

    # Post-activation prospective hazards are useful to the oracle, but must
    # never enter diagnostics of the factual activation risk set.
    altered_oracle = dict(oracle)
    altered_potential = oracle["potential_outcomes"].copy(deep=True)
    post_activation = df["treated"].astype(bool) & ~df["activation_now"].astype(bool)
    assert post_activation.any()
    altered_potential.loc[post_activation, "hazard"] = 1e-30
    altered_oracle["potential_outcomes"] = altered_potential.sample(
        frac=1, random_state=12
    ).reset_index(drop=True)
    altered_diagnostics = summarize_data(
        df.sample(frac=1, random_state=13), altered_oracle, metadata
    )
    pd.testing.assert_series_equal(
        altered_diagnostics["hazard_quantiles"], expected_quantiles
    )

    pd.testing.assert_frame_equal(df, original_df)
    for name, original in original_frames.items():
        pd.testing.assert_frame_equal(oracle[name], original)
    pd.testing.assert_series_equal(oracle["skip_activation_period"], original_skip)
    for name, original in original_exogenous.items():
        np.testing.assert_array_equal(oracle["exogenous"][name], original)
    assert metadata["model_features"] == original_metadata["model_features"]
    pd.testing.assert_frame_equal(
        metadata["scenario_table"], original_metadata["scenario_table"]
    )


@pytest.mark.parametrize(
    "corruption",
    (
        "cumulative",
        "outcome",
        "feature_manifest",
        "observed_static_covariate",
        "potential_decomposition",
        "empty_feature_mapping",
        "metadata_cohort_weights",
    ),
)
def test_validator_rejects_corrupted_data_or_leaking_feature_manifests(corruption):
    df, oracle, metadata = generate_data(n_clients=300, seed=71)
    if corruption == "cumulative":
        df.loc[df.index[0], "ltv_cumulative"] += 500.0
        expected_check = "cumulative_ltv"
    elif corruption == "outcome":
        # Keep the stored oracle and cumulative accounting internally coherent:
        # independent replay must still expose altered factual outcomes.
        df.loc[df.index[0], "ltv_monthly"] += 500.0
        df["ltv_cumulative"] = df.groupby("client_id")["ltv_monthly"].cumsum()
        oracle["potential_outcomes"].loc[
            df.index[0], ["y_factual", "y_no_activation"]
        ] += 500.0
        expected_check = "factual_replay"
    elif corruption == "feature_manifest":
        metadata["model_features"]["baseline"]["did"].append("event_time")
        expected_check = "feature_isolation"
    elif corruption == "observed_static_covariate":
        client_id = df["client_id"].iloc[0]
        # Preserve the within-client static covariate invariant while breaking
        # correspondence to the exogenous client used by the structural model.
        df.loc[df["client_id"].eq(client_id), "x_income"] += 10_000.0
        expected_check = "observed_covariate_replay"
    elif corruption == "potential_decomposition":
        # The observed sum still agrees; each component must match replay too.
        row = df.index[df["treated"].astype(bool)][0]
        oracle["potential_outcomes"].loc[row, "y_no_activation"] += 100.0
        oracle["potential_outcomes"].loc[row, "effect_factual"] -= 100.0
        expected_check = "potential_outcome_replay"
    elif corruption == "empty_feature_mapping":
        metadata["model_features"] = {}
        expected_check = "feature_isolation"
    else:
        metadata["cohort_weights"].loc[0, "weight"] += 0.1
        expected_check = "metadata_consistency"
    with pytest.raises(ValueError, match=expected_check):
        validate_data(df, oracle, metadata)


@pytest.mark.parametrize("column", ("y_start", "y_no_activation", "y_defer_natural"))
def test_validator_rejects_corrupted_individual_potential_outcomes(column):
    df, oracle, metadata = generate_data(n_clients=300, seed=71)
    oracle["individual_effects"].loc[0, column] += 100.0
    with pytest.raises(ValueError, match="deferred_regime_replay"):
        validate_data(df, oracle, metadata)


def test_null_effect_is_exactly_zero_without_changing_selection():
    df, oracle, _ = generate_data(n_clients=900, scenario="null_effect", seed=74)
    baseline, _, _ = generate_data(n_clients=900, scenario="baseline", seed=74)
    pd.testing.assert_series_equal(
        df["first_activation_period"], baseline["first_activation_period"]
    )
    pd.testing.assert_series_equal(df["l_activity_pre"], baseline["l_activity_pre"])
    potential = oracle["potential_outcomes"]
    np.testing.assert_array_equal(potential["y_no_activation"], df["ltv_monthly"])
    assert potential["effect_factual"].eq(0.0).all()
    for name in ("individual_effects", "att_by_cohort", "att_aggregate"):
        values = oracle[name][list(EFFECT_COLUMNS)].to_numpy()
        assert np.equal(values[np.isfinite(values)], 0).all()


def test_reproducibility_and_top_level_api_do_not_touch_the_legacy_rng():
    assert generate_top_level is generate_organic_activation_data
    config = OrganicActivationConfig(n_clients=500, seed=112)
    before = np.random.get_state()
    try:
        np.random.seed(716)
        expected_legacy_draw = np.random.random()
        np.random.seed(716)
        first = OrganicActivationGenerator(config).generate()
        assert np.random.random() == expected_legacy_draw
    finally:
        np.random.set_state(before)
    second = generate_top_level(n_clients=500, seed=112)
    third = generate_data(n_clients=500, seed=112)
    for other in (second, third):
        pd.testing.assert_frame_equal(first[0], other[0])
        for key in (
            "potential_outcomes",
            "individual_effects",
            "att_by_cohort",
            "att_aggregate",
        ):
            pd.testing.assert_frame_equal(first[1][key], other[1][key])
        pd.testing.assert_series_equal(
            first[1]["skip_activation_period"], other[1]["skip_activation_period"]
        )
        for name, values in first[1]["exogenous"].items():
            np.testing.assert_array_equal(values, other[1]["exogenous"][name])
    different, _, _ = generate_top_level(n_clients=500, seed=113)
    assert not first[0].equals(different)


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_replay_hazard_uses_only_the_scenario_designated_sources(scenario):
    config = OrganicActivationConfig(n_clients=500, scenario=scenario, seed=391)
    _, oracle, _ = OrganicActivationGenerator(config).generate()
    exogenous = oracle["exogenous"]
    factual = replay_activation(exogenous, config)
    active_source = {
        "latent_level_confounding": "latent_standard",
        "observed_timevarying_confounding": "state_innovation",
        "hidden_timevarying_confounding": "hidden_innovation",
    }.get(scenario)
    sources = (
        "latent_standard",
        "state_innovation",
        "hidden_innovation",
        "outcome_innovation",
    )
    for source in sources:
        altered = dict(exogenous)
        altered[source] = exogenous[source] + 5.0
        if source == "latent_standard":
            altered["latent_level"] = config.latent_level_std * altered[source]
        changed = replay_activation(altered, config)
        if source == active_source:
            assert np.not_equal(factual, changed).any()
        else:
            np.testing.assert_array_equal(factual, changed)


def test_default_panel_has_all_main_cohorts_and_later_and_never_activators():
    df, oracle, _ = generate_top_level()
    clients = df.drop_duplicates("client_id")
    assert len(clients) == 50_000
    assert df["period"].max() == 23
    cohort_sizes = clients["first_activation_period"].value_counts()
    assert cohort_sizes.reindex(range(1, 11), fill_value=0).gt(0).all()
    assert clients["first_activation_period"].gt(10).sum() > 0
    assert clients["first_activation_period"].isna().sum() > 0
    assert (
        df.loc[
            df["activation_now"].astype(bool) & df["period"].le(10),
            "n_pre_periods_available",
        ]
        .between(1, 10)
        .all()
    )
    assert oracle["att_aggregate"]["event_time"].tolist() == list(range(6))
    assert df["ltv_monthly"].lt(0).any()


def test_weak_overlap_exposes_extreme_probabilities_without_deterministic_hazards():
    df, oracle, _ = generate_data(n_clients=3500, scenario="weak_overlap", seed=42)
    hazard = oracle["potential_outcomes"].loc[
        df["eligible_to_activate"].astype(bool), "hazard"
    ]
    assert hazard.min() < 0.01
    assert hazard.max() > 0.99
    assert hazard.between(0, 1, inclusive="neither").all()


def test_small_panels_report_empty_cohorts_without_resampling_clients():
    df, oracle, _ = generate_data(n_clients=1, seed=17)
    assert df["client_id"].nunique() == 1
    assert len(oracle["exogenous"]["entry_period"]) == 1
    by_cohort = oracle["att_by_cohort"]
    assert by_cohort["n_treated"].eq(0).any()
    assert (
        by_cohort.loc[by_cohort["n_treated"].eq(0), list(EFFECT_COLUMNS)]
        .isna()
        .all()
        .all()
    )


def test_wrapper_forwards_custom_panel_and_effect_configuration():
    df, oracle, metadata = generate_data(
        n_clients=400,
        n_periods=8,
        horizon=3,
        entry_max=2,
        main_cohorts=(1, 3, 5),
        calendar_start="2022-09",
        effect_base=0.0,
        seed=24,
    )
    assert df["period"].max() == 7
    assert df["entry_period"].between(0, 2).all()
    assert df["date"].min() == pd.Timestamp("2022-09-01")
    assert df["date"].max() == pd.Timestamp("2023-04-01")
    assert metadata["main_cohorts"] == (1, 3, 5)
    assert set(oracle["att_by_cohort"]["g"]) == {1, 3, 5}
    assert oracle["att_aggregate"]["event_time"].tolist() == [0, 1, 2]
    assert oracle["potential_outcomes"]["effect_factual"].eq(0.0).all()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_clients": 0},
        {"n_clients": -3},
        {"n_clients": 1.5},
        {"n_periods": 0},
        {"horizon": 0},
        {"entry_max": -1},
        {"entry_max": 24},
        {"scenario": "unknown"},
        {"main_cohorts": (0, 1)},
        {"main_cohorts": (2, 1)},
        {"main_cohorts": (1, 1)},
        {"main_cohorts": (23,), "horizon": 2},
        {"state_rho": 1.0},
        {"hidden_rho": -1.0},
        {"outcome_noise_df": 2.0},
        {"salary_probability": 0.0},
        {"segment_probabilities": (0.5, 0.5, 0.5)},
        {"hazard_intercept": np.nan},
        {"calendar_start": "not-a-date"},
    ],
)
def test_invalid_configuration_fails_explicitly(kwargs):
    with pytest.raises((ValueError, TypeError)):
        OrganicActivationConfig(**kwargs)


def test_configuration_is_immutable():
    config = OrganicActivationConfig()
    with pytest.raises(FrozenInstanceError):
        config.horizon = 2


def test_unknown_dgp_keyword_is_not_silently_ignored():
    with pytest.raises(ValueError, match="Unknown DGP parameters"):
        generate_data(n_clients=10, accidental_parameter=1)
