"""Classic and CUPED planning: independent power references and allocation invariants."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from scipy.optimize import brentq
from statsmodels.stats.power import NormalIndPower

from causalis.data_contracts import RctCausalData
from causalis.shared.rct_design import calculate_mde, calculate_sample_size


def contract(frame):
    return RctCausalData.from_df(
        frame, ["a", "control", "b"], ["y1", "y2"], ["x1", "x2"],
        control_treatment="control",
    )


@pytest.fixture
def data():
    rng = np.random.default_rng(481)
    arm = np.arange(360) % 3
    x1, x2 = rng.normal(size=(2, len(arm)))
    return contract(pd.DataFrame({
        "control": arm == 0, "a": arm == 1, "b": arm == 2,
        "x1": x1, "x2": x2,
        "y1": 20 + 2 * x1 - x2 + rng.normal(size=len(arm)) + 100 * arm,
        "y2": 50 - x1 + 3 * x2 + 2 * rng.normal(size=len(arm)) - 100 * arm,
    }, index=np.arange(len(arm)) // 2))


COMPACT_COLUMNS = [
    "outcome", "control", "treatment", "mde_relative", "mde_absolute",
    "n_control", "n_treatment", "n_total",
]
DETAIL_COLUMNS = [
    "baseline_mean", "variance_raw", "variance_used", "variance_reduction_pct",
    "n_reference", "alpha", "power", "achieved_power",
]


@pytest.mark.parametrize("covariates", [[], ["x1"], ["x1", "x2"]])
@pytest.mark.parametrize("alpha,power", [(0.05, 0.8), (0.2, 0.3), (0.01, 0.95)])
@pytest.mark.parametrize("outcome", ["y1", "y2"])
def test_ols_variances_and_two_sided_power(data, covariates, alpha, power, outcome):
    shares = {"b": 0.2, "control": 0.5, "a": 0.3}
    targets = [0.2, 0.5, 1.0]
    result = calculate_sample_size(
        data, outcome=outcome, covariates=covariates, scenario="user",
        mde_type="absolute", mde=targets, allocation=shares,
        alpha=alpha, power=power, include_details=True,
    )
    control = data.df.loc[data.df.control == 1]
    ols = sm.OLS(control[outcome], sm.add_constant(control[covariates], has_constant="add")).fit()
    assert list(zip(result.mde_absolute, result.treatment)) == [
        (target, arm) for target in targets for arm in ["a", "b"]
    ]
    for row in result.itertuples():
        assert row.outcome == outcome
        assert row.baseline_mean == pytest.approx(control[outcome].mean())
        assert row.variance_raw == pytest.approx(control[outcome].var(ddof=1))
        assert row.variance_used == pytest.approx(ols.mse_resid)
        assert row.variance_reduction_pct == pytest.approx(100 * (1 - ols.mse_resid / row.variance_raw))
        assert row.n_reference == 120
        # An explicit positive bracket also covers continuous requirements below
        # two users, where statsmodels' default solve_power search can fail.
        requirements = [brentq(lambda total: NormalIndPower().power(
            effect_size=row.mde_absolute / np.sqrt(ols.mse_resid),
            nobs1=total * shares["control"], ratio=shares[arm] / shares["control"],
            alpha=alpha, alternative="two-sided",
        ) - power, 1e-8, 1e9) for arm in ["a", "b"]]
        sizes = {arm: int(np.ceil(max(requirements) * share)) for arm, share in shares.items()}
        assert row.n_control == sizes["control"]
        assert row.n_treatment == sizes[row.treatment]
        assert row.n_total == sum(sizes.values())
        expected_power = NormalIndPower().power(
            effect_size=row.mde_absolute / np.sqrt(ols.mse_resid),
            nobs1=row.n_control, ratio=row.n_treatment / row.n_control,
            alpha=alpha, alternative="two-sided",
        )
        assert row.achieved_power == pytest.approx(expected_power, abs=1e-12)
        assert row.achieved_power >= power - 1e-12
        assert row.mde_relative == pytest.approx(100 * row.mde_absolute / row.baseline_mean)


def test_binary_and_onehot_equivalence(data):
    frame = data.df.loc[(data.df.control == 1) | (data.df.a == 1)]
    binary = RctCausalData.from_df(frame, "a", ["y1", "y2"], ["x1", "x2"])
    onehot = RctCausalData.from_df(
        frame, ["a", "control"], ["y1", "y2"], ["x1", "x2"], control_treatment="control",
    )
    for function, options in [
        (calculate_sample_size, {"outcome": "y1", "include_details": True}),
        (calculate_mde, {"outcome": "y1", "sample_size": 801, "include_details": True}),
    ]:
        result = function(binary, covariates=["x1", "x2"],
                          allocation={"a=1": 0.3, "a=0": 0.7}, **options)
        expected = function(onehot, covariates=["x1", "x2"],
                            allocation={"a": 0.3, "control": 0.7}, **options)
        assert set(result.control) == {"a=0"}
        assert set(result.treatment) == {"a=1"}
        pd.testing.assert_frame_equal(result.drop(columns=["control", "treatment"]),
                                      expected.drop(columns=["control", "treatment"]))


def test_default_scenarios_and_compact_details(data):
    compact = calculate_sample_size(data, outcome="y1")
    detailed = calculate_sample_size(data, outcome="y1", include_details=True)
    explicit = calculate_sample_size(
        data, outcome="y1", covariates=[], scenario="user",
        mde_type="relative", mde=[0.5, 1, 5, 10, 20],
    )
    assert compact.columns.tolist() == COMPACT_COLUMNS
    assert detailed.columns.tolist() == COMPACT_COLUMNS + DETAIL_COLUMNS
    pd.testing.assert_frame_equal(compact, detailed[COMPACT_COLUMNS])
    pd.testing.assert_frame_equal(compact, explicit)
    np.testing.assert_allclose(compact.mde_relative, np.repeat([0.5, 1, 5, 10, 20], 2))
    assert compact.outcome.tolist() == ["y1"] * 10
    assert compact.treatment.tolist() == ["a", "b"] * 5
    assert compact.control.tolist() == ["control"] * 10
    assert all(pd.api.types.is_integer_dtype(detailed[name]) for name in (
        "n_reference", "n_control", "n_treatment", "n_total",
    ))
    assert (detailed.achieved_power >= 0.8 - 1e-12).all()


@pytest.mark.parametrize("scale,targets", [("relative", [5, 0.5, 5, 1]), ("absolute", [1, 0.2, 1])])
def test_scenario_order_and_duplicates(data, scale, targets):
    result = calculate_sample_size(
        data, outcome="y2", scenario="user", mde_type=scale, mde=targets,
        allocation={"control": 0.45, "a": 0.35, "b": 0.2}, include_details=True,
    )
    column = "mde_relative" if scale == "relative" else "mde_absolute"
    np.testing.assert_allclose(result[column], np.repeat(targets, 2))
    assert result.treatment.tolist() == ["a", "b"] * len(targets)
    pd.testing.assert_frame_equal(result.iloc[:2], result.iloc[4:6].reset_index(drop=True))


@pytest.mark.parametrize("relative", [0.5, 1, 5])
def test_planner_percent_inputs_equal_absolute_units(data, relative):
    baseline = data.df.loc[data.df.control == 1, "y1"].mean()
    percent = calculate_sample_size(data, outcome="y1", scenario="user", mde_type="relative", mde=[relative])
    absolute = calculate_sample_size(data, outcome="y1", scenario="user", mde_type="absolute", mde=[baseline * (relative / 100)])
    pd.testing.assert_frame_equal(percent, absolute)
    np.testing.assert_allclose(percent.mde_relative, relative)


def test_planner_fits_once_for_all_targets(data, monkeypatch):
    original = np.linalg.lstsq
    calls = []

    def record(*args, **kwargs):
        calls.append(args[0].shape)
        return original(*args, **kwargs)

    monkeypatch.setattr(np.linalg, "lstsq", record)
    calculate_sample_size(data, outcome="y1", covariates=["x1", "x2"])
    assert calls == [(120, 3)]


def test_active_rows_ignored_determinism_and_input_unchanged(data):
    original = data.df.copy(deep=True)
    frame = data.df.copy(deep=True)
    active = frame.control == 0
    frame.loc[active, ["x1", "x2", "y1", "y2"]] *= -1000
    # Even the historical arm counts may change without affecting planning.
    frame = pd.concat([frame, frame.loc[frame.a == 1]])
    changed = contract(frame)
    for function, options in [
        (calculate_sample_size, {"outcome": "y1", "include_details": True}),
        (calculate_mde, {"outcome": "y1", "sample_size": 1000, "include_details": True}),
    ]:
        first = function(data, covariates=["x1", "x2"], **options)
        pd.testing.assert_frame_equal(first, function(changed, covariates=["x1", "x2"], **options))
        np.random.seed(654)
        state = np.random.get_state()
        pd.testing.assert_frame_equal(first, function(data, covariates=["x1", "x2"], **options))
        after = np.random.get_state()
        np.testing.assert_array_equal(state[1], after[1])
        assert state[2:] == after[2:]
    pd.testing.assert_frame_equal(data.df, original)


def test_control_constant_covariate_warns_and_is_dropped(data):
    frame = data.df.copy()
    frame.loc[frame.control == 1, "x1"] = 3.0
    modified = contract(frame)
    with pytest.warns(UserWarning, match="constant in historical control.*x1"):
        result = calculate_sample_size(modified, covariates=["x1", "x2"], outcome="y1", include_details=True)
    expected = calculate_sample_size(modified, covariates=["x2"], outcome="y1", include_details=True)
    pd.testing.assert_frame_equal(result, expected)


def test_negative_variance_reduction_is_not_clipped(data):
    frame = data.df.copy()
    control = frame.control == 1
    x = frame.loc[control, "x1"].to_numpy(copy=True)
    x -= x.mean()
    y = frame.loc[control, "y1"].to_numpy(copy=True)
    y -= y.mean()
    y -= x * (x @ y) / (x @ x)
    frame.loc[control, "y1"] = 20 + y
    result = calculate_sample_size(contract(frame), covariates=["x1"], outcome="y1", include_details=True)
    assert (result.variance_reduction_pct < 0).all()
    np.testing.assert_allclose(result.variance_used / result.variance_raw, 119 / 118)


@pytest.mark.parametrize("baseline", [0, -20])
def test_nonpositive_baseline_absolute_only(data, baseline):
    frame = data.df.copy()
    control = frame.control == 1
    # Exactly zero sum with nonzero variance, so baseline zero is unambiguous.
    frame.loc[control, "y1"] = np.tile([-1.0, 1.0], control.sum() // 2) + baseline
    modified = contract(frame)
    result = calculate_sample_size(modified, covariates=[], outcome="y1", scenario="user", mde_type="absolute", mde=[0.5])
    assert result.mde_relative.isna().all()
    for options in [{}, {"scenario": "user", "mde_type": "relative", "mde": [1]}]:
        with pytest.raises(ValueError, match="positive historical control mean"):
            calculate_sample_size(modified, outcome="y1", covariates=[], **options)
    sensitivity = calculate_mde(modified, outcome="y1", sample_size=100)
    assert sensitivity.mde_relative.isna().all()
    assert (sensitivity.mde_absolute > 0).all()


@pytest.mark.parametrize("change,error", [
    ("collinear", "rank deficient"),
    ("perfect_fit", "residual variance"),
    ("constant_outcome", "variance"),
    ("no_degrees", "degrees of freedom"),
    ("one_control", "at least two"),
])
def test_invalid_historical_control(data, change, error):
    frame = data.df.copy()
    control = frame.control == 1
    if change == "collinear":
        frame.loc[control, "x2"] = 2 * frame.loc[control, "x1"] + 3
    elif change == "perfect_fit":
        frame.loc[control, "y1"] = 2 * frame.loc[control, "x1"] + 3
    elif change == "constant_outcome":
        frame.loc[control, "y1"] = 20
    else:
        count = 3 if change == "no_degrees" else 1
        frame = pd.concat([frame.loc[control].iloc[:count], frame.loc[~control]])
    with pytest.raises(ValueError, match=error):
        calculate_sample_size(contract(frame), covariates=["x1", "x2"], outcome="y1", include_details=True)


@pytest.mark.parametrize("options,error", [
    ({"covariates": "x1"}, "sequence"),
    ({"covariates": np.array(["x1"])}, "sequence"),
    ({"covariates": ["x1", "x1"]}, "duplicates"),
    ({"covariates": [1]}, "strings"),
    ({"covariates": ["y1"]}, "subset"),
    ({"outcome": ["y1"]}, "one declared outcome"),
    ({"outcome": "missing"}, "one declared outcome"),
    ({"outcome": None}, "one declared outcome"),
    ({"outcome": True}, "one declared outcome"),
    ({"outcome": ""}, "one declared outcome"),
    ({"include_details": "yes"}, "boolean"),
    ({"scenario": "other"}, "scenario"),
    ({"scenario": True}, "scenario"),
    ({"mde": [1]}, "does not accept"),
    ({"mde_type": "relative"}, "does not accept"),
    ({"alpha": 0.8, "power": 0.5}, "alpha < power"),
    ({"alpha": np.nan}, "finite positive"),
    ({"alpha": False}, "finite positive"),
    ({"power": 1}, "alpha < power"),
    ({"power": float("inf")}, "finite positive"),
    ({"allocation": [0.5, 0.25, 0.25]}, "exactly"),
    ({"allocation": {"control": 0.5, "a": 0.5}}, "exactly"),
    ({"allocation": {"control": 0.5, "a": 0.5, "b": 0}}, "finite positive"),
    ({"allocation": {"control": 0.5, "a": 0.5, "b": 0.5}}, "sum to 1"),
])
def test_invalid_mde_arguments(data, options, error):
    kwargs = {"covariates": ["x1"], "outcome": "y1", **options}
    with pytest.raises(ValueError, match=error):
        calculate_sample_size(data, **kwargs)


@pytest.mark.parametrize("options,error", [
    ({"mde": None}, "non-empty sequence"),
    ({"mde": []}, "non-empty sequence"),
    ({"mde": 1}, "non-empty sequence"),
    ({"mde": "1"}, "non-empty sequence"),
    ({"mde": {"y1": 1}}, "non-empty sequence"),
    ({"mde": [0]}, "finite positive"),
    ({"mde": [-1]}, "finite positive"),
    ({"mde": [True]}, "finite positive"),
    ({"mde": [np.bool_(False)]}, "finite positive"),
    ({"mde": ["1"]}, "finite positive"),
    ({"mde": [np.nan]}, "finite positive"),
    ({"mde": [np.inf]}, "finite positive"),
    ({"mde": [1e-300]}, "too large"),
    ({"mde_type": None}, "requires mde_type"),
    ({"mde_type": "percent"}, "requires mde_type"),
    ({"mde_type": True}, "requires mde_type"),
    ({"mde_type": "relative", "mde": [np.nextafter(0., 1.)]}, "convert to finite positive"),
])
def test_invalid_user_scenarios(data, options, error):
    kwargs = {"scenario": "user", "mde_type": "absolute", "mde": [1], **options}
    with pytest.raises(ValueError, match=error):
        calculate_sample_size(data, outcome="y1", **kwargs)


def test_required_outcome_and_removed_arguments(data):
    with pytest.raises(TypeError, match="outcome"):
        calculate_sample_size(data)
    for options in [{"outcomes": ["y1"]}, {"sample_size": 1000}]:
        with pytest.raises(TypeError, match="unexpected keyword"):
            calculate_sample_size(data, outcome="y1", **options)


def test_only_selected_outcome_enters_fit(data):
    frame = data.df.copy()
    frame.loc[frame.control == 1, "y2"] = 0
    pd.testing.assert_frame_equal(
        calculate_sample_size(contract(frame), outcome="y1"),
        calculate_sample_size(data, outcome="y1"),
    )


def test_no_declared_confounders_and_tuple_targets(data):
    bare = RctCausalData.from_df(
        data.df, ["a", "control", "b"], ["y1"], [], control_treatment="control",
    )
    with pytest.raises(TypeError, match="outcome"):
        calculate_sample_size(bare)
    default = calculate_sample_size(bare, outcome="y1")
    explicit = calculate_sample_size(
        bare, outcome="y1", covariates=[], scenario="user", mde_type="relative",
        mde=(0.5, 1, 5, 10, 20),
    )
    pd.testing.assert_frame_equal(default, explicit)


def test_contract_type_required(data):
    with pytest.raises(TypeError, match="RctCausalData"):
        calculate_sample_size(data.df, outcome="y1")
    with pytest.raises(TypeError, match="RctCausalData"):
        calculate_mde(data.df, outcome="y1", sample_size=100)


def test_binary_outcome_and_minimum_future_group_sizes(data):
    frame = data.df.copy()
    frame["y1"] = (frame.x1 + frame.x2 > 0).astype(int)
    modified = contract(frame)
    mdes = calculate_sample_size(modified, covariates=["x1"], outcome="y1", include_details=True)
    assert (mdes.mde_absolute > 0).all()
    sensitivity = calculate_mde(modified, outcome="y1", sample_size=1000)
    assert (sensitivity.mde_absolute > 0).all()
    planner = calculate_sample_size(
        modified, outcome="y1", covariates=[], scenario="user", mde_type="absolute",
        mde=[100], include_details=True,
    )
    assert planner.n_control.tolist() == [1, 1]
    assert planner.n_treatment.tolist() == [1, 1]
    assert planner.n_total.tolist() == [3, 3]
    assert (planner.achieved_power >= 0.8).all()


@pytest.mark.parametrize("covariates", [None, [], ["x1"], ["x1", "x2"]])
@pytest.mark.parametrize("alpha,power", [(0.05, 0.8), (0.2, 0.3), (0.01, 0.95)])
def test_mde_matches_independent_power(data, covariates, alpha, power):
    result = calculate_mde(
        data, outcome="y2", sample_size=101, covariates=covariates,
        allocation={"b": 0.2, "control": 0.5, "a": 0.3},
        alpha=alpha, power=power, include_details=True,
    )
    control = data.df.loc[data.df.control == 1]
    variance = (
        sm.OLS(control.y2, sm.add_constant(control[covariates])).fit().mse_resid
        if covariates else control.y2.var(ddof=1)
    )
    assert result.columns.tolist() == COMPACT_COLUMNS + DETAIL_COLUMNS[:-1]
    assert result.treatment.tolist() == ["a", "b"]
    assert result.n_control.tolist() == [51, 51]
    assert result.n_treatment.tolist() == [30, 20]
    assert result.n_total.tolist() == [101, 101]
    for row in result.itertuples():
        assert row.variance_used == pytest.approx(variance)
        assert row.baseline_mean == pytest.approx(control.y2.mean())
        assert row.mde_relative == pytest.approx(100 * row.mde_absolute / control.y2.mean())
        achieved = NormalIndPower().power(
            effect_size=row.mde_absolute / np.sqrt(variance), nobs1=row.n_control,
            ratio=row.n_treatment / row.n_control, alpha=alpha, alternative="two-sided",
        )
        assert achieved == pytest.approx(power, abs=1e-12)


@pytest.mark.parametrize("function,options", [
    (calculate_sample_size, {}), (calculate_mde, {"sample_size": 1000}),
])
def test_unadjusted_defaults_skip_regression_and_ignore_confounders(data, monkeypatch, function, options):
    def forbidden(*args, **kwargs):
        pytest.fail("Unadjusted planning must not fit a regression.")

    monkeypatch.setattr(np.linalg, "lstsq", forbidden)
    baseline = function(data, outcome="y1", include_details=True, **options)
    for covariates in (None, []):
        pd.testing.assert_frame_equal(
            baseline, function(data, outcome="y1", covariates=covariates, include_details=True, **options),
        )
    frame = data.df.copy()
    frame.loc[frame.control == 1, "x2"] = 2 * frame.loc[frame.control == 1, "x1"] + 3
    pd.testing.assert_frame_equal(
        baseline, function(contract(frame), outcome="y1", include_details=True, **options),
    )
    np.testing.assert_allclose(baseline.variance_used, data.df.loc[data.df.control == 1, "y1"].var(ddof=1))
    assert (baseline.variance_used == baseline.variance_raw).all()
    assert (baseline.variance_reduction_pct == 0).all()
    compact = function(data, outcome="y1", **options)
    pd.testing.assert_frame_equal(compact, baseline[COMPACT_COLUMNS])


@pytest.mark.parametrize("total,sizes", [(3, [1, 1, 1]), (10, [4, 3, 3]), (11, [4, 4, 3]), (12, [4, 4, 4])])
def test_mde_exact_total_and_tie_order(data, total, sizes):
    result = calculate_mde(data, outcome="y1", sample_size=total)
    assert result.n_control.tolist() == [sizes[0]] * 2
    assert result.n_treatment.tolist() == sizes[1:]
    assert result.n_total.tolist() == [total] * 2
    assert all(pd.api.types.is_integer_dtype(result[name]) for name in ["n_control", "n_treatment", "n_total"])


@pytest.mark.parametrize("sample_size", [0, -1, 10., True, np.bool_(True), None, [100], (50, 50)])
def test_invalid_sample_size(data, sample_size):
    with pytest.raises(ValueError, match="positive integer"):
        calculate_mde(data, outcome="y1", sample_size=sample_size)


@pytest.mark.parametrize("options,error", [
    ({"sample_size": 2}, "every group"),
    ({"sample_size": 2**53 + 1}, "too large"),
    ({"allocation": {"control": 0.999998, "a": 0.000001, "b": 0.000001}}, "every group"),
    ({"allocation": {"control": 0.5, "a": 0.5}}, "exactly"),
    ({"allocation": {"control": 0.5, "a": 0.4, "b": 0.4}}, "sum to 1"),
    ({"outcome": ["y1"]}, "one declared outcome"),
    ({"outcome": "missing"}, "one declared outcome"),
    ({"include_details": 1}, "boolean"),
    ({"alpha": 0.8, "power": 0.5}, "alpha < power"),
    ({"covariates": ["y1"]}, "subset"),
])
def test_invalid_mde_inputs(data, options, error):
    kwargs = {"outcome": "y1", "sample_size": 1000, **options}
    with pytest.raises(ValueError, match=error):
        calculate_mde(data, **kwargs)


@pytest.mark.parametrize("covariates", [None, ["x1", "x2"]])
def test_sample_size_and_mde_consistency(data, covariates):
    for target in [0.2, 0.5, 1]:
        sizes = calculate_sample_size(
            data, outcome="y1", covariates=covariates,
            scenario="user", mde_type="absolute", mde=[target],
        )
        mdes = calculate_mde(
            data, outcome="y1", covariates=covariates, sample_size=int(sizes.n_total.iloc[0]),
        )
        assert (mdes.mde_absolute <= target + 1e-12).all()
    original = calculate_mde(data, outcome="y1", covariates=covariates, sample_size=1200)
    sizes = calculate_sample_size(
        data, outcome="y1", covariates=covariates, scenario="user",
        mde_type="absolute", mde=[float(original.mde_absolute.iloc[0])],
    )
    assert 1200 <= sizes.n_total.iloc[0] <= 1203


@pytest.mark.parametrize("function,options", [
    (calculate_sample_size, {}), (calculate_mde, {"sample_size": 1000}),
])
def test_all_constant_covariates_fall_back_to_classic(data, function, options):
    frame = data.df.copy()
    frame.loc[frame.control == 1, "x1"] = 3
    modified = contract(frame)
    with pytest.warns(UserWarning, match="constant in historical control"):
        result = function(modified, outcome="y1", covariates=["x1"], include_details=True, **options)
    pd.testing.assert_frame_equal(result, function(modified, outcome="y1", include_details=True, **options))


@pytest.mark.parametrize("first_import", ["causalis.shared.rct_design", "causalis.scenarios.classic_rct", "causalis.scenarios.cuped"])
@pytest.mark.parametrize("runner_directory", [".", "tests"])
def test_public_imports_and_removed_apis(first_import, runner_directory, monkeypatch):
    repo_root = Path(__file__).resolve().parents[2]
    monkeypatch.chdir(repo_root / runner_directory)
    if runner_directory == "tests":
        # IDE runners can expose tests/statistics as the top-level statistics module.
        monkeypatch.setenv("PYTHONPATH", str(repo_root / "tests"))
    script = f"""
import importlib
import sys
sys.path.insert(0, {str(repo_root)!r})
importlib.import_module({first_import!r})
from causalis.shared.rct_design import calculate_mde, calculate_sample_size, check_srm, SRMResult
from causalis.shared.srm import check_srm as shared_srm
assert check_srm is shared_srm
assert callable(calculate_mde) and callable(calculate_sample_size)
cuped = importlib.import_module('causalis.scenarios.cuped')
assert not hasattr(cuped, 'calculate_cuped_mde')
assert not hasattr(cuped, 'calculate_cuped_sample_size')
for removed in ['causalis.scenarios.cuped.design', 'causalis.shared.rct_design.mde']:
    try:
        importlib.import_module(removed)
    except ModuleNotFoundError:
        pass
    else:
        raise AssertionError(removed + ' still exists')
"""
    # Ignore the runner's cwd and PYTHONPATH while testing this checkout's imports.
    result = subprocess.run([sys.executable, "-I", "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_mde_rejects_removed_and_wrong_direction_arguments(data):
    with pytest.raises(TypeError, match="outcome"):
        calculate_mde(data, sample_size=100)
    with pytest.raises(TypeError, match="sample_size"):
        calculate_mde(data, outcome="y1")
    for options in [{"scenario": "default"}, {"mde": [1]}, {"baseline_rate": 0.1}, {"variance": 4}]:
        with pytest.raises(TypeError, match="unexpected keyword"):
            calculate_mde(data, outcome="y1", sample_size=100, **options)
