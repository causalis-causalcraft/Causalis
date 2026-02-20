from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from causalis.dgp.causaldata import CausalData
from causalis.dgp import generate_rct
from causalis.scenarios.unconfoundedness.model import IRM
from causalis.scenarios.unconfoundedness.refutation.unconfoundedness.sensitivity import sensitivity_analysis, get_sensitivity_summary


def _make_cd(n=600, random_state=3, outcome_type="normal"):
    df = generate_rct(n=n, split=0.5, random_state=random_state, outcome_type=outcome_type, k=3, add_ancillary=False)
    y = "y"; d = "d"
    xcols = [c for c in df.columns if c not in {y, d, "m", "m_obs", "tau_link", "g0", "g1", "cate"}]
    return CausalData(df=df[[y, d] + xcols], treatment=d, outcome=y, confounders=xcols)


def test_sensitivity_with_dml_ate_runs_and_returns_dict():
    cd = _make_cd(n=400, random_state=11, outcome_type="normal")
    ml_g = RandomForestRegressor(n_estimators=30, random_state=1)
    ml_m = RandomForestClassifier(n_estimators=30, random_state=1)

    res = IRM(cd, ml_g=ml_g, ml_m=ml_m, n_folds=3).fit()
    res.estimate(score="ATE")
    out = sensitivity_analysis(res, r2_y=0.02, r2_d=0.03, rho=1.0)

    assert isinstance(out, dict)
    # Integration: summary should be retrievable via the getter
    summ = get_sensitivity_summary(res)
    assert isinstance(summ, str)
    assert any(kw in summ for kw in ("Bias-aware Interval", "Intervals"))


def test_sensitivity_with_dml_att_runs_and_returns_dict():
    cd = _make_cd(n=400, random_state=7, outcome_type="normal")
    ml_g = RandomForestRegressor(n_estimators=25, random_state=0)
    ml_m = RandomForestClassifier(n_estimators=25, random_state=0)

    res = IRM(cd, ml_g=ml_g, ml_m=ml_m, n_folds=3).fit()
    res.estimate(score="ATTE")
    out = sensitivity_analysis(res, r2_y=0.01, r2_d=0.04, rho=0.8)

    assert isinstance(out, dict)
    summ = get_sensitivity_summary(res)
    assert isinstance(summ, str)
    assert "Bias-aware Interval" in summ
