"""Outcome-model enrichment with separate training and effect-evaluation users."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.base import clone

from causalis.data_contracts import CausalData, RctCausalData
from causalis.shared.confounders_balance import _compute_balance_table
from causalis.shared.srm import SRMResult, check_srm

from .split import assign_variants_df


@dataclass
class EnrichmentDiagnostics:
    """Counts, assignment checks, and balance for a frozen enrichment rule.

    ``counts`` reports users and conversions separately for each arm and sample.
    SRM uses the caller's planned allocation, never an observed allocation.
    ``srm_selected`` is None when no evaluation users meet the threshold.
    ``balance`` includes all original covariates and the predicted outcome score,
    even covariates that are constant within the selected sample. Diagnostics do
    not establish that covariates were recorded before treatment or that the
    evaluation outcomes have never been used for model/rule selection.
    """

    counts: pd.DataFrame
    srm_full: SRMResult
    srm_evaluation: SRMResult
    srm_selected: SRMResult | None
    balance: pd.DataFrame
    threshold: float
    selected_fraction: float
    warnings: tuple[str, ...]


def _probability(value: Any, name: str, *, open_interval: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number.")
    number = float(value)
    valid = 0 < number < 1 if open_interval else 0 <= number <= 1
    if not np.isfinite(number) or not valid:
        interval = "(0, 1)" if open_interval else "[0, 1]"
        raise ValueError(f"{name} must be in {interval}.")
    return number


def _predict_scores(
    model: Any, features: pd.DataFrame, positive_class: int
) -> np.ndarray:
    probabilities = np.asarray(model.predict_proba(features), dtype=float)
    if (
        probabilities.shape != (len(features), 2)
        or not np.isfinite(probabilities).all()
    ):
        raise ValueError(
            "predict_proba must return a finite array of shape (n_users, 2)."
        )
    if (
        (probabilities < 0).any()
        or (probabilities > 1).any()
        or not np.allclose(probabilities.sum(axis=1), 1)
    ):
        raise ValueError(
            "predict_proba must return probabilities in [0, 1] summing to 1."
        )
    return probabilities[:, positive_class]


class PrognosticEnrichment:
    r"""Learn a binary-outcome selection rule on train; evaluate on held-out users.

    A deterministic hash of ``user_id`` and ``split_seed`` splits the entire
    experiment independently of treatment labels and outcomes. The estimator is
    cloned and fitted only on the specified arm of the training sample. One
    threshold is applied to both evaluation arms. No training user is returned
    by :meth:`select`.

    Parameters
    ----------
    estimator : sklearn-compatible classifier
        Must implement ``fit``, ``predict_proba``, and fitted ``classes_``.
        Pipelines are supported. The original estimator is never fitted.
        Set the estimator's own random seed for reproducible model fitting.
    training_source : {"treatment", "control"}, default "treatment"
        Treatment training learns :math:`P(Y(1)=1\mid X)`, while control training
        learns :math:`P(Y(0)=1\mid X)`. Neither is an uplift prediction.
    train_fraction : float, default 0.3
        Probability of assignment to train; actual sample sizes may differ.
    top_fraction : float or None, default 0.2
        Set the threshold to the ``1 - top_fraction`` quantile of predictions
        over BOTH training arms. This is a training reference, not a separate
        per-arm ranking. Threshold ties are included, so the selected share may
        exceed this fraction. The evaluation share may also differ under drift.
        Set to None when supplying an explicit ``threshold``.
    threshold : float or None, default None
        Fixed probability cutoff in [0, 1], as an alternative to top_fraction.
    split_seed : int, default 42
        Controls the independent analysis split, not the experimental assignment
        or the estimator's randomness. Do not search seeds using evaluation results.

    Notes
    -----
    Features must be measured before experimental assignment. This cannot be
    verified from column names. All preprocessing must be fitted within the
    estimator/pipeline on train; do not pass preprocessing fitted on evaluation.
    Evaluation outcomes must not be used to tune models, cutoffs, sample splits,
    or balance/SRM. Split independence assumes independently randomized users;
    cluster/block randomization needs a design-aware split and inference.

    The target effect is conditional on the learned selected population, not the
    original audience. SRM and covariate balance hold in expectation under ordinary
    randomization, not necessarily in every realized sample. Selecting on a
    prognostic score does not guarantee improved power.

    Examples
    --------
    >>> from sklearn.linear_model import LogisticRegression
    >>> rule = PrognosticEnrichment(LogisticRegression(max_iter=1000))
    >>> rule.fit(data, features=["prior_activity", "prior_spend"])
    >>> selected = rule.select()
    >>> diagnostics = rule.diagnostics(target_allocation={0: 0.5, 1: 0.5})
    >>> effect = DiffInMeans().fit(selected).estimate(method="conversion_ztest")
    """

    def __init__(
        self,
        estimator: Any,
        *,
        training_source: str = "treatment",
        train_fraction: float = 0.3,
        top_fraction: float | None = 0.2,
        threshold: float | None = None,
        split_seed: int = 42,
    ) -> None:
        self.estimator = estimator
        self.training_source = training_source
        self.train_fraction = train_fraction
        self.top_fraction = top_fraction
        self.threshold = threshold
        self.split_seed = split_seed
        self._is_fitted = False
        self._validate_options()

    def _validate_options(self) -> None:
        if self.training_source not in ("treatment", "control"):
            raise ValueError("training_source must be 'treatment' or 'control'.")
        _probability(self.train_fraction, "train_fraction", open_interval=True)
        if (self.top_fraction is None) == (self.threshold is None):
            raise ValueError("Specify exactly one of top_fraction and threshold.")
        if self.top_fraction is not None:
            fraction = _probability(self.top_fraction, "top_fraction")
            if fraction == 0:
                raise ValueError("top_fraction must be in (0, 1].")
        if self.threshold is not None:
            _probability(self.threshold, "threshold")
        if isinstance(self.split_seed, bool) or not isinstance(
            self.split_seed, Integral
        ):
            raise ValueError("split_seed must be an integer.")
        if not callable(getattr(self.estimator, "fit", None)) or not callable(
            getattr(self.estimator, "predict_proba", None)
        ):
            raise ValueError("estimator must implement fit and predict_proba.")

    def fit(
        self,
        data: CausalData | RctCausalData,
        *,
        features: Sequence[str],
        outcome: str | None = None,
    ) -> PrognosticEnrichment:
        """Freeze a model and threshold without using evaluation outcomes.

        ``features`` must explicitly select declared pre-treatment confounders.
        A unique, non-missing ``user_id`` is required. For ``RctCausalData`` with
        multiple outcomes, select one ``outcome`` explicitly. Only a single
        binary treatment column (0=control, 1=treatment) is supported.
        """
        self._is_fitted = False
        self._validate_options()
        if isinstance(data, RctCausalData):
            if len(data.treatment_names) != 1:
                raise ValueError(
                    "Enrichment requires a single binary treatment column."
                )
            if outcome is None:
                if len(data.outcome_names) != 1:
                    raise ValueError("Select one outcome explicitly for RctCausalData.")
                outcome = data.outcome_names[0]
            data = data.for_outcome(outcome)
        if not isinstance(data, CausalData):
            raise ValueError(
                "data must be CausalData or binary-treatment RctCausalData."
            )
        if outcome is not None and outcome != data.outcome_name:
            raise ValueError("outcome must match the declared CausalData outcome.")
        if data.user_id_name is None:
            raise ValueError(
                "Enrichment requires user_id for a disjoint analysis split."
            )
        if isinstance(features, (str, bytes)) or not isinstance(features, Sequence):
            raise ValueError("features must be a non-empty sequence of column names.")
        feature_names = list(features)
        if not feature_names or any(
            not isinstance(name, str) for name in feature_names
        ):
            raise ValueError("features must be a non-empty sequence of column names.")
        if len(set(feature_names)) != len(feature_names):
            raise ValueError("features must not contain duplicates.")
        if not set(feature_names).issubset(data.confounders_names):
            raise ValueError(
                "features must be declared pre-treatment confounders, excluding treatment/outcome/user_id."
            )

        frame = data.df.copy(deep=True)
        ids = frame[data.user_id_name]
        if ids.isna().any() or ids.astype(str).duplicated().any():
            raise ValueError(
                "user_id must be unique and non-missing, including after string conversion."
            )
        x = frame[feature_names]
        if not np.isfinite(x.to_numpy(dtype=float)).all():
            raise ValueError("features must contain finite numeric values.")
        y = frame[data.outcome_name]
        if not y.isin([0, 1]).all():
            raise ValueError(
                "Enrichment requires a binary utilization outcome in {0, 1}."
            )

        partition = assign_variants_df(
            frame[[data.user_id_name]].rename(columns={data.user_id_name: "user_id"}),
            id_col="user_id",
            experiment_id=f"prognostic_enrichment:{int(self.split_seed)}",
            variants={
                "train": float(self.train_fraction),
                "evaluation": 1 - float(self.train_fraction),
            },
            salt="causalis_prognostic_enrichment",
            layer_id="analysis_holdout",
            variant_col="sample",
        )
        train = partition["sample"].to_numpy() == "train"
        if not train.any() or train.all():
            raise ValueError(
                "The analysis split must contain both train and evaluation users."
            )
        training_arm = 1 if self.training_source == "treatment" else 0
        training = train & (frame[data.treatment_name].to_numpy() == training_arm)
        if not training.any() or y.iloc[np.flatnonzero(training)].nunique() != 2:
            raise ValueError(
                "The chosen training arm must contain both binary outcome classes; use more training data."
            )

        fitted_model = clone(self.estimator)
        fitted_model.fit(
            x.iloc[np.flatnonzero(training)], y.iloc[np.flatnonzero(training)]
        )
        classes = np.asarray(getattr(fitted_model, "classes_", []))
        if classes.shape != (2,) or set(classes.tolist()) != {0, 1}:
            raise ValueError(
                "The fitted classifier must expose binary classes_ containing 0 and 1."
            )
        positive_class = int(np.flatnonzero(classes == 1)[0])
        scores = np.empty(len(frame), dtype=float)
        # Freeze the threshold before even scoring evaluation features. Separate
        # prediction batches also isolate batch-dependent user-supplied models.
        scores[train] = _predict_scores(
            fitted_model, x.iloc[np.flatnonzero(train)], positive_class
        )
        cutoff = (
            float(self.threshold)
            if self.threshold is not None
            else float(np.quantile(scores[train], 1 - float(self.top_fraction)))
        )
        scores[~train] = _predict_scores(
            fitted_model, x.iloc[np.flatnonzero(~train)], positive_class
        )
        partition["score"] = scores
        partition["selected"] = ~train & (scores >= cutoff)

        self.model_ = fitted_model
        self.threshold_ = cutoff
        self.features_ = feature_names
        self.training_source_ = self.training_source
        self._frame = frame
        self._partition = partition
        self._treatment = data.treatment_name
        self._outcome = data.outcome_name
        self._confounders = list(data.confounders_names)
        self._user_id = data.user_id_name
        self._is_fitted = True
        return self

    def _require_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError(
                "Call fit before selecting users or requesting diagnostics."
            )

    def get_partition(self) -> pd.DataFrame:
        """Return user IDs, train/evaluation labels, scores, and selection flags.

        The ID column is always named ``user_id``, regardless of the input name.
        The returned copy preserves input row order and index. Assignment and
        selection do not depend on input row order. ``selected`` is always False
        for training users. Mutating the returned frame cannot alter the rule.
        """
        self._require_fitted()
        return self._partition.copy(deep=True)

    def get_selected_df(self) -> pd.DataFrame:
        """Return selected evaluation rows, including when no inference is possible."""
        self._require_fitted()
        mask = self._partition["selected"].to_numpy()
        return self._frame.iloc[np.flatnonzero(mask)].copy(deep=True)

    def select(self) -> CausalData:
        """Return selected evaluation users for existing RCT estimators.

        Constant confounders in the segment are omitted to satisfy CausalData;
        diagnostic balance still includes them. Both arms and a non-constant
        outcome must be present. With no events, use diagnostics/get_selected_df
        to inspect the segment rather than silently changing the cutoff.
        """
        frame = self.get_selected_df()
        if frame[self._treatment].nunique() != 2:
            raise ValueError(
                "Selected evaluation must contain both treatment arms; inspect diagnostics without changing the rule on evaluation."
            )
        if frame[self._outcome].nunique() != 2:
            raise ValueError(
                "Selected evaluation outcome is constant; inspect diagnostics/get_selected_df for sparse-event inference."
            )
        confounders = [name for name in self._confounders if frame[name].nunique() > 1]
        return CausalData.from_df(
            frame,
            treatment=self._treatment,
            outcome=self._outcome,
            confounders=confounders,
            user_id=self._user_id,
        )

    def diagnostics(
        self,
        *,
        target_allocation: dict[int, float],
        alpha: float = 1e-3,
    ) -> EnrichmentDiagnostics:
        """Check the frozen rule using the original planned binary allocation.

        This does not adjust the cutoff or repair a failing check. Empty or
        single-arm selections remain inspectable. Few successes/non-successes
        are flagged because ordinary conversion z-tests use an approximation.
        """
        self._require_fitted()
        if set(target_allocation) != {0, 1}:
            raise ValueError(
                "target_allocation must specify the original probabilities for arms 0 and 1."
            )
        treatment = self._frame[self._treatment].to_numpy()
        outcome = self._frame[self._outcome].to_numpy()
        train = self._partition["sample"].to_numpy() == "train"
        selected = self._partition["selected"].to_numpy()
        samples = {"train": train, "evaluation": ~train, "selected": selected}
        counts = pd.DataFrame(
            [
                {
                    "sample": sample,
                    "treatment": arm,
                    "n_users": int((mask & (treatment == arm)).sum()),
                    "n_events": int(outcome[mask & (treatment == arm)].sum()),
                }
                for sample, mask in samples.items()
                for arm in (0, 1)
            ]
        )
        srm_full = check_srm(treatment, target_allocation, alpha=alpha)
        srm_evaluation = check_srm(treatment[~train], target_allocation, alpha=alpha)
        srm_selected = (
            check_srm(treatment[selected], target_allocation, alpha=alpha)
            if selected.any()
            else None
        )

        messages = []
        selected_counts = counts.loc[counts["sample"] == "selected"]
        if not selected.any():
            messages.append("No evaluation users meet the frozen threshold.")
        if (selected_counts["n_users"] == 0).any():
            messages.append("Selected evaluation does not contain both treatment arms.")
        if (selected_counts["n_events"] < 5).any() or (
            selected_counts["n_users"] - selected_counts["n_events"] < 5
        ).any():
            messages.append(
                "Selected evaluation has few events or non-events; conversion z-test inference may be unreliable."
            )
        for label, result in (
            ("full", srm_full),
            ("evaluation", srm_evaluation),
            ("selected", srm_selected),
        ):
            if result is not None and result.is_srm:
                messages.append(
                    f"SRM detected in {label}; investigate assignment/selection rather than tuning the cutoff."
                )
        score_name = "prognostic_score"
        while score_name in self._frame.columns:
            score_name += "_"
        balance_frame = self._frame.iloc[np.flatnonzero(selected)].copy(deep=True)
        balance_frame[score_name] = self._partition["score"].to_numpy()[selected]
        if (selected_counts["n_users"] >= 2).all():
            balance = _compute_balance_table(
                balance_frame,
                self._confounders + [score_name],
                treatment[selected] == 0,
                treatment[selected] == 1,
            )
            # A zero within-arm variance is not evidence of balance when means
            # differ. Make such degenerate separation explicit in this report.
            different = balance["abs_diff"] > 0
            degenerate = different & (balance["smd"] == 0)
            balance.loc[degenerate, "smd"] = (
                np.sign(
                    balance.loc[degenerate, "mean_d_1"]
                    - balance.loc[degenerate, "mean_d_0"]
                )
                * np.inf
            )
            balance = balance.loc[
                balance["smd"].abs().sort_values(ascending=False).index
            ].reset_index(drop=True)
        else:
            balance = pd.DataFrame(
                columns=[
                    "confounders",
                    "mean_d_0",
                    "mean_d_1",
                    "abs_diff",
                    "smd",
                    "ks_pvalue",
                ]
            )
            messages.append("Balance requires at least two selected users per arm.")
        return EnrichmentDiagnostics(
            counts=counts,
            srm_full=srm_full,
            srm_evaluation=srm_evaluation,
            srm_selected=srm_selected,
            balance=balance,
            threshold=self.threshold_,
            selected_fraction=float(selected.sum() / (~train).sum()),
            warnings=tuple(messages),
        )
