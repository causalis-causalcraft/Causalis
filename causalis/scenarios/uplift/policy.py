"""Interpretable treatment decisions learned from cross-fitted IRM signals."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from numbers import Integral
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted

from causalis.scenarios._orthogonal import _compute_dr_signal_from_irm
from causalis.scenarios.gate.model import _estimate_gate_groupwise_inference
from causalis.scenarios.unconfoundedness.model import IRM


@dataclass(frozen=True)
class UpliftPolicyEvaluation:
    r"""Independent evaluation of a frozen policy, in outcome units.

    Let :math:`\widehat\pi(Z)\in\{0,1\}` be a policy learned on training
    clients, where :math:`Z` contains the policy features. On independent
    evaluation clients, ``summary()`` estimates policy gains and
    ``rules_summary()`` estimates effects within each fixed leaf.

    The causal policy value is

    .. math::

        V(\pi) = \mathbb{E}\!\left[
            \pi(Z)Y(1) + (1-\pi(Z))Y(0)\right].

    Reports contain differences in this value, not the absolute expected outcome.
    Higher outcomes are better; no treatment cost is subtracted.

    Attributes
    ----------
    diagnostics : dict
        Training and evaluation overlap settings, clipping counts and fractions,
        sample sizes, and inference conventions. Clipping can introduce bias.
    alpha : float
        Significance level; intervals have nominal coverage :math:`1-\alpha`.

    Notes
    -----
    Approximate inference assumes unconfoundedness, adequate overlap, suitable
    nuisance estimation, and independent observations. Rule intervals are
    pointwise HC3 intervals, not individual-effect or simultaneous intervals.
    Evaluation conditions on the learned policy; it does not quantify variation
    from retraining or justify tuning the policy using evaluation results.
    """

    _summary_table: pd.DataFrame
    _rules_table: pd.DataFrame
    diagnostics: dict[str, Any]
    alpha: float

    def summary(self) -> pd.DataFrame:
        r"""Return policy gains, standard errors, intervals, and targeting share.

        On the evaluation sample of size :math:`n`, let
        :math:`a_i=\widehat\pi(Z_i)` and let :math:`\Gamma_i` be the
        cross-fitted doubly robust signal defined in :class:`UpliftPolicyTree`.
        Each comparison uses a paired per-client signal:

        .. math::

            S_i^{\mathrm{policy\_vs\_none}} &= a_i\Gamma_i, \\
            S_i^{\mathrm{policy\_vs\_all}} &= (a_i-1)\Gamma_i, \\
            S_i^{\mathrm{all\_vs\_none}} &= \Gamma_i.

        For each signal :math:`S_i`, the reported estimate and standard error are

        .. math::

            \widehat\Delta = \overline S = \frac{1}{n}\sum_{i=1}^n S_i,
            \qquad
            \widehat{\mathrm{SE}}(\widehat\Delta)
            = \sqrt{\frac{\sum_{i=1}^n(S_i-\overline S)^2}{n(n-1)}}.

        The approximate normal interval is

        .. math::

            \widehat\Delta \;\pm\;
            \Phi^{-1}(1-\alpha/2)\widehat{\mathrm{SE}}(\widehat\Delta).

        Returns
        -------
        pandas.DataFrame
            A copy with one row per ``comparison``. ``value`` is the average
            gain per evaluation client, including clients the policy does not
            treat; it is not the average among targeted clients alone.
            ``std_error``, ``ci_lower``, and ``ci_upper`` describe that gain.
            ``n_obs`` is the full evaluation size. ``treatment_fraction`` is
            :math:`n^{-1}\sum_i a_i`, the learned policy's targeting share in
            every row, not the historical treatment rate or the baseline share.

        Notes
        -----
        Positive ``policy_vs_all`` favors the policy over treating everybody;
        negative values favor treating everybody. An interval crossing zero
        does not establish the direction of the gain.
        """
        return self._summary_table.copy(deep=True)

    def rules_summary(self) -> pd.DataFrame:
        r"""Return effects and support for every frozen rule.

        A leaf :math:`\ell` defines a region :math:`R_\ell` by the conjunction
        of its predicates. Its evaluation clients are
        :math:`I_\ell=\{i:Z_i\in R_\ell\}`, with :math:`n_\ell=|I_\ell|`.
        The target is the group average treatment effect

        .. math::

            \tau_\ell = \mathbb{E}[Y(1)-Y(0)\mid Z\in R_\ell],
            \qquad
            \widehat\tau_\ell = \frac{1}{n_\ell}
                \sum_{i\in I_\ell}\Gamma_i.

        For a supported leaf, HC3 inference for the saturated group regression
        gives

        .. math::

            \widehat{\mathrm{SE}}_{\mathrm{HC3}}(\widehat\tau_\ell)
            = \sqrt{\frac{\sum_{i\in I_\ell}
                    (\Gamma_i-\widehat\tau_\ell)^2}{(n_\ell-1)^2}},
            \qquad
            \mathrm{CI}_\ell = \widehat\tau_\ell \;\pm\;
                \Phi^{-1}(1-\alpha/2)
                \widehat{\mathrm{SE}}_{\mathrm{HC3}}(\widehat\tau_\ell).

        Returns
        -------
        pandas.DataFrame
            A copy with one row per training leaf, including leaves with no
            evaluation clients. Columns are:

            * ``rule_id``: stable identifier of the learned leaf.
            * ``conditions``: readable conjunction defining its region.
            * ``predicates``: exact ``(feature, operator, threshold)`` tuples.
            * ``action``: training decision :math:`a_\ell`, either 1 (target)
              or 0 (do not target); evaluation never changes it.
            * ``n_eval``: :math:`n_\ell`.
            * ``n_treated``: :math:`\sum_{i\in I_\ell}D_i`.
            * ``n_control``: :math:`\sum_{i\in I_\ell}(1-D_i)`.
            * ``status``: ``ok`` when both historical arms are present,
              ``empty`` when :math:`n_\ell=0`, or
              ``insufficient_arm_support`` when one historical arm is absent.
              ``ok`` is a support check, not proof of reliable causal estimation.
            * ``value``: :math:`\widehat\tau_\ell`, irrespective of the action.
            * ``std_error``: the HC3 standard error above.
            * ``ci_lower``, ``ci_upper``: endpoints of the pointwise interval.

        Notes
        -----
        Unsupported and empty leaves have NaN effect estimates and intervals.
        Their clients remain in the overall policy evaluation.
        A positive ``value`` means estimated benefit from treatment even if
        ``action`` is 0. An interval entirely above/below zero supplies evidence
        of positive/negative group uplift; an interval crossing zero leaves its
        direction unresolved. These intervals describe group means, not effects
        for individual clients. Disagreement with the training action is evidence
        to review the policy, not an automatic update to it.
        """
        return deepcopy(self._rules_table)


@dataclass(frozen=True)
class _Node:
    rule_id: str | None = None
    action: int = 0
    feature: int | None = None
    threshold: float | None = None
    left: _Node | None = None
    right: _Node | None = None


def _ids(frame: pd.DataFrame, user_id: str) -> pd.Index:
    if not isinstance(user_id, str) or user_id not in frame.columns:
        raise ValueError("A user_id column is required.")
    values = pd.Index(frame[user_id], name=user_id)
    if values.hasnans or not values.is_unique:
        raise ValueError("Client IDs must be unique and nonmissing.")
    return values


def _validated_irm(irm: IRM) -> tuple[pd.DataFrame, pd.Index, tuple, dict]:
    if not isinstance(irm, IRM):
        raise TypeError("UpliftPolicyTree requires a fitted binary-treatment IRM.")
    check_is_fitted(irm, ["g0_hat_", "g1_hat_", "m_hat_"])
    if not hasattr(irm, "_fit_data_roles_"):
        raise ValueError(
            "Refit IRM to record the metadata required for policy learning."
        )
    if irm.weights is not None or irm._fit_weights_used_:
        raise ValueError("UpliftPolicyTree requires unweighted IRMs.")
    if irm.overlap_policy != "clip" or irm._fit_overlap_policy_ != "clip":
        raise ValueError("UpliftPolicyTree requires overlap_policy='clip'.")
    if irm.overlap_threshold != irm._fit_overlap_threshold_:
        raise ValueError("Overlap settings changed after fit; refit IRM.")
    data = irm.data
    roles = (
        data.outcome.name,
        data.treatment.name,
        tuple(data.confounders),
        data.user_id_name,
    )
    if roles != irm._fit_data_roles_:
        raise ValueError("Data role definitions changed after fit; refit IRM.")
    X, y, d, _ = irm._check_data()
    irm._validate_current_data_matches_fit(X=X, y=y, d=d)
    ids = _ids(data.df, data.user_id_name)
    if not ids.equals(irm._fit_index_):
        raise ValueError("Client IDs or their order changed after fit; refit IRM.")
    m = np.asarray(irm.m_hat_, dtype=float)
    if not np.all(np.isfinite(m) & (m > 0) & (m < 1)):
        raise ValueError(
            "Policy signals require finite propensity scores strictly between 0 and 1."
        )
    diagnostics = {
        "overlap_policy": irm._fit_overlap_policy_,
        "overlap_threshold": irm._fit_overlap_threshold_,
        "n_clipped": irm.overlap_n_clipped_,
        "clipped_fraction": irm.overlap_n_clipped_ / len(ids),
        "n_obs": len(ids),
    }
    return data.df, ids, roles, diagnostics


class UpliftPolicyTree(BaseEstimator):
    r"""Learn a shallow, greedy treatment policy from a fitted IRM.

    Let :math:`D\in\{0,1\}` denote historical treatment, :math:`Y` the observed
    outcome, :math:`X` all pre-treatment adjustment variables, and :math:`Z` the
    subset allowed in policy rules. Under consistency, unconfoundedness
    :math:`(Y(0),Y(1))\perp D\mid X`, and overlap
    :math:`0<e(X)=P(D=1\mid X)<1`, the CATE is

    .. math::

        \tau(x)=\mathbb{E}[Y(1)-Y(0)\mid X=x]
               =g_1(x)-g_0(x),
        \qquad g_d(x)=\mathbb{E}[Y\mid X=x,D=d].

    Using out-of-fold nuisance predictions, construct the doubly robust signal

    .. math::

        \Gamma_i = \widehat g_1^{(-k(i))}(X_i)
                   -\widehat g_0^{(-k(i))}(X_i)
          +\frac{D_i\{Y_i-\widehat g_1^{(-k(i))}(X_i)\}}
                 {\widetilde e_i}
          -\frac{(1-D_i)\{Y_i-\widehat g_0^{(-k(i))}(X_i)\}}
                 {1-\widetilde e_i},

    where :math:`k(i)` is the fold containing observation :math:`i` and
    :math:`\widetilde e_i=\min(1-\epsilon,
    \max(\epsilon,\widehat e^{(-k(i))}(X_i)))` is IRM's clipped propensity.
    Clipping stabilizes denominators but can bias the signal; double robustness
    does not remove hidden confounding or guarantee validity under clipping.

    The empirical policy objective, relative to treating nobody, is

    .. math::

        \widehat J(\pi)=\frac{1}{n_{\mathrm{train}}}
            \sum_{i\in\mathcal T}\pi(Z_i)\Gamma_i.

    The tree greedily improves this objective. It does not solve a global
    optimization over all possible trees; see :meth:`fit` for the split reward.

    Parameters
    ----------
    max_depth : int, default 2
        Maximum number of splits along a path; zero learns a constant policy.
    min_samples_leaf : int, default 50
        Minimum observations in each child of an accepted split.
    min_samples_per_arm : int, default 5
        Minimum treated and control observations in each child.

    Notes
    -----
    A leaf treats when its training mean signal is positive. Greedy search can
    miss interactions needing a zero-gain first split. If no split is feasible,
    a constant root policy is retained even when the sample is smaller than the
    child-size limits.

    All adjustment variables belong in IRM; ``policy_features`` only limits the
    rules. Features must be numeric, finite, and measured before intervention.
    Higher outcomes are better; costs, capacity constraints, weighted samples,
    and overlap-based sample dropping are not supported. Decisions estimate
    group benefit, not whether each individual will benefit.

    Split clients before fitting either IRM or tuning the policy. Evaluation
    checks disjoint IDs but cannot verify the provenance of nuisance learners
    or prevent reuse of evaluation results for tuning.
    """

    def __init__(
        self,
        *,
        max_depth: int = 2,
        min_samples_leaf: int = 50,
        min_samples_per_arm: int = 5,
    ):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_per_arm = min_samples_per_arm

    def fit(self, train_irm: IRM, *, policy_features: Sequence[str] | None = None):
        r"""Discover rules using only the training IRM's cross-fitted signals.

        For a node containing training indices :math:`A`, the optimal constant
        action and its unnormalized reward are

        .. math::

            a_A = \mathbf{1}\!\left\{\sum_{i\in A}\Gamma_i>0\right\},
            \qquad W(A)=\max\!\left(0,\sum_{i\in A}\Gamma_i\right).

        A candidate feature/threshold pair partitions :math:`A` into
        :math:`L=\{i\in A:Z_{ij}\le t\}` and
        :math:`R=\{i\in A:Z_{ij}>t\}`. Its gain is

        .. math::

            G(j,t;A)=W(L)+W(R)-W(A).

        Accept the feasible split with greatest positive gain, allowing for
        floating-point roundoff. Both children must satisfy the total and
        per-arm sample limits. Recurse until no improvement is available or
        ``max_depth`` is reached. Feature order and then ascending thresholds
        break ties. A zero leaf signal recommends no treatment.

        Parameters
        ----------
        train_irm : IRM
            Fitted, unweighted IRM with ``overlap_policy='clip'` and unchanged
            CausalData containing unique, nonmissing client IDs.
        policy_features : sequence of str, optional
            Ordered, nonempty subset of fitted confounders allowed in the rules.
            Defaults to all confounders. Other confounders still enter the
            nuisance models and the DR signal.

        Returns
        -------
        UpliftPolicyTree
            The fitted policy. No IRM models are fitted or modified.

        Notes
        -----
        Signals are computed from fitted nuisances, independently of the last
        requested IRM estimand (including ATTE) and any lazy CATE scoring models.
        """
        for name, minimum in (
            ("max_depth", 0),
            ("min_samples_leaf", 1),
            ("min_samples_per_arm", 1),
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, Integral)
                or value < minimum
            ):
                raise ValueError(f"{name} must be an integer >= {minimum}.")
        frame, ids, roles, diagnostics = _validated_irm(train_irm)
        if isinstance(policy_features, str):
            raise ValueError("policy_features must be a sequence of feature names.")
        features = list(roles[2] if policy_features is None else policy_features)
        if (
            not features
            or any(not isinstance(f, str) for f in features)
            or len(set(features)) != len(features)
            or not set(features).issubset(roles[2])
        ):
            raise ValueError(
                "policy_features must be a nonempty, unique subset of fitted confounders."
            )
        X = self._features(frame, features)
        phi, d, _ = _compute_dr_signal_from_irm(train_irm)
        rules: list[dict] = []

        def grow(rows: np.ndarray, depth: int, path: tuple) -> _Node:
            total = float(np.sum(phi[rows]))
            tolerance = 32 * np.finfo(float).eps * float(np.sum(np.abs(phi[rows])))
            best_reward = max(0.0, total)
            best = None
            if depth < self.max_depth:
                for feature in range(X.shape[1]):
                    ordered = rows[np.argsort(X[rows, feature], kind="stable")]
                    values = X[ordered, feature]
                    left_n = np.arange(1, len(rows))
                    left_t = np.cumsum(d[ordered])[:-1]
                    total_t = np.sum(d[ordered])
                    right_n = len(rows) - left_n
                    right_t = total_t - left_t
                    valid = (
                        (values[:-1] < values[1:])
                        & (left_n >= self.min_samples_leaf)
                        & (right_n >= self.min_samples_leaf)
                        & (left_t >= self.min_samples_per_arm)
                        & (left_n - left_t >= self.min_samples_per_arm)
                        & (right_t >= self.min_samples_per_arm)
                        & (right_n - right_t >= self.min_samples_per_arm)
                    )
                    candidates = np.flatnonzero(valid)
                    if not candidates.size:
                        continue
                    left_sum = np.cumsum(phi[ordered])[:-1]
                    rewards = np.maximum(0, left_sum) + np.maximum(0, total - left_sum)
                    cut = candidates[np.argmax(rewards[candidates])]
                    reward = float(rewards[cut])
                    # Ignore roundoff-only gains; keep the first feature/threshold on ties.
                    if reward > best_reward + tolerance:
                        lo, hi = float(values[cut]), float(values[cut + 1])
                        threshold = lo / 2.0 + hi / 2.0
                        # Adjacent floats may have no representable midpoint.
                        if not lo <= threshold < hi:
                            threshold = lo
                        best_reward = reward
                        best = (
                            feature,
                            threshold,
                            ordered[: cut + 1],
                            ordered[cut + 1 :],
                        )
            if best is not None:
                feature, threshold, left, right = best
                return _Node(
                    feature=feature,
                    threshold=threshold,
                    left=grow(
                        left, depth + 1, path + ((features[feature], "<=", threshold),)
                    ),
                    right=grow(
                        right, depth + 1, path + ((features[feature], ">", threshold),)
                    ),
                )
            rule_id = f"rule_{len(rules) + 1}"
            action = int(total > 0.0)
            rules.append(
                {
                    "rule_id": rule_id,
                    "conditions": " AND ".join(f"{f} {op} {v!r}" for f, op, v in path)
                    or "All clients",
                    "predicates": path,
                    "action": action,
                    "n_train": len(rows),
                    "n_treated_train": int(np.sum(d[rows])),
                    "n_control_train": int(len(rows) - np.sum(d[rows])),
                    "train_mean_uplift_descriptive": total / len(rows),
                }
            )
            return _Node(rule_id=rule_id, action=action)

        root = grow(np.arange(len(frame)), 0, ())
        # Commit fitted state only after successful construction.
        self.root_ = root
        self.policy_features_ = tuple(features)
        self._rules_table_ = pd.DataFrame(rules)
        self._training_ids_ = ids.copy()
        self._roles_ = roles
        self.training_diagnostics_ = diagnostics.copy()
        return self

    @staticmethod
    def _features(frame: pd.DataFrame, features: Sequence[str]) -> np.ndarray:
        if not isinstance(frame, pd.DataFrame) or not frame.columns.is_unique:
            raise ValueError("Clients must be a DataFrame with unique column names.")
        missing = [f for f in features if f not in frame.columns]
        if missing:
            raise ValueError(f"Missing policy features: {missing}.")
        if any(
            not pd.api.types.is_numeric_dtype(frame[f])
            or pd.api.types.is_complex_dtype(frame[f])
            for f in features
        ):
            raise ValueError("Policy features must be numeric.")
        if frame.loc[:, list(features)].isna().any().any():
            raise ValueError("Policy features must be finite and nonmissing.")
        X = frame.loc[:, list(features)].to_numpy(dtype=float)
        if not np.all(np.isfinite(X)):
            raise ValueError("Policy features must be finite and nonmissing.")
        return X

    def rules(self) -> pd.DataFrame:
        r"""Return the learned rules and descriptive training statistics.

        For training clients :math:`\mathcal T_\ell` in leaf :math:`\ell`,

        .. math::

            \overline\Gamma_{\ell,\mathrm{train}}
              =\frac{1}{|\mathcal T_\ell|}
                 \sum_{i\in\mathcal T_\ell}\Gamma_i,
            \qquad
            a_\ell=\mathbf{1}\{
                \overline\Gamma_{\ell,\mathrm{train}}>0\}.

        Returns
        -------
        pandas.DataFrame
            A copy containing ``rule_id``, readable ``conditions``, exact
            ``predicates`` tuples ``(feature, operator, threshold)``, ``action``
            (:math:`a_\ell`), ``n_train``, ``n_treated_train``,
            ``n_control_train``, and ``train_mean_uplift_descriptive``
            (:math:`\overline\Gamma_{\ell,\mathrm{train}}`). All predicates
            along a path must hold; the leaves form a disjoint partition.

        Notes
        -----
        Training means are selected on the same data used to discover the rules
        and carry no independent confidence intervals. Use :meth:`evaluate`
        and :meth:`UpliftPolicyEvaluation.rules_summary` for held-out evidence.
        """
        check_is_fitted(self, ["root_"])
        return deepcopy(self._rules_table_)

    def assign(self, clients: pd.DataFrame, *, user_id: str) -> pd.DataFrame:
        r"""Assign clients without outcomes or historical treatment labels.

        For disjoint learned regions :math:`R_\ell` and frozen leaf actions
        :math:`a_\ell`, the assignment function is

        .. math::

            \widehat\pi(z)=\sum_{\ell=1}^K
                a_\ell\mathbf{1}\{z\in R_\ell\}.

        Each client belongs to exactly one leaf. A threshold test sends
        :math:`z_j\le t` left and :math:`z_j>t` right, including equality on
        the left. Neither outcomes nor estimated individual effects enter this
        assignment step.

        Parameters
        ----------
        clients : pandas.DataFrame
            Client IDs and the fitted policy feature columns. Features must be
            numeric, finite, and nonmissing. Extra columns are ignored.
        user_id : str
            Column containing unique, nonmissing IDs. Cannot be ``rule_id``
            or ``action``, which are reserved output column names.

        Returns
        -------
        pandas.DataFrame
            Original ID column plus ``rule_id`` and integer ``action``:
            1 means target, 0 means do not target. Input order, ID dtype, and
            index are preserved. Empty inputs return the same output schema.
        """
        check_is_fitted(self, ["root_"])
        X = self._features(clients, self.policy_features_)
        _ids(clients, user_id)
        if user_id in {"rule_id", "action"}:
            raise ValueError(
                "user_id cannot use reserved output names 'rule_id' or 'action'."
            )
        actions = np.empty(len(X), dtype=int)
        leaves = np.empty(len(X), dtype=object)

        def visit(node: _Node, rows: np.ndarray):
            if node.rule_id is not None:
                actions[rows] = node.action
                leaves[rows] = node.rule_id
                return
            left = X[rows, node.feature] <= node.threshold
            visit(node.left, rows[left])
            visit(node.right, rows[~left])

        visit(self.root_, np.arange(len(X)))
        result = clients.loc[:, [user_id]].copy()
        result["rule_id"] = leaves
        result["action"] = actions
        return result

    def evaluate(self, eval_irm: IRM, *, alpha: float = 0.05) -> UpliftPolicyEvaluation:
        r"""Evaluate frozen decisions on an IRM fitted on disjoint clients.

        For an independent evaluation population and a fixed learned policy,
        the estimands are

        .. math::

            \Delta_{\pi,0} &= V(\widehat\pi)-V(0)
                =\mathbb{E}[\widehat\pi(Z)\tau(X)], \\
            \Delta_{\pi,1} &= V(\widehat\pi)-V(1)
                =\mathbb{E}[(\widehat\pi(Z)-1)\tau(X)], \\
            \Delta_{1,0} &= V(1)-V(0)=\mathbb{E}[\tau(X)].

        Estimate them with evaluation-only cross-fitted DR signals. See
        :meth:`UpliftPolicyEvaluation.summary` for paired-signal standard errors
        and :meth:`UpliftPolicyEvaluation.rules_summary` for leaf HC3 intervals.

        Parameters
        ----------
        eval_irm : IRM
            Independently fitted, unweighted IRM with ``overlap_policy='clip'``.
            Outcome, treatment, and ordered confounder definitions must match
            training. Its original CausalData and IDs must remain unchanged,
            and its IDs must be disjoint from training IDs.
        alpha : float, default 0.05
            Significance level in :math:`(0,1)` for nominal
            :math:`100(1-\alpha)\%` intervals.

        Returns
        -------
        UpliftPolicyEvaluation
            Overall policy gains, per-rule effects and support, and overlap
            diagnostics. The policy, training summaries, and both IRMs are
            left unchanged.

        Notes
        -----
        Unsupported leaves have unavailable subgroup estimates; their clients
        remain in the overall policy evaluation, whose inference relies on
        population overlap. An interval crossing zero does not establish a zero
        effect, and evaluation never changes a learned action. If these results
        are used to revise or select policies, evaluate the selected policy on
        another untouched sample.
        """
        check_is_fitted(self, ["root_"])
        if not np.isfinite(alpha) or not 0 < alpha < 1:
            raise ValueError("alpha must be in (0, 1).")
        frame, ids, roles, diagnostics = _validated_irm(eval_irm)
        if roles[:3] != self._roles_[:3]:
            raise ValueError(
                "Evaluation outcome, treatment, and confounder definitions must match training."
            )
        if len(ids.intersection(self._training_ids_)):
            raise ValueError("Training and evaluation client IDs must be disjoint.")
        assignments = self.assign(frame, user_id=roles[3])
        phi, d, _ = _compute_dr_signal_from_irm(eval_irm)
        actions = assignments["action"].to_numpy()
        z = float(norm.ppf(1 - alpha / 2))
        summary = []
        for name, signal in (
            ("policy_vs_none", actions * phi),
            ("policy_vs_all", (actions - 1) * phi),
            ("all_vs_none", phi),
        ):
            value = float(np.mean(signal))
            se = float(np.std(signal, ddof=1) / np.sqrt(len(signal)))
            summary.append(
                {
                    "comparison": name,
                    "value": value,
                    "std_error": se,
                    "ci_lower": value - z * se,
                    "ci_upper": value + z * se,
                    "n_obs": len(phi),
                    "treatment_fraction": float(np.mean(actions)),
                }
            )

        rule_report = (
            self.rules()
            .drop(
                columns=[
                    "n_train",
                    "n_treated_train",
                    "n_control_train",
                    "train_mean_uplift_descriptive",
                ]
            )
            .set_index("rule_id")
        )
        leaves = assignments["rule_id"].to_numpy()
        supported = {}
        for rule_id in rule_report.index:
            mask = leaves == rule_id
            n = int(np.sum(mask))
            nt = int(np.sum(d[mask]))
            status = (
                "empty"
                if n == 0
                else ("insufficient_arm_support" if nt == 0 or nt == n else "ok")
            )
            rule_report.loc[rule_id, ["n_eval", "n_treated", "n_control", "status"]] = [
                n,
                nt,
                n - nt,
                status,
            ]
            if status == "ok":
                supported[rule_id] = mask
        inference_columns = ["value", "std_error", "ci_lower", "ci_upper"]
        rule_report[inference_columns] = np.nan
        if supported:
            # HC3 is local to each group. Zero rows for unsupported groups do
            # not affect the supported groups' estimates or variances.
            names, values, ses, _, _, lower, upper, _ = (
                _estimate_gate_groupwise_inference(
                    phi=phi,
                    basis=pd.DataFrame(supported),
                    cov_type="HC3",
                    alpha=alpha,
                )
            )
            rule_report.loc[names, inference_columns] = np.column_stack(
                [values, ses, lower, upper]
            )
        for col in ("n_eval", "n_treated", "n_control"):
            rule_report[col] = rule_report[col].astype(int)
        return UpliftPolicyEvaluation(
            pd.DataFrame(summary),
            rule_report.reset_index(),
            {
                "training": deepcopy(self.training_diagnostics_),
                "evaluation": diagnostics,
                "inference": "approximate_normal",
                "rule_intervals": "pointwise_HC3",
            },
            float(alpha),
        )
