from __future__ import annotations

import numpy as np
from typing import Dict, Any, Optional, Tuple, Union
import pandas as pd
from scipy.stats import norm

_ESSENTIALLY_ZERO = 1e-32


def _compute_sensitivity_bias_unified(
    sigma2: Union[float, np.ndarray],
    nu2: Union[float, np.ndarray],
    psi_sigma2: np.ndarray,   # (n,)
    psi_nu2: np.ndarray,      # (n, K-1) или (n,)
) -> Tuple[np.ndarray, np.ndarray]:
    """
    max_bias_k = sqrt(max(sigma2 * nu2_k, 0))

    Возвращает:
      - max_bias: (K-1,) (или (1,) если nu2 скаляр)
      - psi_max_bias: (n, K-1)
    """
    sigma2_f = float(np.asarray(sigma2).reshape(()))
    if sigma2_f <= 0.0:
        # shape согласуем по nu2
        nu2_arr = np.atleast_1d(np.asarray(nu2, dtype=float))
        return np.zeros_like(nu2_arr), np.zeros((psi_sigma2.shape[0], nu2_arr.shape[0]), dtype=float)

    nu2_arr = np.atleast_1d(np.asarray(nu2, dtype=float))  # (K-1,) или (1,)
    Km1 = nu2_arr.shape[0]

    psi_sigma2 = np.asarray(psi_sigma2, dtype=float).reshape(-1)  # (n,)
    n = psi_sigma2.shape[0]
    psi_sigma2 = psi_sigma2 - psi_sigma2.mean()

    psi_nu2 = np.asarray(psi_nu2, dtype=float)
    if psi_nu2.ndim == 1:
        # допустим случай (n,) => (n,1)
        psi_nu2 = psi_nu2.reshape(-1, 1)
    if psi_nu2.shape[0] != n:
        raise ValueError("psi_nu2 must have same number of rows as psi_sigma2.")
    if psi_nu2.shape[1] != Km1:
        raise ValueError(f"psi_nu2 must have shape (n, {Km1}).")

    # центрируем по каждому контрасту
    psi_nu2 = psi_nu2 - psi_nu2.mean(axis=0, keepdims=True)

    prod = sigma2_f * nu2_arr
    prod = np.maximum(prod, 0.0)
    max_bias = np.sqrt(prod)  # (K-1,)

    psi_max_bias = np.zeros((n, Km1), dtype=float)
    for j in range(Km1):
        if nu2_arr[j] <= 0.0 or max_bias[j] <= _ESSENTIALLY_ZERO:
            psi_max_bias[:, j] = 0.0
        else:
            denom = 2.0 * max_bias[j]
            # delta-method: d sqrt(sigma2*nu2) = (sigma2 dnu2 + nu2 dsigma2) / (2 sqrt(sigma2*nu2))
            psi_max_bias[:, j] = (sigma2_f * psi_nu2[:, j] + nu2_arr[j] * psi_sigma2) / denom

    return max_bias, psi_max_bias


# Backward-compatible alias (как у тебя задумано)
def compute_sensitivity_bias(
    sigma2: Union[float, np.ndarray],
    nu2: Union[float, np.ndarray],
    psi_sigma2: np.ndarray,
    psi_nu2: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    return _compute_sensitivity_bias_unified(sigma2, nu2, psi_sigma2, psi_nu2)


# еще один legacy-алиас (если где-то зовёшь так)
def compute_sensitivity_bias_local(
    sigma2: Union[float, np.ndarray],
    nu2: Union[float, np.ndarray],
    psi_sigma2: np.ndarray,
    psi_nu2: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    return _compute_sensitivity_bias_unified(sigma2, nu2, psi_sigma2, psi_nu2)


def combine_nu2(
    m_alpha: np.ndarray,  # (n, K-1)
    rr: np.ndarray,       # (n, K-1)
    cf_y: float,
    r2_d: float,
    rho: float,
    use_signed_rr: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Для каждого контраста k:
      base_{i,k} = (a_{i,k}^2)*cf_y + (b_{i,k}^2)*cf_d + 2*rho*sqrt(cf_y*cf_d)*a_{i,k}*b_{i,k}
      a = sqrt(2*m_alpha), b = rr (signed) или abs(rr) (worst-case)
    Возвращает:
      nu2: (K-1,)
      psi_nu2: (n, K-1) (центрированная по столбцам)
    """
    cf_y = float(cf_y)
    r2_d = float(r2_d)
    rho = float(np.clip(rho, -1.0, 1.0))

    if cf_y < 0 or r2_d < 0:
        raise ValueError("cf_y and r2_d must be >= 0.")
    if r2_d >= 1.0:
        raise ValueError("r2_d must be < 1.0.")

    m_alpha = np.asarray(m_alpha, dtype=float)
    rr = np.asarray(rr, dtype=float)
    if m_alpha.shape != rr.shape:
        raise ValueError("m_alpha and rr must have the same shape (n, K-1).")
    if m_alpha.ndim != 2:
        raise ValueError("m_alpha and rr must be 2D arrays of shape (n, K-1).")

    cf_d = r2_d / (1.0 - r2_d)

    a = np.sqrt(2.0 * np.maximum(m_alpha, 0.0))  # (n, K-1)
    b = rr if use_signed_rr else np.abs(rr)

    base = (a * a) * cf_y + (b * b) * cf_d + 2.0 * rho * np.sqrt(cf_y * cf_d) * a * b
    base = np.maximum(base, 0.0)  # численная стабилизация

    nu2 = base.mean(axis=0)  # (K-1,)
    psi_nu2 = base - nu2[None, :]  # (n, K-1)

    return nu2, psi_nu2


# если тебе нужна функция с прежней сигнатурой _combine_nu2_local:
def _combine_nu2_local(
    m_alpha: np.ndarray,
    rr: np.ndarray,
    cf_y: float,
    r2_d: float,
    rho: float,
    _=None,
    use_signed_rr: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    return combine_nu2(m_alpha, rr, cf_y, r2_d, rho, use_signed_rr=use_signed_rr)


def pulltheta_se_ci(effect_estimation: Any, alpha: float) -> Tuple[Union[float, np.ndarray],
                                                                   Union[float, np.ndarray],
                                                                   Union[Tuple[float, float], np.ndarray]]:
    """
    Возвращает:
      theta: float или (K-1,)
      se: float или (K-1,)
      ci: (2,) или (K-1, 2)
    """
    z = float(norm.ppf(1 - alpha / 2.0))

    # helper: normalize to scalar-or-1d
    def _as_1d(x):
        arr = np.asarray(x, dtype=float)
        if arr.ndim == 0:
            return float(arr)
        return arr.reshape(-1)

    # 1) CausalEstimate-подобный объект
    if hasattr(effect_estimation, "value"):
        theta = _as_1d(getattr(effect_estimation, "value"))

        ci_low = getattr(effect_estimation, "ci_lower_absolute", None)
        ci_high = getattr(effect_estimation, "ci_upper_absolute", None)

        # SE: из model_options или восстановим из CI
        opts = getattr(effect_estimation, "model_options", {}) or {}
        se = opts.get("std_error", None)
        if se is not None:
            se = _as_1d(se)
        elif ci_low is not None and ci_high is not None:
            ci_low = _as_1d(ci_low); ci_high = _as_1d(ci_high)
            se = (np.asarray(ci_high, float) - np.asarray(theta, float)) / z
        else:
            se = 0.0

        if ci_low is not None and ci_high is not None:
            ci = np.column_stack([np.asarray(ci_low, float), np.asarray(ci_high, float)])
            # если скаляр — вернем tuple
            if np.asarray(theta).ndim == 0:
                return float(theta), float(se), (float(ci[0, 0]), float(ci[0, 1]))
            return np.asarray(theta, float), np.asarray(se, float), ci

        # fallback CI
        if np.asarray(theta).ndim == 0:
            return float(theta), float(se), (float(theta - z*se), float(theta + z*se))
        theta_arr = np.asarray(theta, float); se_arr = np.asarray(se, float)
        ci = np.column_stack([theta_arr - z*se_arr, theta_arr + z*se_arr])
        return theta_arr, se_arr, ci

    # 2) dict (legacy)
    if isinstance(effect_estimation, dict):
        theta = effect_estimation.get("coefficient", None)
        se = effect_estimation.get("std_error", None)
        ci = effect_estimation.get("confidence_interval", None)

        model = effect_estimation.get("model", None)

        if theta is None and model is not None and hasattr(model, "coef_"):
            theta = model.coef_
        if se is None and model is not None and hasattr(model, "se_"):
            se = model.se_

        theta = _as_1d(theta if theta is not None else 0.0)
        se = _as_1d(se if se is not None else 0.0)

        if ci is not None:
            ci_arr = np.asarray(ci, dtype=float)
            # допускаем (2,) или (K-1,2)
            if ci_arr.ndim == 1 and ci_arr.shape[0] == 2 and np.asarray(theta).ndim == 0:
                return float(theta), float(se), (float(ci_arr[0]), float(ci_arr[1]))
            return np.asarray(theta, float), np.asarray(se, float), ci_arr

        # fallback CI
        if np.asarray(theta).ndim == 0:
            return float(theta), float(se), (float(theta - z*se), float(theta + z*se))
        theta_arr = np.asarray(theta, float); se_arr = np.asarray(se, float)
        ci_arr = np.column_stack([theta_arr - z*se_arr, theta_arr + z*se_arr])
        return theta_arr, se_arr, ci_arr

    # 3) model instance with coef_ and se_
    if hasattr(effect_estimation, "coef_") and hasattr(effect_estimation, "se_"):
        theta = _as_1d(effect_estimation.coef_)
        se = _as_1d(effect_estimation.se_)

        if np.asarray(theta).ndim == 0:
            return float(theta), float(se), (float(theta - z*se), float(theta + z*se))

        theta_arr = np.asarray(theta, float); se_arr = np.asarray(se, float)
        ci = np.column_stack([theta_arr - z*se_arr, theta_arr + z*se_arr])
        return theta_arr, se_arr, ci

    return 0.0, 0.0, (0.0, 0.0)


def _to_1d(x) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 0:
        return arr.reshape(1)
    return arr.reshape(-1)


def _ci_to_2d(ci, J: int) -> np.ndarray:
    ci_arr = np.asarray(ci, dtype=float)
    if ci_arr.ndim == 1:
        if ci_arr.shape[0] != 2:
            raise ValueError("sampling_ci as 1d must have length 2.")
        return np.tile(ci_arr.reshape(1, 2), (J, 1))
    if ci_arr.ndim == 2 and ci_arr.shape[1] == 2:
        if ci_arr.shape[0] != J:
            raise ValueError(f"sampling_ci must have {J} rows.")
        return ci_arr
    raise ValueError("sampling_ci must be shape (2,) or (J,2).")


def _maybe_squeeze_scalar(x, J: int):
    # для совместимости: если J==1 возвращаем float/tuple, иначе np.ndarray
    if J == 1:
        if isinstance(x, tuple) and len(x) == 2:
            return (float(x[0]), float(x[1]))
        arr = np.asarray(x)
        if arr.ndim == 0:
            return float(arr)
        if arr.size == 1:
            return float(arr.reshape(-1)[0])
    return x


def compute_bias_aware_ci(
    effect_estimation: Dict[str, Any] | Any,
    _=None,
    cf_y: float = 0.0,
    r2_d: float = 0.0,
    rho: float = 1.0,
    H0: float = 0.0,
    alpha: float = 0.05,
    use_signed_rr: bool = False,  # оставляю в сигнатуре; здесь не используется (у тебя так было)
) -> Dict[str, Any]:
    """
    Multi-treatment (pairwise 0 vs k) bias-aware CI.

    Returns dict with arrays of length J=K-1:
      - theta, se : (J,)
      - sampling_ci, theta_bounds_cofounding, bias_aware_ci : (J,2)
      - max_bias, nu2, rv, rva : (J,)
      - sigma2 : scalar
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if cf_y < 0 or r2_d < 0:
        raise ValueError("cf_y and r2_d must be >= 0")
    if r2_d >= 1.0:
        raise ValueError("r2_d must be < 1.0")

    # --- extract theta/se/ci (у тебя psi: (n, K-1) => theta/se тоже ожидаем (K-1,)) ---
    # ВАЖНО: используй функцию pulltheta_se_ci из предыдущего блока, которую мы исправляли.
    theta, se, sampling_ci = pulltheta_se_ci(effect_estimation, alpha=alpha)
    theta_arr = _to_1d(theta)
    se_arr = _to_1d(se)
    J = theta_arr.shape[0]
    if se_arr.shape[0] not in (1, J):
        raise ValueError("se must be scalar or length J.")
    if se_arr.shape[0] == 1 and J > 1:
        se_arr = np.repeat(se_arr, J)

    z = float(norm.ppf(1 - alpha / 2.0))
    sampling_ci_2d = _ci_to_2d(sampling_ci, J)

    # --- get diagnostic elements (предпочтительно из diag) ---
    model = None
    diag = getattr(effect_estimation, "diagnostic_data", None)

    if isinstance(effect_estimation, dict):
        model = effect_estimation.get("model", None)
        if diag is None:
            diag = effect_estimation.get("diagnostic_data", None)
    elif hasattr(effect_estimation, "coef_"):
        model = effect_estimation

    elems = None
    if diag is not None and getattr(diag, "sigma2", None) is not None:
        elems = {
            "sigma2": diag.sigma2,
            "nu2": diag.nu2,
            "psi_sigma2": diag.psi_sigma2,
            "psi_nu2": diag.psi_nu2,
            "psi": getattr(diag, "psi", None),
        }
    elif model is not None and hasattr(model, "_sensitivity_element_est"):
        elems = model._sensitivity_element_est()

    # --- defaults if no sensitivity elements ---
    sigma2 = np.nan
    nu2 = np.full(J, np.nan, dtype=float)
    max_bias = np.zeros(J, dtype=float)

    if elems is not None:
        sigma2 = float(np.asarray(elems.get("sigma2", np.nan)).reshape(()))
        nu2_raw = np.asarray(elems.get("nu2", np.full(J, np.nan)), dtype=float)
        nu2 = _to_1d(nu2_raw)
        if nu2.shape[0] != J:
            raise ValueError(f"nu2 must have length J={J}, got {nu2.shape[0]}.")

        bias_factor = float(np.sqrt(cf_y * r2_d / (1.0 - r2_d))) if (cf_y > 0 and r2_d > 0) else 0.0
        core = np.sqrt(np.maximum(sigma2 * nu2, 0.0))  # (J,)
        max_bias = core * bias_factor  # (J,)

    rho_clip = float(np.clip(rho, -1.0, 1.0))
    bound_width = np.abs(rho_clip) * max_bias  # (J,)
    theta_lower = theta_arr - bound_width
    theta_upper = theta_arr + bound_width
    theta_bounds = np.column_stack([theta_lower, theta_upper])  # (J,2)

    # --- robustness values (по твоей формуле, векторно) ---
    denom_rv = np.abs(rho_clip) * np.sqrt(np.maximum(sigma2 * nu2, 0.0))  # (J,)
    delta_theta = np.abs(theta_arr - float(H0))

    rv = np.full(J, np.nan, dtype=float)
    ok = (denom_rv > 1e-16) & (delta_theta > 0)
    D = np.zeros(J, dtype=float)
    D[ok] = delta_theta[ok] / denom_rv[ok]
    f2 = D**2
    rv[ok] = (np.sqrt(f2[ok]**2 + 4.0 * f2[ok]) - f2[ok]) / 2.0
    rv[~ok] = np.where(delta_theta[~ok] == 0, 0.0, np.nan)

    delta_theta_a = np.maximum(delta_theta - z * se_arr, 0.0)
    rva = np.zeros(J, dtype=float)
    ok_a = (denom_rv > 1e-16) & (delta_theta_a > 0)
    Da = np.zeros(J, dtype=float)
    Da[ok_a] = delta_theta_a[ok_a] / denom_rv[ok_a]
    f2a = Da**2
    rva[ok_a] = (np.sqrt(f2a[ok_a]**2 + 4.0 * f2a[ok_a]) - f2a[ok_a]) / 2.0

    # --- bias-aware CI (faithful if psi available; else approx) ---
    bias_aware_ci = np.column_stack([theta_lower - z * se_arr, theta_upper + z * se_arr])  # fallback

    if elems is not None and all(k in elems for k in ("psi", "psi_sigma2", "psi_nu2")) and elems["psi"] is not None:
        psi = np.asarray(elems["psi"], dtype=float)            # (n,J)
        psi_sigma2 = np.asarray(elems["psi_sigma2"], dtype=float).reshape(-1)  # (n,)
        psi_nu2 = np.asarray(elems["psi_nu2"], dtype=float)    # (n,J)

        if psi.ndim == 1 and J == 1:
            psi = psi.reshape(-1, 1)
        if psi.shape[1] != J:
            raise ValueError(f"psi must have shape (n, {J}).")
        n = psi.shape[0]

        # correction_{i,k}
        bias_factor = float(np.sqrt(cf_y * r2_d / (1.0 - r2_d))) if (cf_y > 0 and r2_d > 0) else 0.0
        denom = 2.0 * np.sqrt(np.maximum(sigma2 * nu2, 0.0))  # (J,)

        correction = np.zeros_like(psi, dtype=float)
        good = denom > _ESSENTIALLY_ZERO
        if np.any(good) and bias_factor > 0:
            # (n,J): sigma2*psi_nu2 + nu2*psi_sigma2
            numer = sigma2 * psi_nu2 + (nu2[None, :] * psi_sigma2[:, None])
            correction[:, good] = (np.abs(rho_clip) * bias_factor) * (numer[:, good] / denom[None, good])

        psi_plus = psi + correction
        psi_minus = psi - correction

        # se for bounds
        se_lower = np.sqrt(np.var(psi_minus, axis=0, ddof=1) / n)  # (J,)
        se_upper = np.sqrt(np.var(psi_plus, axis=0, ddof=1) / n)   # (J,)

        bias_aware_ci = np.column_stack([theta_lower - z * se_lower, theta_upper + z * se_upper])

    out = dict(
        theta=theta_arr,
        se=se_arr,
        alpha=float(alpha),
        z=float(z),
        H0=float(H0),
        sampling_ci=sampling_ci_2d,
        theta_bounds_cofounding=theta_bounds,
        bias_aware_ci=bias_aware_ci,
        max_bias=max_bias,
        sigma2=float(sigma2),
        nu2=nu2,
        rv=rv,
        rva=rva,
        params=dict(
            cf_y=float(cf_y),
            r2_d=float(r2_d),
            rho=float(rho_clip),
            use_signed_rr=bool(use_signed_rr),
        ),
    )

    # Для совместимости со старым scalar-API: если J==1 -> сжимаем
    if J == 1:
        out["theta"] = float(theta_arr[0])
        out["se"] = float(se_arr[0])
        out["sampling_ci"] = (float(sampling_ci_2d[0, 0]), float(sampling_ci_2d[0, 1]))
        out["theta_bounds_cofounding"] = (float(theta_bounds[0, 0]), float(theta_bounds[0, 1]))
        out["bias_aware_ci"] = (float(bias_aware_ci[0, 0]), float(bias_aware_ci[0, 1]))
        out["max_bias"] = float(max_bias[0])
        out["nu2"] = float(nu2[0])
        out["rv"] = float(rv[0]) if np.isfinite(rv[0]) else rv[0]
        out["rva"] = float(rva[0])
    return out


def format_bias_aware_summary(res: Dict[str, Any], label: str | None = None) -> str:
    lbl = label or "theta"

    # scalar case (backward-compatible)
    if np.asarray(res["theta"]).ndim == 0:
        ci_l, ci_u = res["sampling_ci"]
        th_l, th_u = res["theta_bounds_cofounding"]
        bci_l, bci_u = res["bias_aware_ci"]
        theta = float(res["theta"]); se = float(res["se"])
        alpha = res["alpha"]; z = res["z"]
        cf = res["params"]

        lines = []
        lines.append("================== Bias-aware Interval ==================")
        lines.append("")
        lines.append("------------------ Scenario          ------------------")
        lines.append(f"Significance Level: alpha={alpha}")
        lines.append(f"Null Hypothesis: H0={res.get('H0', 0.0)}")
        lines.append(f"Sensitivity parameters: cf_y={cf['cf_y']}; r2_d={cf['r2_d']}, rho={cf['rho']}, use_signed_rr={cf['use_signed_rr']}")
        lines.append("")
        lines.append("------------------ Components        ------------------")
        lines.append(f"{'':>12} {'theta':>11} {'se':>11} {'z':>8} {'max_bias':>12} {'sigma2':>12} {'nu2':>12}")
        lines.append(f"{lbl:>12} {theta:11.6f} {se:11.6f} {z:8.4f} {res['max_bias']:12.6f} {res['sigma2']:12.6f} {res['nu2']:12.6f}")
        lines.append("")
        lines.append("------------------ Intervals         ------------------")
        lines.append(f"{'':>12} {'Sampling CI l':>14} {'Conf. θ l':>12} {'Bias-aware l':>14} {'Bias-aware u':>14} {'Conf. θ u':>12} {'Sampling CI u':>14}")
        lines.append(f"{lbl:>12} {ci_l:14.6f} {th_l:12.6f} {bci_l:14.6f} {bci_u:14.6f} {th_u:12.6f} {ci_u:14.6f}")

        if "rv" in res and "rva" in res:
            lines.append("")
            lines.append("------------------ Robustness Values ------------------")
            lines.append(f"{'':>12} {'RV (%)':>15} {'RVa (%)':>15}")
            lines.append(f"{lbl:>12} {res['rv']*100:15.6f} {res['rva']*100:15.6f}")

        return "\n".join(lines)

    # vector case: J=K-1
    theta = np.asarray(res["theta"], float).reshape(-1)
    se = np.asarray(res["se"], float).reshape(-1)
    max_bias = np.asarray(res["max_bias"], float).reshape(-1)
    nu2 = np.asarray(res["nu2"], float).reshape(-1)
    J = theta.shape[0]

    sampling_ci = np.asarray(res["sampling_ci"], float).reshape(J, 2)
    theta_bounds = np.asarray(res["theta_bounds_cofounding"], float).reshape(J, 2)
    bias_aware_ci = np.asarray(res["bias_aware_ci"], float).reshape(J, 2)

    idx = [f"0 vs {k}" for k in range(1, J + 1)]
    df = pd.DataFrame({
        "theta": theta,
        "se": se,
        "max_bias": max_bias,
        "sigma2": float(res["sigma2"]),
        "nu2": nu2,
        "sampling_ci_l": sampling_ci[:, 0],
        "sampling_ci_u": sampling_ci[:, 1],
        "theta_l": theta_bounds[:, 0],
        "theta_u": theta_bounds[:, 1],
        "bias_aware_ci_l": bias_aware_ci[:, 0],
        "bias_aware_ci_u": bias_aware_ci[:, 1],
    }, index=idx)

    cf = res["params"]
    lines = []
    lines.append("================== Bias-aware Interval ==================")
    lines.append("")
    lines.append("------------------ Scenario          ------------------")
    lines.append(f"Label: {lbl}")
    lines.append(f"Significance Level: alpha={res['alpha']}")
    lines.append(f"Null Hypothesis: H0={res.get('H0', 0.0)}")
    lines.append(f"Sensitivity parameters: cf_y={cf['cf_y']}; r2_d={cf['r2_d']}, rho={cf['rho']}, use_signed_rr={cf['use_signed_rr']}")
    lines.append("")
    lines.append(df.to_string(float_format=lambda x: f"{x: .6f}"))
    return "\n".join(lines)


def get_sensitivity_summary(
    effect_estimation: Dict[str, Any] | Any,
    _=None,
    label: Optional[str] = None,
) -> Optional[str]:

    if isinstance(effect_estimation, dict):
        if "model" not in effect_estimation and "bias_aware" not in effect_estimation:
            return None
        effect_dict = effect_estimation
    elif hasattr(effect_estimation, "coef_") and hasattr(effect_estimation, "se_"):
        effect_dict = {"model": effect_estimation}
    elif hasattr(effect_estimation, "value") and hasattr(effect_estimation, "diagnostic_data"):
        effect_dict = {
            "model": None,
            "diagnostic_data": effect_estimation.diagnostic_data,
            "bias_aware": getattr(effect_estimation, "sensitivity_analysis", {}),
        }
    else:
        return None

    model = effect_dict.get("model", None)

    if label is None:
        label = "theta"
        if hasattr(effect_estimation, "treatment") and isinstance(effect_estimation.treatment, str):
            label = effect_estimation.treatment
        elif model is not None:
            data_obj = getattr(model, "data", getattr(model, "data_contracts", None))
            t = getattr(data_obj, "treatment", None)
            label = getattr(t, "name", None) or "theta"

    res = effect_dict.get("bias_aware", None)

    # fallback: sampling-only
    if not isinstance(res, dict) or not res:
        theta, se, ci = pulltheta_se_ci(effect_estimation, alpha=0.05)
        theta_arr = _to_1d(theta)
        se_arr = _to_1d(se)
        J = theta_arr.shape[0]
        z = float(norm.ppf(1 - 0.05 / 2.0))
        ci2d = _ci_to_2d(ci, J)
        bias_aware_ci = np.column_stack([theta_arr - z * se_arr, theta_arr + z * se_arr])

        res = dict(
            theta=theta_arr if J > 1 else float(theta_arr[0]),
            se=se_arr if J > 1 else float(se_arr[0]),
            alpha=0.05,
            z=z,
            sampling_ci=ci2d if J > 1 else (float(ci2d[0, 0]), float(ci2d[0, 1])),
            theta_bounds_cofounding=np.column_stack([theta_arr, theta_arr]) if J > 1 else (float(theta_arr[0]), float(theta_arr[0])),
            bias_aware_ci=bias_aware_ci if J > 1 else (float(bias_aware_ci[0, 0]), float(bias_aware_ci[0, 1])),
            max_bias=np.zeros(J) if J > 1 else 0.0,
            sigma2=np.nan,
            nu2=np.full(J, np.nan) if J > 1 else np.nan,
            params=dict(cf_y=0.0, r2_d=0.0, rho=0.0, use_signed_rr=False),
        )

    return format_bias_aware_summary(res, label=label)


def sensitivity_analysis(
    effect_estimation: Dict[str, Any] | Any,
    _=None,
    cf_y: float = 0.0,
    r2_d: float = 0.0,
    rho: float = 1.0,
    H0: float = 0.0,
    alpha: float = 0.05,
    use_signed_rr: bool = False,
) -> Dict[str, Any]:

    res = compute_bias_aware_ci(
        effect_estimation,
        cf_y=cf_y,
        r2_d=r2_d,
        rho=rho,
        H0=H0,
        alpha=alpha,
        use_signed_rr=use_signed_rr,
    )

    if isinstance(effect_estimation, dict):
        effect_estimation["bias_aware"] = res

    return res