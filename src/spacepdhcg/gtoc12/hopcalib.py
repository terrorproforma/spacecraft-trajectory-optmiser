"""Calibrated low-thrust hop cost from the archive of SCvx-certified legs.

The beam and the collect DP price a hop as ``propellant = m (1 - exp(-f x ΔV_L / v_e))`` with
``ΔV_L`` the zero-revolution Lambert ΔV and ``f`` an inflation factor. ``screening`` fits ``f`` on
the authority ratio alone (``1.05 + 0.65 r``). This module fits ``f`` on the archived certified
hops as a linear model in the quantities the DP knows for every pair and epoch:

``f = c0 + c1 r + c2 (TOF / yr) + c3 |Δa| / 0.1 AU + c4 |Δλ| / π``

(``r`` = Lambert ΔV / full-thrust authority, ``Δa`` the semi-major-axis difference, ``Δλ`` the
mean-longitude difference at departure, wrapped to ``[0, π]``) and then shifts ``c0`` so that a
chosen quantile of the residual is non-positive (the forward mass check must close, so the model
errs on the heavy side). :func:`certified_hops` rebuilds the samples from ``route_summary.json``
archives (the Lambert ΔV is recomputed, the SCvx ΔV is stored); :func:`fit_inflation` returns
the coefficients with the in-sample and, when a holdout is given, out-of-sample residual
distribution; :class:`InflationFit` evaluates the model vectorised for the pair table.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from . import constants as C
from .clusters import mean_longitude
from .data import AsteroidCatalogue
from .ephemeris import asteroid_state
from .screening import exhaust_velocity_km_s, lambert_hops, thrust_authority_km_s

FloatArray = NDArray[np.float64]
EARTH_ID = 0
FEATURES = ("authority_ratio", "tof_years", "delta_a_per_0.1au", "delta_longitude_per_pi")


@dataclass(slots=True)
class HopSamples:
    """Certified asteroid-to-asteroid hops with the features the fit uses."""

    source: NDArray[np.int64]
    target: NDArray[np.int64]
    departure: FloatArray
    tof_days: FloatArray
    mass_kg: FloatArray
    lambert_dv: FloatArray
    scvx_dv: FloatArray
    delta_a_au: FloatArray
    delta_longitude_rad: FloatArray
    run: list[str]

    def __len__(self) -> int:
        return int(self.source.shape[0])

    @property
    def inflation(self) -> FloatArray:
        return self.scvx_dv / self.lambert_dv

    @property
    def authority_ratio(self) -> FloatArray:
        return self.lambert_dv / thrust_authority_km_s(self.mass_kg, self.tof_days, 1.0)

    def design(self) -> FloatArray:
        return design_matrix(
            self.authority_ratio, self.tof_days, self.delta_a_au, self.delta_longitude_rad
        )

    def subset(self, mask: NDArray[np.bool_]) -> HopSamples:
        idx = np.flatnonzero(mask)
        return HopSamples(
            self.source[idx],
            self.target[idx],
            self.departure[idx],
            self.tof_days[idx],
            self.mass_kg[idx],
            self.lambert_dv[idx],
            self.scvx_dv[idx],
            self.delta_a_au[idx],
            self.delta_longitude_rad[idx],
            [self.run[i] for i in idx],
        )


def design_matrix(
    authority_ratio: FloatArray,
    tof_days: FloatArray,
    delta_a_au: FloatArray,
    delta_longitude_rad: FloatArray,
) -> FloatArray:
    r = np.asarray(authority_ratio, dtype=np.float64)
    columns = np.broadcast_arrays(
        np.ones_like(r),
        r,
        np.asarray(tof_days, dtype=np.float64) / C.YEAR_DAYS,
        np.abs(np.asarray(delta_a_au, dtype=np.float64)) / 0.1,
        np.abs(np.asarray(delta_longitude_rad, dtype=np.float64)) / np.pi,
    )
    return np.stack(columns, axis=-1)  # (..., 5): works for 1-D samples and 2-D tables


def wrapped_longitude_difference(
    catalogue: AsteroidCatalogue,
    source: NDArray[np.int64],
    target: NDArray[np.int64],
    epoch: FloatArray,
) -> FloatArray:
    """Mean-longitude difference target - source at ``epoch`` wrapped to ``[-π, π]``."""

    src = np.searchsorted(catalogue.ids, source)
    tgt = np.searchsorted(catalogue.ids, target)
    epochs = np.asarray(epoch, dtype=np.float64)
    out = np.empty(epochs.shape[0])
    for i in range(epochs.shape[0]):
        ls = mean_longitude(catalogue, np.asarray([src[i]]), float(epochs[i]))[0]
        lt = mean_longitude(catalogue, np.asarray([tgt[i]]), float(epochs[i]))[0]
        out[i] = (lt - ls + np.pi) % (2.0 * np.pi) - np.pi
    return out


def certified_hops(
    catalogue: AsteroidCatalogue, sources: list[Path] | tuple[Path, ...]
) -> HopSamples:
    """Certified asteroid-to-asteroid legs of every ``route_summary.json`` under ``sources``."""

    rows: list[tuple[str, int, int, float, float, float, float]] = []
    for root in sources:
        for path in sorted(Path(root).rglob("route_summary.json")):
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            for leg in record.get("legs") or []:
                if not leg.get("certified") or leg.get("status") != "feasible":
                    continue
                a, b = int(leg["from"]), int(leg["to"])
                if a == EARTH_ID or b == EARTH_ID or a == b:
                    continue
                rows.append(
                    (
                        Path(root).name,
                        a,
                        b,
                        float(leg["t0"]),
                        float(leg["tf"]) - float(leg["t0"]),
                        float(leg["mass_before"]),
                        float(leg["delta_v_km_s"]),
                    )
                )
    if not rows:
        empty = np.zeros(0)
        return HopSamples(
            np.zeros(0, np.int64),
            np.zeros(0, np.int64),
            empty,
            empty,
            empty,
            empty,
            empty,
            empty,
            empty,
            [],
        )
    run = [r[0] for r in rows]
    source = np.array([r[1] for r in rows], dtype=np.int64)
    target = np.array([r[2] for r in rows], dtype=np.int64)
    departure = np.array([r[3] for r in rows])
    tof = np.array([r[4] for r in rows])
    mass = np.array([r[5] for r in rows])
    scvx = np.array([r[6] for r in rows])
    r_s, v_s = asteroid_state(catalogue, source, departure)
    r_t, v_t = asteroid_state(catalogue, target, departure + tof)
    hop = lambert_hops(r_s, v_s, r_t, v_t, departure, tof)
    lambert = np.where(hop.feasible, hop.total_delta_v, np.nan)
    src = np.searchsorted(catalogue.ids, source)
    tgt = np.searchsorted(catalogue.ids, target)
    delta_a = (catalogue.semi_major_axis_km[tgt] - catalogue.semi_major_axis_km[src]) / C.AU_KM
    delta_l = wrapped_longitude_difference(catalogue, source, target, departure)
    ok = np.isfinite(lambert) & (lambert > 1e-6) & np.isfinite(scvx)
    samples = HopSamples(source, target, departure, tof, mass, lambert, scvx, delta_a, delta_l, run)
    return samples.subset(ok)


@dataclass(slots=True)
class InflationFit:
    """``f = X c`` with the conservative shift applied; ``inflation()`` evaluates it."""

    coefficients: tuple[float, ...]
    quantile: float
    floor: float = 1.0
    residuals: dict[str, object] = field(default_factory=dict)

    def inflation(
        self,
        lambert_dv_km_s: FloatArray,
        mass_kg: FloatArray | float,
        tof_days: FloatArray,
        delta_a_au: FloatArray | float,
        delta_longitude_rad: FloatArray | float,
    ) -> FloatArray:
        dv = np.asarray(lambert_dv_km_s, dtype=np.float64)
        authority = thrust_authority_km_s(mass_kg, tof_days, 1.0)
        ratio = np.where(np.isfinite(dv), dv, 0.0) / np.maximum(authority, 1e-12)
        x = design_matrix(
            ratio,
            np.broadcast_to(np.asarray(tof_days, dtype=np.float64), dv.shape),
            np.broadcast_to(np.asarray(delta_a_au, dtype=np.float64), dv.shape),
            np.broadcast_to(np.asarray(delta_longitude_rad, dtype=np.float64), dv.shape),
        )
        return np.maximum(x @ np.asarray(self.coefficients), self.floor)

    def summary(self) -> dict[str, object]:
        return {
            "features": ["1", *FEATURES],
            "coefficients": [float(c) for c in self.coefficients],
            "quantile": self.quantile,
            "floor": self.floor,
            "residuals": self.residuals,
        }

    @classmethod
    def from_summary(cls, record: dict[str, object]) -> InflationFit:
        return cls(
            tuple(float(c) for c in record["coefficients"]),  # type: ignore[index]
            float(record.get("quantile", 0.65)),  # type: ignore[arg-type]
            float(record.get("floor", 1.0)),  # type: ignore[arg-type]
            dict(record.get("residuals") or {}),  # type: ignore[arg-type]
        )


def _residual_stats(residual: FloatArray) -> dict[str, float]:
    if residual.size == 0:
        return {"n": 0}
    q = np.percentile(residual, [5, 10, 25, 50, 75, 90, 95])
    return {
        "n": int(residual.size),
        "mean": float(residual.mean()),
        "rms": float(np.sqrt(np.mean(residual**2))),
        "p5": float(q[0]),
        "p10": float(q[1]),
        "p25": float(q[2]),
        "median": float(q[3]),
        "p75": float(q[4]),
        "p90": float(q[5]),
        "p95": float(q[6]),
        "share_under_priced": float(np.mean(residual > 0.0)),
    }


def fit_inflation(
    train: HopSamples,
    holdout: HopSamples | None = None,
    *,
    quantile: float = 0.65,
    max_inflation: float = 2.5,
) -> InflationFit:
    """Least-squares fit of the inflation model, shifted so ``quantile`` of the training
    residuals (measured - model) are non-positive. Residual statistics are stored for the
    training set, the holdout and, per feature bin, the ratio-only baseline of ``screening``.
    """

    keep = train.inflation <= max_inflation
    train = train.subset(keep)
    if len(train) < 8:
        raise ValueError("need at least eight certified hops to fit the inflation model")
    x = train.design()
    y = train.inflation
    coefficients, *_ = np.linalg.lstsq(x, y, rcond=None)
    residual = y - x @ coefficients
    shift = float(np.percentile(residual, 100.0 * quantile))
    coefficients = coefficients.copy()
    coefficients[0] += shift
    fit = InflationFit(tuple(float(c) for c in coefficients), quantile)
    baseline = 1.05 + 0.65 * train.authority_ratio
    stats: dict[str, object] = {
        "train": _residual_stats(y - x @ coefficients),
        "train_ratio_only_baseline": _residual_stats(y - baseline),
        "train_flat_1.2": _residual_stats(y - 1.2),
    }
    if holdout is not None and len(holdout) > 0:
        hx = holdout.design()
        hy = holdout.inflation
        stats["holdout"] = _residual_stats(hy - hx @ coefficients)
        stats["holdout_ratio_only_baseline"] = _residual_stats(
            hy - (1.05 + 0.65 * holdout.authority_ratio)
        )
        stats["holdout_flat_1.2"] = _residual_stats(hy - 1.2)
        # propellant error of the priced hop (kg) on the holdout: model vs certified
        v_e = exhaust_velocity_km_s()
        model_kg = holdout.mass_kg * (1.0 - np.exp(-(hx @ coefficients) * holdout.lambert_dv / v_e))
        real_kg = holdout.mass_kg * (1.0 - np.exp(-holdout.scvx_dv / v_e))
        stats["holdout_propellant_error_kg"] = _residual_stats(real_kg - model_kg)
    bins: dict[str, dict[str, float]] = {}
    ratio = train.authority_ratio
    for lo, hi in ((0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 1.0)):
        m = (ratio >= lo) & (ratio < hi)
        if np.any(m):
            bins[f"r_{lo:.1f}_{hi:.1f}"] = {
                "n": int(m.sum()),
                "measured_median": float(np.median(y[m])),
                "model_median": float(np.median((x @ coefficients)[m])),
            }
    stats["train_bins"] = bins
    fit.residuals = stats
    return fit


def fit_summary_path(root: Path) -> Path:
    return Path(root) / "results" / "gtoc12" / "hop_inflation_fit.json"


def load_fit(path: Path) -> InflationFit | None:
    try:
        return InflationFit.from_summary(json.loads(Path(path).read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None
