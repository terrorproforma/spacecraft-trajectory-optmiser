"""Co-moving asteroid clusters for cluster-first route generation.

Two asteroids can be linked by a cheap (50-100 kg, sub-revolution) hop only when they are on
nearly the same ellipse *and* nearly the same place on it: the archived solutions keep
|Δa| <= 0.04 AU, |Δe-vector| <= 0.06, relative inclination <= 4.5° and a phase difference of a
few degrees at departure (``references.py``).  Because such orbits have almost equal mean
motions, the phase difference drifts by only ~2° per year per 0.04 AU of Δa, so a group that is
co-located early in the mission stays a group: a *co-moving cluster* is a genuine, nearly static
object and a ship that lands in a dense one can chain 9-10 hops without long phasing waits.

:class:`ComovingClusters` bins the pool in the six-dimensional scaled space
``(a, e_x, e_y, i_x, i_y, λ)`` with a KD-tree, reports the co-moving density of every asteroid
(neighbours within ``radius`` bands) and the drift-aware phasing window of any pair, and labels
greedy density-ordered clusters.  The beam search uses the density as a cluster-first prior on
the Earth target and on expansions (``SearchSettings.cluster_*``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

from . import constants as C
from .data import AsteroidCatalogue

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class ClusterBands:
    a_au: float = 0.04
    e: float = 0.06
    i_deg: float = 4.5
    phase_deg: float = 8.0  # wider than the 3.3° hop band: the phase drifts over the mission
    reference_epoch: float = C.MISSION_START_MJD + 3.0 * C.YEAR_DAYS  # mid deploy phase
    radius: float = 1.5  # neighbourhood radius in band units (Euclidean in scaled space)


def mean_longitude(catalogue: AsteroidCatalogue, index: IntArray, epoch: float) -> FloatArray:
    """Mean longitude ``Ω + ω + M`` (rad, wrapped to [0, 2π)) at ``epoch``."""

    n = np.sqrt(C.MU_SUN_KM3_S2 / catalogue.semi_major_axis_km[index] ** 3) * C.DAY_S  # rad/day
    mean_anomaly = catalogue.mean_anomaly_rad[index] + n * (epoch - catalogue.epoch_mjd[index])
    return np.mod(
        catalogue.ascending_node_rad[index]
        + catalogue.argument_of_perihelion_rad[index]
        + mean_anomaly,
        2.0 * np.pi,
    )


def mean_motion_rad_per_day(catalogue: AsteroidCatalogue, index: IntArray) -> FloatArray:
    return np.sqrt(C.MU_SUN_KM3_S2 / catalogue.semi_major_axis_km[index] ** 3) * C.DAY_S


class ComovingClusters:
    def __init__(
        self,
        catalogue: AsteroidCatalogue,
        ids: IntArray,
        bands: ClusterBands | None = None,
    ) -> None:
        self.catalogue = catalogue
        self.bands = bands or ClusterBands()
        self.ids = np.asarray(sorted(int(item) for item in ids), dtype=np.int64)
        self.index = catalogue.index_of(self.ids)
        self.features = self._features()
        self.tree = cKDTree(self.features)
        self._position = {int(item): k for k, item in enumerate(self.ids)}
        self.density = self._density()
        self.labels = self._label()

    # -- features ------------------------------------------------------------------------

    def _features(self) -> FloatArray:
        b = self.bands
        cat = self.catalogue
        idx = self.index
        a = cat.semi_major_axis_km[idx] / C.AU_KM
        varpi = cat.ascending_node_rad[idx] + cat.argument_of_perihelion_rad[idx]
        e_vec = cat.eccentricity[idx][:, None] * np.stack([np.cos(varpi), np.sin(varpi)], axis=1)
        node = cat.ascending_node_rad[idx]
        i_vec = np.rad2deg(cat.inclination_rad[idx])[:, None] * np.stack(
            [np.cos(node), np.sin(node)], axis=1
        )
        lam = np.rad2deg(mean_longitude(cat, idx, b.reference_epoch))
        # the phase is periodic: embed it on a circle so that 359° and 1° are neighbours; the
        # chord 2 sin(Δ/2) ~ Δ for small differences, scaled so one band = one unit
        scale = 360.0 / (2.0 * np.pi * b.phase_deg)
        phase = scale * np.stack([np.cos(np.deg2rad(lam)), np.sin(np.deg2rad(lam))], axis=1)
        return np.column_stack([a / b.a_au, e_vec / b.e, i_vec / b.i_deg, phase]).astype(np.float64)

    def _density(self) -> IntArray:
        counts = self.tree.query_ball_point(self.features, r=self.bands.radius, return_length=True)
        return np.asarray(counts, dtype=np.int64) - 1  # exclude the asteroid itself

    def _label(self) -> IntArray:
        """Greedy clusters: densest unlabelled asteroid seeds a cluster of its unlabelled ball."""

        labels = np.full(self.ids.shape[0], -1, dtype=np.int64)
        order = np.lexsort((self.ids, -self.density))
        next_label = 0
        for seed in order:
            if labels[seed] >= 0:
                continue
            members = self.tree.query_ball_point(self.features[seed], r=self.bands.radius)
            members = [m for m in members if labels[m] < 0]
            labels[members] = next_label
            labels[seed] = next_label
            next_label += 1
        return labels

    # -- queries -------------------------------------------------------------------------

    def contains(self, asteroid_id: int) -> bool:
        return asteroid_id in self._position

    def density_of(self, asteroid_id: int) -> int:
        return int(self.density[self._position[asteroid_id]])

    def label_of(self, asteroid_id: int) -> int:
        return int(self.labels[self._position[asteroid_id]])

    def neighbours(self, asteroid_id: int) -> IntArray:
        """Co-moving neighbours (within ``radius`` bands), sorted by scaled distance then ID."""

        k = self._position[asteroid_id]
        members = self.tree.query_ball_point(self.features[k], r=self.bands.radius)
        members = np.asarray([m for m in members if m != k], dtype=np.int64)
        if members.shape[0] == 0:
            return members
        distance = np.linalg.norm(self.features[members] - self.features[k], axis=1)
        order = np.lexsort((self.ids[members], distance))
        return self.ids[members[order]]

    def unvisited_potential(self, asteroid_id: int, visited: set[int]) -> int:
        return int(sum(1 for item in self.neighbours(asteroid_id) if int(item) not in visited))

    def cluster_members(self, label: int) -> IntArray:
        return self.ids[self.labels == label]

    def phasing_window(
        self, source: int, target: int, epoch: float, band_deg: float, horizon_days: float
    ) -> tuple[float, float] | None:
        """Earliest ``[open, close]`` (MJD) after ``epoch`` when |phase(target) - phase(source)|
        is within ``band_deg``, from the linear drift of the mean longitudes; ``None`` if the
        window does not open within ``horizon_days``."""

        cat = self.catalogue
        idx = cat.index_of(np.asarray([source, target], dtype=np.int64))
        lam = mean_longitude(cat, idx, epoch)
        n = mean_motion_rad_per_day(cat, idx)
        delta = np.rad2deg(np.angle(np.exp(1j * (lam[1] - lam[0]))))  # (-180, 180]
        rate = np.rad2deg(n[1] - n[0])  # deg/day
        if abs(delta) <= band_deg:
            close = np.inf if abs(rate) < 1e-12 else (np.sign(rate) * band_deg - delta) / rate
            return epoch, epoch + min(close, horizon_days)
        if abs(rate) < 1e-12:
            return None
        # phase moves at ``rate``: time until it reaches the near edge of the band
        target_delta = -band_deg if rate > 0 else band_deg
        wait = (target_delta - delta) / rate
        if wait < 0.0:
            wait += 360.0 / abs(rate)  # wrap around the full circle
        if wait > horizon_days:
            return None
        duration = 2.0 * band_deg / abs(rate)
        return epoch + wait, epoch + min(wait + duration, horizon_days)

    def summary(self) -> dict[str, object]:
        sizes = np.bincount(self.labels[self.labels >= 0])
        return {
            "asteroids": int(self.ids.shape[0]),
            "clusters": int(sizes.shape[0]),
            "density_p50": float(np.median(self.density)),
            "density_p90": float(np.percentile(self.density, 90)),
            "density_max": int(self.density.max()) if self.density.size else 0,
            "largest_cluster": int(sizes.max()) if sizes.size else 0,
            "clusters_ge_10": int((sizes >= 10).sum()),
            "bands": {
                "a_au": self.bands.a_au,
                "e": self.bands.e,
                "i_deg": self.bands.i_deg,
                "phase_deg": self.bands.phase_deg,
                "radius": self.bands.radius,
                "reference_epoch": self.bands.reference_epoch,
            },
        }
