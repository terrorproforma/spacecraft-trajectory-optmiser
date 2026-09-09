import { countAtOrBefore } from "./kepler.js";

/** Segment i joins saved samples i and i+1. Missing intervals are never drawn. */
export function replayGaps(series) {
  const gaps = series.gap_after_indices ?? [];
  if (!Array.isArray(gaps) || gaps.some((i, n) => !Number.isInteger(i) || i < 0 ||
      i >= series.points_txyz.length - 1 || (n > 0 && i <= gaps[n - 1]))) {
    throw new Error("Invalid saved trajectory gap indices");
  }
  return new Set(gaps);
}

export function visibleSegmentRanges(pointCount, visible, gaps = new Set()) {
  const end = Math.max(0, Math.min(pointCount, visible) - 1), ranges = [];
  let start = 0;
  for (const gap of gaps) {
    if (gap >= end) break;
    if (gap > start) ranges.push([start, gap - start]);
    start = gap + 1;
  }
  if (start < end) ranges.push([start, end - start]);
  return ranges;
}

/** A saved endpoint is visible; a missing interval has no current-position marker. */
export function currentReplaySample(ship, epoch) {
  const index = countAtOrBefore(ship.times, epoch) - 1;
  if (index < 0) return -1;
  if (ship.gapAfter?.has(index) && epoch > ship.times[index] && epoch < ship.times[index + 1]) return -1;
  return index;
}

export function sampleSourceLabel(ship) {
  return ship.display_sample_kind === "native_nodes_and_certified_endpoints"
    ? "Saved solver nodes and certified endpoints" : "Archived trajectory samples";
}
