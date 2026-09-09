import test from 'node:test';
import assert from 'node:assert/strict';
import { replayGaps, visibleSegmentRanges, currentReplaySample, sampleSourceLabel } from '../replay.js';
import { bodyInstances } from '../gtoc12.js';

test('missing coast segments stay absent at every playback boundary', () => {
  const gaps = replayGaps({ points_txyz: Array(5), gap_after_indices: [1, 3] });
  const expected = [[], [], [[0, 1]], [[0, 1]], [[0, 1], [2, 1]], [[0, 1], [2, 1]]];
  for (let visible = 0; visible <= 5; visible++) assert.deepEqual(visibleSegmentRanges(5, visible, gaps), expected[visible]);
  assert.deepEqual(visibleSegmentRanges(5, 5), [[0, 4]]);
  assert.deepEqual(visibleSegmentRanges(3, 3, new Set([0, 1])), []);
});

test('malformed gaps fail before geometry construction', () => {
  for (const gaps of [[-1], [4], [1, 1], [2, 1], [1.2], '1']) {
    assert.throws(() => replayGaps({ points_txyz: Array(5), gap_after_indices: gaps }), /gap/);
  }
});

test('current position disappears only inside a missing interval', () => {
  const ship = { times: [0, 10, 20], gapAfter: new Set([1]) };
  for (const [epoch, expected] of [[-1, -1], [0, 0], [5, 0], [10, 1], [11, -1], [19, -1], [20, 2], [21, 2]]) {
    assert.equal(currentReplaySample(ship, epoch), expected);
  }
});

test('body instances omit a ship during an unsampled coast', () => {
  const ship = { times: [0, 10, 20], gapAfter: new Set([1]), points: [[0, 1, 0], [1, 0, 0], [0, -1, 0]], colour: [1, 1, 1, 1], index: 0 };
  const scene = { ships: [ship], asteroids: [], asteroidStatus: [] };
  const radii = { sun: 1, earth: 1, ship: 1 };
  const positions = { earth: [0, 0, 1] };
  for (const [epoch, count] of [[10, 3], [15, 2], [20, 3]]) assert.equal(bodyInstances(scene, { selected: null, epoch }, radii, positions).count, count);
});

test('solver samples carry an explicit source label', () => {
  assert.match(sampleSourceLabel({ display_sample_kind: 'native_nodes_and_certified_endpoints' }), /solver nodes and certified endpoints/);
  assert.equal(sampleSourceLabel({}), 'Archived trajectory samples');
});
