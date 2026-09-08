# Publication notes

This bundle preserves all **80 retrieved remote files** byte-for-byte, including
the representative best Result and its existing viewer export. Both original
retrieval manifests and their recorded source paths, byte counts and SHA-256
digests are retained under `evidence/`. The audit README and supporting metadata
are copied unchanged.

The two transport archives are omitted because their extracted contents are
already present. Their original hashes and sizes remain in
`publication-manifest.json`, `download-verified.json` and
`viewer-download-verified.json`. Packaging verified every remote file both before
and after copying. It performed no network access, remote writes, GPU work or
new trajectory verification.

The [retrieval audit](README.md) explains the component measurements and the
unequal work in the faster-looking campaign sample. In particular, the separate
selection-only experiment forces an eligible winner; the main ship-2 benchmark
has no improving winner. All four retained best fleets pass both recorded
checkers at 12,810.135953 weighted kg. The 120.727-second candidate1 campaign
includes a failed second refinement and does not establish a selection speedup.

See [GPU joint itinerary](../../../../docs/GPU_JOINT_ITINERARY.md) for the audited
H100 interpretation, and [local GPU joint candidates](../../../../docs/GPU_JOINT_CANDIDATES.md)
for the separate v596 source/validation snapshot. File hashes for this publication
are recorded in `evidence-files.json`.
