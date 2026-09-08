# Completion benchmark results (v621)

The production GPU completion adapter helps the fitted cost model at practical shortlist sizes. The small flat-model batch is slower on the GPU. The production default remains unchanged.

This was one local RTX 5090 run of the frozen final-b Python adapter and normal full CUDA core. It evaluated **20 unique historical controls**, repeated in fixed sequences: **60 GPU calls and 14,200 proxy candidate evaluations per backend**. All prescribed decisions, cargo, metadata and numerical comparisons passed; retained workspace buffers were reused. There were no new trajectories, physics certificates or score changes.

Each row has one first call and four warm calls per backend. Warm timings are medians; the speed ratio is CPU median divided by GPU median. Packing share is the median share from matching individual GPU samples. Method timings include normal host packing and plan readout.

| Model | Batch | Warm CPU ms | Warm GPU ms | CPU/GPU ratio | GPU proxy eval/s | Host packing share | Native kernel µs |
|---|---:|---:|---:|---:|---:|---:|---:|
| Flat | 4 | 0.333 | 0.751 | 0.443× | 5,326 | 23.7% | 29.50 |
| Flat | 24 | 1.210 | 1.256 | 0.963× | 19,103 | 50.6% | 31.81 |
| Flat | 48 | 2.250 | 1.945 | 1.157× | 24,678 | 58.3% | 65.06 |
| Flat | 64 | 2.934 | 2.357 | 1.245× | 27,151 | 65.6% | 71.54 |
| Flat | 256 | 10.559 | 6.898 | 1.531× | 37,115 | 81.0% | 87.65 |
| Flat | 1024 | 42.672 | 25.061 | 1.703× | 40,860 | 86.5% | 85.50 |
| Existing fit | 4 | 1.212 | 0.815 | 1.488× | 4,909 | 34.2% | 41.79 |
| Existing fit | 24 | 4.850 | 1.617 | 3.000× | 14,847 | 60.9% | 43.79 |
| Existing fit | 48 | 9.544 | 2.812 | 3.394× | 17,070 | 71.3% | 84.77 |
| Existing fit | 64 | 13.190 | 3.389 | 3.892× | 18,885 | 75.1% | 80.80 |
| Existing fit | 256 | 49.677 | 11.228 | 4.425× | 22,801 | 87.9% | 96.62 |
| Existing fit | 1024 | 200.120 | 41.207 | 4.856× | 24,850 | 91.4% | 100.13 |

The observed v616 shortlist setting was 24; the source default is 48. The four-request case uses historical DP metadata at a schedule-sized batch, not a measured heuristic scheduler invocation. Sizes 64/256/1024 are prospective scaling probes. Repetition provides a controlled component comparison; it does not show that a live search can assemble those batches at no cost.

The fitted model gains about 3.000× at 24 and 3.394× at 48. The flat model is approximately tied at 24 (0.963×), and its four-request batch loses substantially (0.443×). At 1024 fitted requests the gain is 4.856×, with host metadata packing dominating the remaining GPU method time. This points toward retained metadata/device inputs as a next performance hypothesis; it does not justify making every small completion batch use the GPU.

## First call and setup

A new completion workspace was created for each group. Only the first group is the first GPU completion in this process; these are not repeated cold-device measurements. Backend setup is separate from the first method call. First-call latency must not be averaged into warm throughput.

| Model | Batch | First CPU ms | First GPU ms | Backend setup ms | Backend close ms |
|---|---:|---:|---:|---:|---:|
| Flat | 4 | 0.518 | 235.305 | 68.603 | 0.801 |
| Flat | 24 | 1.177 | 1.726 | 0.227 | 0.399 |
| Flat | 48 | 2.248 | 2.448 | 0.205 | 0.431 |
| Flat | 64 | 2.849 | 3.642 | 0.198 | 0.474 |
| Flat | 256 | 10.463 | 7.372 | 0.209 | 0.520 |
| Flat | 1024 | 42.629 | 27.102 | 0.227 | 0.603 |
| Existing fit | 4 | 1.354 | 1.300 | 0.191 | 0.466 |
| Existing fit | 24 | 4.957 | 2.139 | 0.205 | 0.406 |
| Existing fit | 48 | 9.194 | 3.232 | 0.226 | 0.435 |
| Existing fit | 64 | 12.825 | 3.889 | 0.243 | 0.433 |
| Existing fit | 256 | 52.135 | 11.508 | 0.210 | 0.405 |
| Existing fit | 1024 | 197.972 | 42.327 | 0.209 | 0.708 |

The measured worker took 6.310143 s. Timed CPU method calls summed to 1.694853 s and GPU method calls to 0.740371 s. Input construction summed to 0.607139 s; post-method checking/capture summed to 1.574397 s. Setup, close, signatures, journal writes and other runner work account for further time. These diagnostic checks are not full trajectory/fleet verification and are excluded from the method throughput numbers.

Raw vectors, parity results, compact first outcomes, every timing sample and outcome hash are retained under `output/`; `raw-index.json` freezes their hashes. `summary.json` contains exact values and sample definitions. Four warm samples per row support this bounded comparison, not a broad statistical performance claim.

No production source, defaults or objective changed as a result of this summary. GPU-native search work remains incomplete outside the tested completion component; the retained mission score is unchanged.
