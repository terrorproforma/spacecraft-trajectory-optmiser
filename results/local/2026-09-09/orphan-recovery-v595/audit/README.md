# Independent audit of the promoted v595 fleet

**Pass.** A fresh CPU-only run of both the independent and official verifiers
reproduces the promoted fleet's results. Neither implementation nor solver settings
were edited during this audit; no GPU execution was performed.

| Metric | Original v11 | Promoted v595 |
| --- | ---: | ---: |
| Physical returned mass | 14,047.802874744 kg | **14,051.854893909 kg** |
| Fixed-bonus weighted score | 12,805.194102489 kg | **12,810.135953049 kg** |
| Physical mass per ship | 610.774038032 kg | **610.950212779 kg** |
| Ships | 23 | 23 |
| Asteroids whose ore reached Earth | 194 | 195 |
| Deployed asteroid footprint | 196 | 196 |

The gain is **4.052019165 physical kg** and **4.941850560 weighted kg**. Ship 15 is
the only changed ship, rising from 605.639972621 to 609.691991786 physical kg. It
deploys on 19102 at MJD 66969 and collects there at MJD 67425, after a 456-day camp,
returning 12.484599589 kg from that miner. Re-timing its other pickups costs
8.432580424 raw kg, leaving the net gain above. The new miner's coefficient is 1;
the differently weighted changes elsewhere explain why weighted gain exceeds raw gain.

All 22 other ships have token-identical solution sections. The promoted file is
exactly the first candidate file and was not replaced by the later, lower-scoring
second candidate. The original solution, recorded driver and all 52 frozen Python
source hashes match. The viewer manifest's solution and trajectory-data hashes match
the actual files.

The promoted solution SHA-256 is:

```text
33701ef2b797f44ef2e8aa50a2dd59cb238df9aab604e7cef25ead6cbdd669e8
```

`fresh-verification.json` contains the new independent result, the official result
and stdout, independently scored asteroid masses, per-ship unloaded mass, pinned
input hashes and unchanged default tolerances. The independent check took 20.405 s;
the official check took 0.197 s. The independent maximum position, velocity and mass
errors exactly match the unchanged baseline maxima. Directly summing emitted Earth
unloading events also reproduces total returned mass within 1e-7 kg.

## Actual work and timing scope

- 18 new collection orders, each evaluated on 15-day and 5-day grids: **36** retime
  driver calls, **277** internal DP and **277** forward calls.
- Four proxy layouts closed; 32 failed the surrogate mass budget. Two candidates
  improved the proxy objective enough to refine. One was promoted.
- **23,142** joint candidate evaluations in **132** GPU batches, and **106,024,898**
  logical Lambert branch requests over **53,012,449** element-hop requests. These
  are intermediate search work, not millions of independently certified trajectories.
- Two complete route refinements used **36 native leg solves**. All returned
  `converged`, with **159 iterations / 153 accepted iterations**. Their measured
  native-call times sum to **9.650684 s**; the two route refinements total
  **11.487757 s** including surrounding work.

The **78.115478 s** campaign timer begins after imports, data loading, source/library
hashing and original solution parsing. It includes the initial complete-fleet
verification, GPU search and context setup, both route refinements, both subsequent
complete-fleet verifications, the viewer export and checkpoint writing. It excludes
the final report serialization and process exit. The fresh audit verification is
additional and is not part of that timer. This is neither pure GPU compute time nor
a repeated comparative speed benchmark.

## Publication qualifications

Publish `output-compatible`, not the preceding failed `output` directory. The
successful run used `SPACEPDHCG_TEST_GTOC12_JOINT_DEVICE_SELECTION=0`: joint candidate
evaluation runs on GPU, while the winning row is chosen using CPU NumPy argmax.
The earlier attempt requested an unavailable native winner-selection entry point.
Use the recorded compatible flag when reproducing this library/configuration.

SCvx seed, dynamics, assembly, QOCO and outer-loop backends are CUDA, at 40 iterations
and 2-day nodes with unchanged physics tolerances. Python orchestration, compatible
winner selection and independent verification remain on CPU. The result supports a
verified score improvement; it does not establish that the whole program is GPU-native.

No blocking publication issue was found. `audit.json` gives machine-readable checks,
and `audit.py` reproduces artifact, score and work-count assertions without GPU access.
