# Completed finite four-route batch

The single frozen run completed successfully. All four prescribed routes passed their flight and wait certificates. It used **72 native flight solves, 286 SCvx iterations and 6 wait certificates**, with the CUDA SCvx outer loop and **QOCO** conic solver. There were 4 archived first-leg seeds and 68 native cold starts, no new route generation or Lambert calls, and no retries, retiming or cargo reduction.

| Candidate | Flights | SCvx iterations | Wait certificates | Route stage, seconds | Final dry mass, kg |
| --- | ---: | ---: | ---: | ---: | ---: |
| Ship 19 rank 6 | 19 | 74 | 1 | 5.190 | 568.791844 |
| Ship 8 rank 0 | 17 | 70 | 2 | 4.790 | 589.801389 |
| Ship 19 rank 7 | 17 | 67 | 1 | 4.335 | 578.790980 |
| Ship 8 rank 1 | 19 | 75 | 2 | 4.732 | 554.927723 |

The route stages include native calls, certificate work and retained-output overhead. The complete worker took **43.831800 seconds**, including **21.430983 seconds** for its independent CPU fleet checker. One CUDA fleet selection took 0.003124 seconds as reported by its wrapper. These are unpaired measurements; they do not establish a speedup over another implementation or measure throughput of newly generated solutions.

The selector chose ship 19 rank 6 and ship 8 rank 0. Both original full-fleet checkers passed on Result SHA `3daeb39230604a2efb2795dd8a7b7c362363cb01348a066801316f536cc74542`: **13,016.287798820184 weighted kg** and **14,283.2032854218 raw kg**. This is +23.880057595487 weighted kg and +11.718001369200 raw kg against the sealed v628 baseline. All numerical and physics acceptance settings remained unchanged.

The useful policy change was to use the complete fleet's actual raw-mass margin. Ship 8 rank 0 brings approximately **12.676890 more weighted kg while returning 12.813142 fewer raw kg** than its previous route. That route can be admitted because the fleet still satisfies its original ship-count rule. The gain comes from certifying different route choices excluded by earlier local filters; it is not evidence that the weighted score is the same quantity as physical cargo mass.

The newer fleet was handled separately. Its other 21 sections were retained byte-for-byte while these same two certified sections were inserted. One additional authorized **CPU-only** checker pair passed on Result SHA `765cb7ef97d38926317dbbf4ae8700626dffec06c48af3d6f68dae47c154f3da`: **13,023.704900978004 weighted kg**, **14,291.006160165 raw kg**, 23 ships and 199 asteroids. That composition took 22.804060 seconds and performed no further native solve, GPU call, route search or leg certification. Its reports are in `../frontier-union-v629/output/`.

The tracked GPU session exited 0. The supervisor retained the shared lock until its subreaper confirmed no owned descendants and the final compute-process inventory was empty. `output/launch-report.json` binds the original ready SHA `6a63b9506655dd1590c7c470a5a5314315ba565475658a82ea3adbd28cd9bf6b`. The prelaunch README, ready manifest and their indexed files are unchanged.

Display preparation is separate again: `../frontier-visualizer-v629/` reuses the unchanged 21 histories and constructs the two new displays from saved native nodes, certificate endpoints and exact Result events. It introduces no trajectory propagation or intermediate samples. The three waits without saved dense histories are explicit display gaps.
