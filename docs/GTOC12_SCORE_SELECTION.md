# Verified objective selection

GTOC12 route, retiming and fleet commands now report raw cargo separately from
the configured objective. With bonus weights enabled, `score_kg` is the
independently verified `weighted_score_fixed_bonus_kg`. The offline official
checker's rounded raw cargo cannot replace it. Missing or nonfinite weighted
scores reject the result. With weights disabled, `score_kg` is verified raw kg.

Previously a heavier, lower-weighted candidate could replace a better route,
and the final cluster pass could overwrite its better verified incumbent.
Every retimed variant now passes verification before entering selection; a
failed primary variant cannot hide another valid variant. The cluster command
retains its best verified incumbent and reports final verification failure with
a nonzero exit status.

Cooperative collectors need their miners' deploying ships to be verified. The
candidate is checked with its transitive, epoch-matched supplier context. Its
ranking mass includes only the asteroids that candidate returns. Verification
and viewer artifacts refer to the same context.

The global ship-count rule can reject a small supplier context even when a
larger selected fleet would be valid. A context failing **only Error301** may
produce an explicitly provisional route column after independent trajectory,
mass and mining checks. It is not a scored fleet: `ok` and `accepted` remain
false, `score_kg` is null, and `candidate_score_kg` is separately labelled.
Unknown official errors, additional violations and missing suppliers reject
the candidate. The final selected fleet must pass full verification; an empty
master is never exported as a scored solution.

## Validation

123 focused CPU regression tests pass, covering weighted/raw objective reversal,
incumbent retention, failed variants, cooperative dependencies, provisional
columns, empty masters and final verification. Ruff passes. Source hashes were
unchanged during validation.

Two synthetic probes also exercised the pinned official executable. Three
zero-cargo coasting ships fail only Error301 (official exit 3); adding a 1 kg
mass inconsistency produces Error203 as well and is rejected. These fixtures
are classifier tests, not accepted missions. No physics tolerance changed.

[Logs, commands, hashes and official-checker probes](../results/local/2026-09-09/objective-verification-v591/summary.json)
are retained. This correction does not itself establish a higher fleet score.
The retained best fleet remains 12,805.194102489 weighted kg and
14,047.802874744 raw kg across 23 ships.
