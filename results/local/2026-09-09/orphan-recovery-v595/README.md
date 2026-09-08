# Improved verified 23-ship fleet — v595

**14,051.854894 physical kg / 12,810.135953 fixed-bonus weighted kg**,
610.950213 physical kg per ship. Both mission checkers and a fresh independent
audit pass. The gain over retained v11 is 4.052019 raw kg / 4.941851 weighted kg.

See [the experiment report and loading instructions](../../../../docs/GPU_ORPHAN_RECOVERY.md)
and [independent audit](audit/README.md). `Result.txt` is the improved full fleet;
`raw.tar.gz` contains the complete original successful output and failed-start
evidence. `source.tar.gz` holds the pinned Python and input snapshot. This run
used GPU candidate arithmetic with the recorded CPU winner-selection compatibility
flag; it does not establish a fully GPU-controlled application.
