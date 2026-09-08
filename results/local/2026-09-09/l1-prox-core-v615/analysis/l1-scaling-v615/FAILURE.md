# Initial CPU analysis attempt

The first script used the independent auditor's extended-precision sparse
matrix type. SciPy ARPACK rejected that type before producing an eigenpair:
`ValueError: matrix type must be 'f', 'd', 'F', or 'D'`.

The original script and auditor are retained here. The corrected native-FP64
analysis and its complete outputs are in the adjacent `l1-scaling-v615b`
directory. This was a CPU analysis failure; no solver or GPU calls occurred,
and no partial eigenvalue result is claimed.
