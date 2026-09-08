# Reduced-coordinate balance diagnostic

This CPU-only calculation uses the retained qualified original-QP reference
vectors to inspect the diagonal distance proxy after removal of the epigraph
variables and their two multipliers. It performs no solver or CUDA calls.

For `a=||B D x_ref||` and `b=||O R y_ref||`, the expression
`omega*a*a+b*b/omega` is minimized at `omega=b/a`. The field named
`unit_over_oracle_distance_bound_coefficient` records `(a*a+b*b)/(2*a*b)`.
It is only a ratio of diagonal distance coefficients. It is not the complete
PDHG metric, which also includes a cross term, and it is not a convergence-rate
or speedup bound. None of these oracle weights is used to tune a solver run.

The projected reference is not independently qualified for the reduced
nonsmooth KKT system. Its tiny nonzero values can have interior original-QP
dual pairs; neither snapping those values nor replacing their duals is justified
before checking the original supplied point. The separate exact-map review
quantifies this hazard.

Joint smooth-plus-L1 normalization has a diagonal reference coefficient ratio
of about 9,703 for conditioning and 1.38 for difficult. Smooth-only normalization
reverses the tradeoff: about 9.68 and 85,602. Both hypothetical policies retain
the complete original L1 penalty in the mathematical objective and proximal
step; only the positive coordinate normalization differs. These diagnostics
justify neither a universal normalization change nor a performance claim.
Retain the documented joint baseline and let actual cold results determine
whether a separate adaptive-balancing experiment is useful.
