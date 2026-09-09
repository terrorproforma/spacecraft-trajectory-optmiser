The first package passed its Linux audit but failed the Windows comparison of the final diagnostic summary. The original package and failure output are preserved in `portability/original-package.zip` and `portability/original-windows-failure.log` inside `evidence.zip`.

The only difference was the reporting-only ideal rocket-equivalent propellant scale: `3.036027177824058` kg from Linux versus `3.036027177824057` kg from Windows. This is two floating-point ULPs, about 8.88e-16 kg, from the `expm1` calculation. It is not measured propellant, a certificate result or a physics gate.

The portable summary comparison now allows at most two ULPs for that single named field. Every indexed byte, original input, residual component, axis and unit conversion, counter, status and physical qualification comparison remains exact. Regression checks reject a change to any other field and an estimate outside that two-ULP bound. The sealed ready inputs and the eight raw output files are unchanged.
