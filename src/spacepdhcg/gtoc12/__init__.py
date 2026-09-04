"""GTOC12 "Sustainable Asteroid Mining" replay track for SpacePDHCG/OrbitWeaver.

Public surface:

- :mod:`spacepdhcg.gtoc12.constants` -- official constants and rule helpers;
- :mod:`spacepdhcg.gtoc12.data` -- pinned data access (catalogue, bonus table, verifier);
- :mod:`spacepdhcg.gtoc12.ephemeris` -- official two-body ephemerides and Kepler propagation;
- :mod:`spacepdhcg.gtoc12.solution` -- official solution-file model, parser and writer;
- :mod:`spacepdhcg.gtoc12.verifier` -- independent verifier/scorer;
- :mod:`spacepdhcg.gtoc12.official` -- wrapper around the organisers' verifier binary;
- :mod:`spacepdhcg.gtoc12.lambert`, :mod:`spacepdhcg.gtoc12.screening`,
  :mod:`spacepdhcg.gtoc12.reduced_instance`, :mod:`spacepdhcg.gtoc12.search`,
  :mod:`spacepdhcg.gtoc12.low_thrust`, :mod:`spacepdhcg.gtoc12.pipeline` -- OrbitWeaver route
  search and low-thrust refinement on the reduced instance;
- :mod:`spacepdhcg.gtoc12.viewer_export` -- trajectory-viewer dataset emission.
"""

from __future__ import annotations

from .verifier import Gtoc12Verifier, VerificationReport, verify_solution_file

__all__ = ["Gtoc12Verifier", "VerificationReport", "verify_solution_file"]
