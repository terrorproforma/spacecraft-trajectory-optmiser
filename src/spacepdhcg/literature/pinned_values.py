"""Curated provenance records: every literature value used by the campaign, with digits as printed.

Sources were read on 2026-09-03 through the open-access copies named in each record.  Values
that could not be recovered from any source are recorded as ``descriptive-only`` with the reason.
"""

from __future__ import annotations

from spacepdhcg.literature.provenance import ProvenanceRecord, ProvenanceSource

ACCESSED = "2026-09-03"

SZMUK_2018 = ProvenanceSource(
    title="Successive Convexification for 6-DoF Mars Rocket Powered Landing with Free-Final-Time",
    authors="M. Szmuk, B. Acikmese",
    year=2018,
    doi="10.2514/6.2018-0617",
    url="https://arxiv.org/abs/1802.03827",
    version="arXiv:1802.03827 (ar5iv HTML rendering)",
    accessed=ACCESSED,
)
ACIKMESE_2007 = ProvenanceSource(
    title="Convex Programming Approach to Powered Descent Guidance for Mars Landing",
    authors="B. Acikmese, S. R. Ploen",
    year=2007,
    doi="10.2514/1.27553",
    url="https://doi.org/10.2514/1.27553",
    version="JGCD 30(5) 1353-1366",
    accessed=ACCESSED,
)
WENZEL_2018 = ProvenanceSource(
    title="On-Board Convex Optimization for Powered Descent Landing of EAGLE (Master thesis, DLR)",
    authors="A. Wenzel",
    year=2018,
    url="https://elib.dlr.de/118732/1/MasterthesisAWENZEL.pdf",
    location="Chapter 4, Mars pinpoint landing example reproduced from [1] (Acikmese & Ploen 2007)",
    accessed=ACCESSED,
)
BLACKMORE_2010 = ProvenanceSource(
    title=(
        "Minimum-Landing-Error Powered-Descent Guidance for Mars Landing Using Convex Optimization"
    ),
    authors="L. Blackmore, B. Acikmese, D. P. Scharf",
    year=2010,
    doi="10.2514/1.47202",
    url="http://larsjamesblackmore.com/BlackmoreEtAlJGCD10.pdf",
    version="JGCD 33(4) 2010",
    accessed=ACCESSED,
)
TAFAZZOL_2024 = ProvenanceSource(
    title=(
        "Comparison of Control Regularization Techniques for Minimum-Fuel Low-Thrust Trajectory "
        "Design Using Indirect Methods"
    ),
    authors="S. Tafazzol, E. Taheri",
    year=2024,
    url="https://arxiv.org/abs/2409.01490",
    version="arXiv:2409.01490 (HTML)",
    accessed=ACCESSED,
)
CHARI_2024 = ProvenanceSource(
    title=(
        "Fast Monte Carlo Analysis for 6-DoF Powered-Descent Guidance via GPU-Accelerated "
        "Sequential Convex Programming"
    ),
    authors="G. M. Chari et al.",
    year=2024,
    doi="10.2514/6.2024-1762",
    url="https://arxiv.org/abs/2404.18034",
    accessed=ACCESSED,
)
TOPS_REPO = ProvenanceSource(
    title="ESA zero-order-hold repository (TOPS database)",
    authors="D. Izzo, H. Holt, G. Acciarini, L. Beauregard, Y. Shimane",
    year=2026,
    url="https://gitlab.com/EuropeanSpaceAgency/zero-order-hold",
    revision="24fe8849b403af376773f09b64b5132e5591b94e",
    version="main @ 2026-04-24",
    accessed=ACCESSED,
)
GTOPX = ProvenanceSource(
    title="GTOPX Space Mission Benchmarks (official solution files)",
    authors="M. Schlueter, M. Neshat, M. Wahib, M. Munetomo, M. Wagner",
    year=2021,
    doi="10.1016/j.softx.2021.100666",
    url="https://www.midaco-solver.com/index.php/about/benchmarks/gtopx",
    version="GTOPX 1.0",
    licence="GPL",
    accessed=ACCESSED,
)
GTOC9_KELVINS = ProvenanceSource(
    title="GTOC9 The Kessler Run (Kelvins problem pages)",
    year=2017,
    url="https://kelvins.esa.int/gtoc9-kessler-run/",
    accessed=ACCESSED,
)
GTOC12_PORTAL = ProvenanceSource(
    title="GTOC12 Sustainable Asteroid Mining (GTOC portal archive)",
    year=2023,
    url="https://sophia.estec.esa.int/gtoc_portal/?page_id=1261",
    accessed=ACCESSED,
)


def _rec(**kwargs) -> ProvenanceRecord:
    kwargs.setdefault("approximate", False)
    kwargs.setdefault("verification_status", "source-verified")
    kwargs["id"] = str(kwargs["id"]).lower()
    return ProvenanceRecord(**kwargs)


def _table(
    profile: str, source: ProvenanceSource, location: str, rows: list[tuple]
) -> list[ProvenanceRecord]:
    records = []
    for quantity, value, text, units in rows:
        records.append(
            _rec(
                id=f"{profile}.{quantity}",
                profile=profile,
                quantity=quantity,
                value=value,
                value_text=text,
                units=units,
                evidence_label="published-reference",
                extraction_method="table",
                source=ProvenanceSource(**{**source.as_dict(), "location": location}),
            )
        )
    return records


def curated_records() -> list[ProvenanceRecord]:
    records: list[ProvenanceRecord] = []

    # ------------------------------------------------------------- Szmuk & Acikmese 2018
    p = "szmuk-acikmese-2018-pd6-2d"
    records += _table(
        p,
        SZMUK_2018,
        "Table 1 (Simulation Parameters)",
        [
            ("gravity", [-1.0, 0.0, 0.0], "-e_1", "UL/UT^2"),
            ("wet_mass", 2.0, "2.00", "UM"),
            ("dry_mass", 1.0, "1.00", "UM"),
            ("thrust_min", 0.3, "0.30", "UM UL/UT^2"),
            ("thrust_max", 5.0, "5.00", "UM UL/UT^2"),
            ("gimbal_max_deg", 20.0, "20", "deg"),
            ("tilt_max_deg", 90.0, "90", "deg"),
            ("glide_slope_deg", 20.0, "20", "deg"),
            ("omega_max_deg", 60.0, "60", "deg/UT"),
            ("inertia_diagonal", 0.01, "1e-2 . I_3x3", "UM UL^2"),
            ("thrust_arm", [-0.01, 0.0, 0.0], "-1e-2 . e_1", "UL"),
        ],
    )
    records += _table(
        p,
        SZMUK_2018,
        "Table 2 (B.C.'s and Algorithm Parameters)",
        [
            ("virtual_weight", 1.0e5, "1e5", "dimensionless"),
            ("trust_weight", 1.0e-3, "1e-3", "dimensionless"),
            ("sigma_trust_weight", 1.0e-1, "1e-1", "dimensionless"),
            ("virtual_tolerance", 1.0e-10, "1e-10", "dimensionless"),
            ("trust_tolerance", 1.0e-3, "1e-3", "dimensionless"),
            ("max_iterations", 15, "15", "dimensionless"),
            ("nodes", 50, "50", "dimensionless"),
            ("initial_position", [4.0, 4.0, 0.0], "[4 4 0]^T", "UL"),
            ("final_velocity", [-0.1, 0.0, 0.0], "-1e-1 . e_1", "UL/UT"),
            ("initial_omega", [0.0, 0.0, 0.0], "0", "deg/UT"),
            ("final_quaternion", [1.0, 0.0, 0.0, 0.0], "[1 0 0 0]^T", "dimensionless"),
        ],
    )
    records.append(
        _rec(
            id=f"{p}.initial_velocity_horizontal",
            profile=p,
            quantity="initial_velocity_horizontal_west",
            value=4.0,
            value_text="4",
            units="UL/UT",
            evidence_label="published-reference",
            extraction_method="text",
            source=ProvenanceSource(
                **{
                    **SZMUK_2018.as_dict(),
                    "location": (
                        "Section 4.1: 'initial horizontal velocity of 4 [UL/UT] to the west'"
                    ),
                }
            ),
            notes="vertical component not stated; assumed zero in the profile",
        )
    )
    records.append(
        _rec(
            id=f"{p}.alpha_mdot",
            profile=p,
            quantity="alpha_mdot",
            value=None,
            value_text="1/(I_sp g_0) (no numerical value printed)",
            units="UT/UL",
            evidence_label="descriptive-only",
            extraction_method="unrecoverable",
            source=ProvenanceSource(
                **{
                    **SZMUK_2018.as_dict(),
                    "location": "Section 2.1 definition; absent from Table 1",
                }
            ),
            unrecoverable_reason=(
                "the paper defines alpha_mdot symbolically but Table 1 prints no value; the "
                "profile assumes 0.01 UT/UL"
            ),
        )
    )
    records.append(
        _rec(
            id=f"{p}.converged_time_of_flight",
            profile=p,
            quantity="converged_time_of_flight",
            value=None,
            value_text="Figure 2 (time-of-flight versus iteration; no digits printed)",
            units="UT",
            evidence_label="descriptive-only",
            extraction_method="unrecoverable",
            source=ProvenanceSource(**{**SZMUK_2018.as_dict(), "location": "Figure 2"}),
            unrecoverable_reason="the converged t_f is shown only graphically",
            objective_convention="time-of-flight",
        )
    )
    records.append(
        _rec(
            id=f"{p}.tf_guess_sweep_spread",
            profile=p,
            quantity="tf_guess_sweep_spread",
            value=0.01,
            value_text="within 0.01 [UT] of each other",
            units="UT",
            evidence_label="published-reference",
            extraction_method="text",
            source=ProvenanceSource(**{**SZMUK_2018.as_dict(), "location": "Section 4.1"}),
            objective_convention="time-of-flight",
        )
    )
    records.append(
        _rec(
            id=f"{p}.iterations_to_converge",
            profile=p,
            quantity="iterations_to_converge",
            value=6,
            value_text="6 ('convergence was obtained by the sixth iteration')",
            units="iterations",
            evidence_label="published-reference",
            extraction_method="text",
            source=ProvenanceSource(**{**SZMUK_2018.as_dict(), "location": "Section 4.1"}),
        )
    )

    # ------------------------------------------- Acikmese & Ploen 2007 (via secondary sources)
    p = "acikmese-ploen-2007-pd3"
    for quantity, value, text, units in [
        ("gravity_mars", -3.7114, "-3.7114", "m/s^2"),
        ("wet_mass", 1905.0, "1905", "kg"),
        ("specific_impulse", 225.0, "225", "s"),
        ("thruster_max_thrust", 3100.0, "3100", "N"),
        ("throttle_min", 0.3, "0.3", "dimensionless"),
        ("throttle_max", 0.8, "0.8", "dimensionless"),
        ("thruster_count", 6, "6", "dimensionless"),
        ("cant_angle", 27.0, "27", "deg"),
        ("glide_slope", 4.0, "4", "deg"),
        ("initial_position", [1500.0, 0.0, 2000.0], "[1500 0 2000]", "m"),
        ("initial_velocity", [-75.0, 0.0, 100.0], "[-75 0 100]", "m/s"),
        ("time_of_flight_glide_slope", 81.0, "81", "s"),
        ("time_of_flight_no_glide_slope", 72.0, "72", "s"),
    ]:
        records.append(
            _rec(
                id=f"{p}.{quantity}",
                profile=p,
                quantity=quantity,
                value=value,
                value_text=text,
                units=units,
                evidence_label="published-reference",
                extraction_method="secondary-citation",
                source=ProvenanceSource(
                    **{**ACIKMESE_2007.as_dict(), "location": "Section V numerical example"}
                ),
                secondary_source=WENZEL_2018,
                notes=(
                    "read from the DLR thesis parameter table that reproduces the 2007 example; "
                    "the JGCD text is paywalled"
                ),
            )
        )
    records.append(
        _rec(
            id=f"{p}.fuel_used_glide_slope",
            profile=p,
            quantity="fuel_used_glide_slope",
            value=399.5,
            value_text="399.5",
            units="kg",
            evidence_label="published-reference",
            extraction_method="secondary-citation",
            source=ProvenanceSource(
                **{
                    **ACIKMESE_2007.as_dict(),
                    "location": "numerical example with 4 deg glide slope, t_f = 81 s",
                }
            ),
            secondary_source=WENZEL_2018,
            objective_convention="propellant-used",
            notes=(
                "Wenzel: 'the fuel consumption of 400.48 kg is slightly more than the one found by "
                "[1] with 399.5 kg'"
            ),
        )
    )
    records.append(
        _rec(
            id=f"{p}.fuel_used_no_glide_slope",
            profile=p,
            quantity="fuel_used_no_glide_slope",
            value=387.9,
            value_text="387.9",
            units="kg",
            evidence_label="published-reference",
            extraction_method="secondary-citation",
            source=ProvenanceSource(
                **{
                    **ACIKMESE_2007.as_dict(),
                    "location": "numerical example without glide slope, t_f = 72 s",
                }
            ),
            secondary_source=WENZEL_2018,
            objective_convention="propellant-used",
            notes="Wenzel: 'the simulations in [1] show a fuel consumption of 387.9 kg'",
        )
    )
    records.append(
        _rec(
            id=f"{p}.alpha_definition",
            profile=p,
            quantity="mass_flow_coefficient_convention",
            value="alpha = 1/(I_sp g_e cos phi)",
            value_text=(
                "alpha = 1/(I_sp g_e cos phi) reproduces 399.5/387.9 kg; 1/(I_sp g_e) does not "
                "(362/351 kg)"
            ),
            units="s/m",
            evidence_label="measured-local",
            extraction_method="local-measurement",
            source=ProvenanceSource(
                title="spacepdhcg literature reproduction",
                url="file:benchmarks/literature/reference_reproduction.json",
                accessed=ACCESSED,
            ),
            notes="formulation discrepancy resolved by evaluating both conventions (Phase 1 rule)",
        )
    )

    # ------------------------------------------------------------- Blackmore 2010
    p = "blackmore-2010-pd3-case1"
    records += _table(
        p,
        BLACKMORE_2010,
        "Eq. (72)",
        [
            ("gravity", [-3.7114, 0.0, 0.0], "g = [-3.7114 0 0]^T m/s^2", "m/s^2"),
            ("dry_mass", 1505.0, "1505", "kg"),
            ("wet_mass", 1905.0, "1905", "kg"),
            ("alpha_printed", 4.53e-4, "4.53e-4", "s/m"),
            ("rho1", 4972.0, "4972", "N"),
            ("rho2", 13260.0, "13260", "N"),
        ],
    )
    records.append(
        _rec(
            id=f"{p}.glide_slope",
            profile=p,
            quantity="glide_slope",
            value=4.0,
            value_text="4",
            units="deg",
            evidence_label="published-reference",
            extraction_method="text",
            source=ProvenanceSource(
                **{
                    **BLACKMORE_2010.as_dict(),
                    "location": "Section V: 'descending at an angle shallower than 4 deg'",
                }
            ),
        )
    )
    records.append(
        _rec(
            id=f"{p}.initial_position",
            profile=p,
            quantity="initial_position",
            value=[1500.0, 500.0, 2000.0],
            value_text="r0 = [1500 m 500 m 2000 m]",
            units="m",
            evidence_label="published-reference",
            extraction_method="text",
            source=ProvenanceSource(
                **{**BLACKMORE_2010.as_dict(), "location": "Section V case 1 (Fig. 2 caption)"}
            ),
        )
    )
    records.append(
        _rec(
            id=f"{p}.initial_velocity",
            profile=p,
            quantity="initial_velocity",
            value=[-75.0, 0.0, 100.0],
            value_text="rdot0 = [-75 m/s 0 100 m/s]",
            units="m/s",
            evidence_label="published-reference",
            extraction_method="text",
            source=ProvenanceSource(
                **{**BLACKMORE_2010.as_dict(), "location": "Section V case 1 (Fig. 2 caption)"}
            ),
        )
    )
    records.append(
        _rec(
            id=f"{p}.fuel_used",
            profile=p,
            quantity="fuel_used",
            value=399.4,
            value_text="399.4",
            units="kg",
            evidence_label="published-reference",
            extraction_method="text",
            source=ProvenanceSource(
                **{
                    **BLACKMORE_2010.as_dict(),
                    "location": (
                        "Section V case 1: 'This solution requires 399.4 kg of fuel and has "
                        "t_f = 78.4 s'"
                    ),
                }
            ),
            objective_convention="propellant-used",
        )
    )
    records.append(
        _rec(
            id=f"{p}.time_of_flight",
            profile=p,
            quantity="time_of_flight",
            value=78.4,
            value_text="78.4",
            units="s",
            evidence_label="published-reference",
            extraction_method="text",
            source=ProvenanceSource(**{**BLACKMORE_2010.as_dict(), "location": "Section V case 1"}),
            notes="golden search terminated at a 3.0 s interval; 55 discretisation points",
        )
    )

    # ------------------------------------------------------------- Tafazzol & Taheri
    for p, location, rows in [
        (
            "tafazzol-taheri-earth-mars",
            "Table 1",
            [
                ("sun_mu", 132712440018.0, "132712440018", "km^3/s^2"),
                ("initial_mass", 1000.0, "1000", "kg"),
                ("specific_impulse", 2000.0, "2000", "s"),
                ("maximum_thrust", 0.5, "0.5", "N"),
                (
                    "departure_position",
                    [-140699693.0, -51614428.0, 980.0],
                    "[-140699693, -51614428, 980]",
                    "km",
                ),
                (
                    "departure_velocity",
                    [9.774596, -28.07828, 4.337725e-4],
                    "[9.774596, -28.07828, 4.337725e-4]",
                    "km/s",
                ),
                (
                    "arrival_position",
                    [-172682023.0, 176959469.0, 7948912.0],
                    "[-172682023, 176959469, 7948912]",
                    "km",
                ),
                (
                    "arrival_velocity",
                    [-16.427384, -14.860506, 9.21486e-2],
                    "[-16.427384, -14.860506, 9.21486e-2]",
                    "km/s",
                ),
                ("time_of_flight", 348.795, "348.795", "day"),
            ],
        ),
        (
            "tafazzol-taheri-earth-dionysus",
            "Table 3",
            [
                ("sun_mu", 132712440018.0, "132712440018", "km^3/s^2"),
                ("initial_mass", 4000.0, "4000", "kg"),
                ("specific_impulse", 3000.0, "3000", "s"),
                ("maximum_thrust", 0.32, "0.32", "N"),
                (
                    "departure_position",
                    [-3637871.081, 147099798.784, -2261.441],
                    "[-3637871.081, 147099798.784, -2261.441]",
                    "km",
                ),
                (
                    "departure_velocity",
                    [-30.265097, -0.8486854, 0.505e-4],
                    "[-30.265097, -0.8486854, 0.505e-4]",
                    "km/s",
                ),
                (
                    "arrival_position",
                    [-302452014.884, 316097179.632, 82872290.0755],
                    "[-302452014.884, 316097179.632, 82872290.0755]",
                    "km",
                ),
                (
                    "arrival_velocity",
                    [-4.533473, -13.110309, 0.656163],
                    "[-4.533473, -13.110309, 0.656163]",
                    "km/s",
                ),
                ("time_of_flight", 3534.0, "3534", "day"),
            ],
        ),
    ]:
        records += _table(p, TAFAZZOL_2024, location, rows)
    records.append(
        _rec(
            id="tafazzol-taheri-earth-mars.final_mass",
            profile="tafazzol-taheri-earth-mars",
            quantity="final_mass",
            value=603.935,
            value_text="603.935",
            units="kg",
            evidence_label="published-reference",
            extraction_method="text",
            source=ProvenanceSource(
                **{
                    **TAFAZZOL_2024.as_dict(),
                    "location": "Section 4.1: 'm(t_f) = m_f = 603.935 kg'",
                }
            ),
            objective_convention="final-mass",
        )
    )
    records.append(
        _rec(
            id="tafazzol-taheri-earth-dionysus.final_mass",
            profile="tafazzol-taheri-earth-dionysus",
            quantity="final_mass",
            value=2718.33,
            value_text="2718.33",
            units="kg",
            evidence_label="published-reference",
            extraction_method="text",
            source=ProvenanceSource(
                **{
                    **TAFAZZOL_2024.as_dict(),
                    "location": "Section 4.2: 'm(t_f) = m_f = 2718.33 kg'",
                }
            ),
            objective_convention="final-mass",
        )
    )
    records.append(
        _rec(
            id="tafazzol-taheri-earth-dionysus.revolutions",
            profile="tafazzol-taheri-earth-dionysus",
            quantity="revolutions",
            value=5,
            value_text="5 ('The most optimal solution involves five orbital revolutions')",
            units="revolutions",
            evidence_label="published-reference",
            extraction_method="text",
            source=ProvenanceSource(**{**TAFAZZOL_2024.as_dict(), "location": "Section 4.2"}),
        )
    )

    # ------------------------------------------------------------- Chari 2024 (abstract-level only)
    p = "chari-2024-pd6-monte-carlo"
    records.append(
        _rec(
            id=f"{p}.initial_position_distribution",
            profile=p,
            quantity="initial_position_distribution",
            value=[[6.0, 9.0], [3.0, 6.0], [1.0, 2.0]],
            value_text="[U(6,9), U(3,6), U(1,2)]",
            units="nondimensional",
            evidence_label="published-reference",
            extraction_method="manifest-import",
            source=CHARI_2024,
            verification_status="requires-source-verification",
            notes=(
                "imported from the campaign baselines manifest; the paper body was not retrieved "
                "during this freeze"
            ),
        )
    )

    # ------------------------------------------------------------- TOPS
    p = "esa-tops-2026"
    records.append(
        _rec(
            id=f"{p}.problem_count_pinned",
            profile=p,
            quantity="problem_count_pinned_revision",
            value=34,
            value_text="34 (5 two-body + 9 MEE + 14 CR3BP + 6 solar sail)",
            units="problems",
            evidence_label="published-reference",
            extraction_method="data-file",
            source=ProvenanceSource(**{**TOPS_REPO.as_dict(), "location": "zoh/dbs/_tops_*.json"}),
            notes=(
                "the ISSFD paper text describes a 28-problem suite; the pinned revision already "
                "contains 34"
            ),
        )
    )
    records.append(
        _rec(
            id=f"{p}.problem_count_paper",
            profile=p,
            quantity="problem_count_paper",
            value=28,
            value_text="28-problem suite",
            units="problems",
            evidence_label="published-reference",
            extraction_method="text",
            source=ProvenanceSource(
                **{**TOPS_REPO.as_dict(), "location": "README.md 'The TOPS database'"}
            ),
        )
    )

    # ------------------------------------------------------------- GTOPX
    p = "gtopx-2021"
    for benchmark, text in [
        ("cassini1", "4.930708733982513"),
        ("rosetta", "1.343367"),
        ("messenger_reduced", "8.629944278158570"),
        ("gtoc1", "-1581950.131840605288744"),
        ("cassini2", "8.382998066171544"),
        ("messenger_full", "1.957873051874192"),
        ("sagas", "18.187747259364485"),
        ("cassini1_minlp", "3.500717819978564"),
    ]:
        records.append(
            _rec(
                id=f"{p}.best_known.{benchmark}",
                profile=p,
                quantity=f"best_known_objective_{benchmark}",
                value=float(text),
                value_text=text,
                units="km/s (GTOC1: -score)",
                evidence_label="published-reference",
                extraction_method="data-file",
                source=ProvenanceSource(
                    **{
                        **GTOPX.as_dict(),
                        "location": (
                            f"solutions/{benchmark.replace('_minlp', '-minlp')}.txt f[0] line"
                        ),
                    }
                ),
                objective_convention="gtopx objective",
            )
        )

    # ------------------------------------------------------------- GTOC9 constants
    p = "gtoc9-example-validation"
    for quantity, value, text, units in [
        ("alpha", 2.0e-6, "2.0e-6", "MEUR/kg^2"),
        ("c_min", 45.0, "45", "MEUR"),
        ("c_max", 55.0, "55", "MEUR"),
        ("unremoved_debris_cost", 55.0018, "55.0018", "MEUR"),
        ("t_w", 5.0, "5", "day"),
        ("m_de", 30.0, "30", "kg"),
        ("m_dry", 2000.0, "2000", "kg"),
        ("m_p_max", 5000.0, "5000", "kg"),
        ("r_p_min", 6600000.0, "6600000", "m"),
        ("mu", 398600.4418e9, "398600.4418e9", "m^3/s^2"),
        ("J2", 1.08262668e-3, "1.08262668e-3", "dimensionless"),
        ("r_eq", 6378137.0, "6378137", "m"),
        ("Isp", 340.0, "340", "s"),
        ("g0", 9.80665, "9.80665", "m/s^2"),
        ("eps_r", 100.0, "100", "m"),
        ("eps_v", 1.0, "1", "m/s"),
        ("eps_m", 1.0e-8, "1E-8", "kg"),
        ("window_start", 23467.0, "23467", "MJD2000"),
        ("window_end", 26419.0, "26419", "MJD2000"),
    ]:
        records.append(
            _rec(
                id=f"{p}.constant.{quantity}",
                profile=p,
                quantity=quantity,
                value=value,
                value_text=text,
                units=units,
                evidence_label="published-reference",
                extraction_method="table",
                source=ProvenanceSource(
                    **{
                        **GTOC9_KELVINS.as_dict(),
                        "location": "constants and submission-format pages",
                    }
                ),
            )
        )

    # ------------------------------------------------------------- GTOC12 verifier identity
    p = "gtoc12-official-verifier"
    records.append(
        _rec(
            id=f"{p}.verifier_sha256",
            profile=p,
            quantity="verifier_binary_sha256",
            value="d4e4bc81129266420b27c9bde038bce9eda1960e7de9c695772fbfdb1cc82cd6",
            value_text="d4e4bc81129266420b27c9bde038bce9eda1960e7de9c695772fbfdb1cc82cd6",
            units="sha256",
            evidence_label="published-reference",
            extraction_method="data-file",
            source=ProvenanceSource(
                **{
                    **GTOC12_PORTAL.as_dict(),
                    "location": (
                        "GTOC12_Verification_Program.zip -> GTOC12_Verification/Linux/GTOC12_Verify"
                    ),
                }
            ),
        )
    )
    return records
