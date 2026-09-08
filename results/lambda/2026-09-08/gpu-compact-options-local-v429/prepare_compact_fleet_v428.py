from pathlib import Path
base=Path('build/performance/run_paired_fleet_v418.py').read_text().replace('paired-fleet-v418','compact-fleet-v428').replace('build-spacepdhcg-stationary-v411','build-spacepdhcg-compact-options-v423').replace('paired_fleet418','compact_fleet428').replace("SPACEPDHCG_TEST_GTOC12_PAIRED_EPHEMERIDES='1'", "SPACEPDHCG_TEST_GTOC12_COMPACT_OPTIONS='1'")
base=base.replace("r=dict(pid=", "r=dict(source_sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in ['src/spacepdhcg/gtoc12/search.py','src/spacepdhcg/gtoc12/gpu_lambert.py','src/spacepdhcg/gtoc12/lambert.py']},pid=")
Path('build/performance/run_compact_fleet_v428.py').write_text(base)
