from pathlib import Path
base=Path('build/performance/profile_search_v419.py').read_text().replace('search-profile-v419','compact-profile-v427').replace("['screen_hops','paired_hops','screen_elements','_prepare']", "['screen_hops','paired_hops','paired_options','screen_elements','_prepare']")
Path('build/performance/profile_compact_v427.py').write_text(base)
base=Path('build/performance/run_search_profile_v419.py').read_text().replace('search-profile-v419','compact-profile-v427').replace('build-spacepdhcg-stationary-v411','build-spacepdhcg-compact-options-v423').replace('profile_search_v419.py','profile_compact_v427.py').replace('search_profile419','compact_profile427')
base=base.replace("OPENBLAS_NUM_THREADS='1'", "SPACEPDHCG_TEST_GTOC12_COMPACT_OPTIONS='1',OPENBLAS_NUM_THREADS='1'")
base=base.replace("r=dict(pid=", "r=dict(source_sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in ['src/spacepdhcg/gtoc12/search.py','src/spacepdhcg/gtoc12/gpu_lambert.py','src/spacepdhcg/gtoc12/lambert.py']},pid=")
Path('build/performance/run_compact_profile_v427.py').write_text(base)
