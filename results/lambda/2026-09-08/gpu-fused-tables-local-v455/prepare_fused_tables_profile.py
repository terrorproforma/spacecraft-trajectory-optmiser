from pathlib import Path
base=Path('build/performance/profile_v446.py').read_text().replace('pipeline-profile-v446','fused-tables-profile-v450')
base=base.replace('end=float(end)))','end=float(end),epochs=np.asarray(table.epochs,dtype=np.float64).tolist(),tofs=np.asarray(tofs,dtype=np.float64).tolist()))')
Path('build/performance/profile_fused_tables_v450.py').write_text(base)
base=Path('build/performance/run_profile_v446.py').read_text().replace('pipeline-profile-v446','fused-tables-profile-v450').replace('pipeline_profile446','fused_profile450').replace('profile_v446.py','profile_fused_tables_v450.py').replace('build-spacepdhcg-earth-beam-v441','build-spacepdhcg-fused-tables-v447').replace("SPACEPDHCG_TEST_GTOC12_COMPACT_OPTIONS='1'", "SPACEPDHCG_TEST_GTOC12_FUSED_TABLES='1'")
Path('build/performance/run_fused_tables_profile_v450.py').write_text(base)
for prefix in ['profile_fused_tables_v','run_fused_tables_profile_v']:
    Path('build/performance/'+prefix+'451.py').write_text(Path('build/performance/'+prefix+'450.py').read_text().replace('450','451'))
