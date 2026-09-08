from pathlib import Path
p=Path('build/performance')
profile=(p/'profile_v446.py').read_text().replace('pipeline-profile-v446','pipeline-profile-v492')
profile=profile.replace('gpu_scvx,gpu_beam','gpu_scvx,gpu_beam,gpu_retime,retiming,verifier,low_thrust')
a=profile.index('native_init=');b=profile.index('for cls in ',a)
profile=profile[:a]+profile[b:]
profile=profile.replace('gpu_collection.GpuCollection]:','gpu_collection.GpuCollection,gpu_retime.GpuRetime,retiming.Retimer,verifier.Gtoc12Verifier]:')
profile=profile.replace("start=time.perf_counter()\ntry:runpy", "low_thrust.certify_leg=timed('Verification.certify_leg',low_thrust.certify_leg)\nstart=time.perf_counter()\ntry:runpy")
profile=profile.replace(' Grid trace hashes immutable elements and axes, excluding the subsequent end-date mask.',' Includes independent verification timers; no extra CUDA synchronization is inserted.')
(p/'profile_v492.py').write_text(profile)
run=(p/'run_profile_v446.py').read_text().replace('pipeline-profile-v446','pipeline-profile-v492').replace('profile_v446.py','profile_v492.py').replace('pipeline_profile446','pipeline_profile492').replace('build-spacepdhcg-earth-beam-v441','build-spacepdhcg-workspace-pool-v490')
run=run.replace("SPACEPDHCG_TEST_GTOC12_COMPACT_OPTIONS='1',",'')
run=run.replace("r=dict(source_sha256=", "r=dict(source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),source_sha256=")
(p/'run_profile_v492.py').write_text(run)
