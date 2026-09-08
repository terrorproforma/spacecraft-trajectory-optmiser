from pathlib import Path
p=Path('build/performance')
run=(p/'run_profile_v492.py').read_text().replace('pipeline-profile-v492','resident-options-capture-v493').replace('profile_v492.py','capture_resident_options_v493.py').replace('pipeline_profile492','resident_options_capture493')
(p/'run_resident_options_v493.py').write_text(run)
