from pathlib import Path
p=Path('build/performance')
s=(p/'analyze_preserve_objective.py').read_text()
a=s.index("    scope='");b=s.index("',\n    lost_baseline",a)
s=s[:a]+"    scope='One full pass per mode with identical core and QOCO binaries, two objective-preserving Ruiz passes in both modes. Baseline rebuilds scaled workspaces; candidate reuses compatible ones. Original budgets and accuracy tolerances unchanged. Solver time excludes certification."+s[b:]
(p/'analyze_scaled_pool_legs.py').write_text(s)
s=(p/'analyze_preserve_campaign.py').read_text().replace('existing pool eligibility remains in force.', 'candidate enables compatible scaled workspace reuse; zero-Ruiz baseline uses existing reuse.')
(p/'analyze_scaled_pool_campaign.py').write_text(s)
s=(p/'verify_preserve_preparation.py').read_text().replace('build-qoco-preserve-objective-v534','build-qoco-scaled-pool-v540').replace('preserve-preparation-verification.json','scaled-pool-preparation-verification.json')
(p/'verify_scaled_pool_preparation.py').write_text(s)
