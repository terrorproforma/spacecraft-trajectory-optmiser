from pathlib import Path
p=Path('build/performance')
s=(p/'analyze_scaled_pool_legs.py').read_text()
a=s.index("    scope='");b=s.index("',\n    lost_baseline",a)
s=s[:a]+"    scope='One full pass per mode with identical binaries and production zero-Ruiz workspace reuse. Only device initialization differs. Original budgets and accuracy tolerances unchanged. Solver time excludes certification."+s[b:]
(p/'analyze_device_initialization_legs.py').write_text(s)
s=(p/'analyze_scaled_pool_campaign.py').read_text().replace("(2 if c['candidate'] else 0)",'0')
a=s.index("output = dict(scope='");b=s.index("',\n",a)
s=s[:a]+"output = dict(scope='ABBA two runs per mode with identical binaries and production zero-Ruiz workspace reuse. Only device initialization differs. Both mission checkers must pass."+s[b:]
(p/'analyze_device_initialization_campaign.py').write_text(s)
