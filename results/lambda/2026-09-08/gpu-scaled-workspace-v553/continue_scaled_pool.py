from pathlib import Path
p=Path('build/performance')
s=(p/'status_scaled_pool_v541.py').read_text().replace('v541','v545').replace('v544','v548')
(p/'status_scaled_pool_v545.py').write_text(s)
s=(p/'validate_scaled_pool_v542.py').read_text().replace("root=Path('build/performance/scaled-pool-v542')", "root=Path('build/performance/scaled-pool-v547')")
start=s.index("  run('pytest'")
end=s.index('  replay=',start)
s=s[:start]+s[end:]
s=s.replace("'build/performance/validate_scaled_pool_v542.py'", "'build/performance/validate_scaled_pool_v547.py'")
(p/'validate_scaled_pool_v547.py').write_text(s)
s=(p/'launch_scaled_pool_campaign_v544.py').read_text().replace('v541','v545').replace('v544','v548').replace('scaled543','scaled548')
(p/'launch_scaled_pool_campaign_v548.py').write_text(s)
