from pathlib import Path
p=Path('build/performance')
s=(p/'analyze_conditioning_legs.py').read_text()
s=s.replace('zero versus two Ruiz iterations.', 'zero Ruiz versus two Ruiz passes with objective magnitude preserved.')
(p/'analyze_preserve_objective.py').write_text(s)
s=(p/'analyze_conditioning_campaign.py').read_text()
s=s.replace('Zero versus two Ruiz iterations;', 'Zero Ruiz versus two Ruiz passes with objective magnitude preserved;')
(p/'analyze_preserve_campaign.py').write_text(s)
