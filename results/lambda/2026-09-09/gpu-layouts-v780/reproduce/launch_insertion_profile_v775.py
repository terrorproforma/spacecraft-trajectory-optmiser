from pathlib import Path
p=Path('build/performance')
s=(p/'bench_insertions_v769.py').read_text().replace('spacepdhcg-insertions-v768','spacepdhcg-insertions-v772')
s=s.replace('import dataclasses,','import cProfile,pstats,dataclasses,')
s=s.replace('range(6)','range(1)').replace("modes=['scalar','native'] if repeat%2==0 else ['native','scalar']", "modes=['native']")
s=s.replace('started=time.perf_counter()', 'prof=cProfile.Profile();prof.enable();started=time.perf_counter()')
s=s.replace('seconds=time.perf_counter()-started', "seconds=time.perf_counter()-started;prof.disable();prof.dump_stats(str(root/(str(identifier)+'.prof')))\n                        ps=pstats.Stats(prof);report.setdefault('profiles',{})[str(identifier)]=[dict(file=k[0],line=k[1],name=k[2],primitive_calls=v[0],calls=v[1],own_seconds=v[2],cumulative_seconds=v[3]) for k,v in sorted(ps.stats.items(),key=lambda x:-x[1][3])[:45]]")
s=s.replace("                    assert outputs['scalar']==outputs['native'],outputs",'')
s=s[:s.index("    report['medians']")]+"    report['success']=True\nexcept BaseException:report['error']=traceback.format_exc()\nreport['complete']=True;save()\n"
(p/'profile_insertions_v775.py').write_text(s)
launch=(p/'launch_insertions_bench_v769.py').read_text().replace('spacepdhcg-insertions-v769','spacepdhcg-insertions-v775').replace("script=Path('build/performance/bench_insertions_v769.py').read_text().replace('spacepdhcg-insertions-v768','spacepdhcg-insertions-v770')", "script=Path('build/performance/profile_insertions_v775.py').read_text()")
launch=launch.replace('spacepdhcg-insertions-v770/final','spacepdhcg-insertions-v772/final')
exec(compile(launch,__file__,'exec'))
