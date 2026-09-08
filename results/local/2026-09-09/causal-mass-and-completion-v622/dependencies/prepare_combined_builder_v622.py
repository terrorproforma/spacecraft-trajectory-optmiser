"""Generate the bounded combined-build helper from the reviewed mass-only builder."""
from pathlib import Path

folder = Path(__file__).resolve().parent
source = (folder / 'build_mass_core_v622a.py').read_text()
source = source.replace('spacepdhcg-mass-v622a', 'spacepdhcg-combined-v622a')
source = source.replace('mass-core-v622a', 'combined-core-v622a')
source = source.replace("BASE = 'fdf52ae31d259240ffebdcf79ed0965dce8b9298'", "BASE = '258e5c3aad1fd8e592527c759f1514925039cf2a'")
source = source.replace("    'cpp/cuda/tests/persistent_mass_test.cu', 'cpp/cuda/CMakeLists.txt',\n]", """    'cpp/cuda/tests/persistent_mass_test.cu', 'cpp/cuda/CMakeLists.txt',
    'cpp/cuda/include/spacepdhcg/cuda/gtoc12_completion_c_api.h',
    'cpp/cuda/src/gtoc12_completion.cu', 'cpp/cuda/src/gtoc12_completion_model.cuh',
]""")
needle = "    for name in OWNED:\n        (repo/name).parent.mkdir"
replacement = """    mass_pin=LIVE/'build/performance/mass-core-v622a/manifest.json'
    compact_pin=LIVE/'build/performance/completion-model-v622g/report.json'
    assert sha(mass_pin)=='ca970d7acd9893ba71d7e58f85393cb2a26c003ca931e2071e2718e4292ad026'
    assert sha(compact_pin)=='56a65032e639a6eb9af08052e9f3df87f5f1c84cfe9cf2ece6f3f63eafb2a12a'
    mass=json.loads(mass_pin.read_text());compact=json.loads(compact_pin.read_text())
    assert len(OWNED)==14
    for name in OWNED:
        expected=mass['owned_sha256'][name] if name in mass['owned_sha256'] else compact['owned_sources'][name]
        assert sha(LIVE/name)==expected,name
    for name in OWNED:
        (repo/name).parent.mkdir"""
assert needle in source
source = source.replace(needle, replacement)
source = source.replace("'foreign_overlays':'Excluded: compact completion, search, fleet and other working-tree changes.',", """'foreign_overlays':'Only eleven exact mass and three exact compact C++ overlays. Committed base258 fleet changes are included; other working-tree files are excluded.',
        'mass_component_manifest_sha256':sha(mass_pin),'compact_component_report_sha256':sha(compact_pin),
        'compact_host_scope':'Compile contains no Python runtime; smoke host remains separately pinned to g report and its source archive.',
        'compact_host_owned_sha256':{k:v for k,v in compact['owned_sources'].items() if not k.startswith('cpp/')},""")
source = source.replace("'persistent_mass_test','persistent_snapshot_replay','persistent_snapshot_conversion_test'", "'persistent_mass_test','gtoc12_completion_test','persistent_snapshot_replay','persistent_snapshot_conversion_test'")
source = source.replace("    conversion=build_dir/'cuda/persistent_snapshot_conversion_test'", "    completion=build_dir/'cuda-tests/gtoc12_completion_test'\n    conversion=build_dir/'cuda/persistent_snapshot_conversion_test'")
source = source.replace("('conversion_test',conversion)]", "('conversion_test',conversion),('completion_test',completion)]")
source = source.replace("call('cpu-snapshot',[str(conversion)])", "call('cpu-snapshot',[str(conversion)]);call('cpu-completion',[str(completion),'--cpu-only'])")
source = source.replace("'-R','persistent_mass_test'", "'-R','persistent_mass_test|gtoc12_completion_test'")
source = source.replace("for name in ('spacepdhcg_cuda_workspace_set_mass_options','spacepdhcg_cuda_workspace_mass_diagnostics'):", "for name in ('spacepdhcg_cuda_workspace_set_mass_options','spacepdhcg_cuda_workspace_mass_diagnostics','spacepdhcg_gtoc12_completion_evaluate_host','spacepdhcg_gtoc12_completion_evaluate_compact_host'):")
compile(source, 'build_combined_core_v622a.py', 'exec')
output = folder / 'build_combined_core_v622a.py'
assert not output.exists()
output.write_text(source)
print(output)
