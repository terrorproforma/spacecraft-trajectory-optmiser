from pathlib import Path

tests = Path("/home/ubuntu/spacepdhcg/v2/tests/test_literature_gpu_preflight.py")
t = tests.read_text()
old = '    monkeypatch.setattr(gpu_preflight, "_read_command_line", {own: "spacepdhcg literature gpu-run"}.get)\n'
new = '    monkeypatch.setattr(gpu_preflight, "_read_command_line", {own: "spacepdhcg literature"}.get)\n'
assert old in t
tests.write_text(t.replace(old, new))
print("ok")
