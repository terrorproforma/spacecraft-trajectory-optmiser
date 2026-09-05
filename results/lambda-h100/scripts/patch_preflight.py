"""Patch the literature GPU preflight so it never counts its own process as a foreign device holder.

On native Linux nvidia-smi lists compute processes; once the first target of a multi-target
`spacepdhcg literature gpu-run` opens a CUDA context, the preflight for every later target saw the
gpu-run process itself and refused ("other compute processes hold the device (pid <self>)").
WSL nvidia-smi never lists compute apps, so this was invisible on the RTX 5090 host.
"""
from pathlib import Path

root = Path("/home/ubuntu/spacepdhcg/v2")
src = root / "src/spacepdhcg/literature/gpu_preflight.py"
text = src.read_text()

old_doc = """* other compute processes are reported and, by default, also refuse (a shared device is not a
  clean measurement either) unless ``allow_shared=True``.
"""
new_doc = """* other compute processes are reported and, by default, also refuse (a shared device is not a
  clean measurement either) unless ``allow_shared=True``;
* the calling process itself is never a foreign holder: a multi-target ``gpu-run`` keeps its CUDA
  context between targets, and native-Linux ``nvidia-smi`` lists it (WSL never lists compute
  apps, which hid this self-refusal on the RTX 5090 host).
"""
assert old_doc in text
text = text.replace(old_doc, new_doc)

old_check = """    owners = [process for process in processes if process.g4_owner]
    if owners:"""
new_check = """    own_pid = os.getpid()
    foreign = [process for process in processes if process.pid != own_pid]
    owners = [process for process in foreign if process.g4_owner]
    if owners:"""
assert old_check in text
text = text.replace(old_check, new_check)

old_shared = """    if processes and not allow_shared:
        described = ", ".join(f"pid {p.pid}" for p in processes)"""
new_shared = """    if foreign and not allow_shared:
        described = ", ".join(f"pid {p.pid}" for p in foreign)"""
assert old_shared in text
text = text.replace(old_shared, new_shared)
src.write_text(text)

tests = root / "tests/test_literature_gpu_preflight.py"
ttext = tests.read_text()
anchor = "def test_preflight_passes_on_an_idle_device_and_checks_the_qoco_library(tmp_path) -> None:"
assert anchor in ttext
new_test = '''def test_preflight_ignores_its_own_process_between_targets_of_one_gpu_run(monkeypatch) -> None:
    import os

    own = os.getpid()
    monkeypatch.setattr(gpu_preflight, "_read_command_line", {own: "spacepdhcg literature gpu-run"}.get)
    result = gpu_preflight.preflight(runner=lambda command: f"{own}, python\\n")
    assert result.ok and result.reason == "device free"
    assert [process.pid for process in result.processes] == [own]
    refused = gpu_preflight.preflight(runner=lambda command: f"{own}, python\\n4242, python\\n")
    assert not refused.ok and "pid 4242" in refused.reason and f"pid {own}" not in refused.reason


'''
ttext = ttext.replace(anchor, new_test + anchor)
tests.write_text(ttext)
print("patched", src, tests)
