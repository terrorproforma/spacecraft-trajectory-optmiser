"""Commit only the audited completion payload; leave unrelated staged changes alone."""
from pathlib import Path
import hashlib
import json
import subprocess

ROOT = Path(__file__).resolve().parents[2]
FOLDER = ROOT / "build/performance/completion-tranche-v621"
payload = json.loads((FOLDER / "payload.json").read_text())
SHA = lambda data: hashlib.sha256(data).hexdigest()

def git(*args, **kwargs):
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, **kwargs)

before = git("rev-parse", "HEAD").stdout.decode().strip()
assert before == payload["head_at_audit"], "HEAD changed; review the newer parent before committing"
assert not (FOLDER / "commit.json").exists()
for name, record in payload["paths"].items():
    data = (ROOT / name).read_bytes()
    assert SHA(data) == record["sha256"] and len(data) == record["bytes"], name
git("add", "-f", "--pathspec-from-file=" + str(FOLDER / "paths.nul"), "--pathspec-file-nul")
request = "".join(":" + name + "\n" for name in payload["paths"]).encode()
raw = git("cat-file", "--batch", input=request).stdout
cursor = staged_bytes = 0
normalized = []
for name, record in payload["paths"].items():
    end = raw.index(b"\n", cursor)
    _, kind, size = raw[cursor:end].split()
    assert kind == b"blob"
    size = int(size)
    data = raw[end + 1:end + 1 + size]
    if SHA(data) != record["sha256"] or size != record["bytes"]:
        assert not name.startswith("results/"), name
        assert data == (ROOT / name).read_bytes().replace(b"\r\n", b"\n"), name
        normalized.append(name)
    assert raw[end + 1 + size:end + 2 + size] == b"\n"
    cursor = end + 2 + size
    staged_bytes += size
assert cursor == len(raw)
git("diff", "--cached", "--check", "--", *[p for p in payload["paths"] if not p.startswith("results/")])
assert git("rev-parse", "HEAD").stdout.decode().strip() == before
git("commit", "--quiet", "--only", "--pathspec-from-file=" + str(FOLDER / "paths.nul"),
    "--pathspec-file-nul", "-m", "Add retained CUDA route completion and verify model parity")
after = git("rev-parse", "HEAD").stdout.decode().strip()
parent = git("rev-parse", "HEAD^").stdout.decode().strip()
assert parent == before
changed = set(git("diff-tree", "--no-commit-id", "--name-only", "-r", "-z", "HEAD").stdout.decode().strip("\0").split("\0"))
assert changed == set(payload["paths"]), changed ^ set(payload["paths"])
report = {"commit": after, "parent": parent, "files": len(changed),
          "working_bytes": payload["bytes"], "staged_bytes": staged_bytes,
          "only_audited_paths_committed": True, "source_crlf_normalization_only": normalized,
          "network_actions": 0}
with (FOLDER / "commit.json").open("x") as stream:
    json.dump(report, stream, indent=2)
    stream.write("\n")
print(json.dumps(report, indent=2))
